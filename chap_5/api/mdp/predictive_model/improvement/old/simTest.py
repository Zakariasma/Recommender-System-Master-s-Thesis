import random
import statistics
import time
from collections import defaultdict
from sqlalchemy import create_engine

from chap_5.api.mdp.config import DATABASE_URL

K = 3
SAMPLE_SIZE = 1000

MOVIE_TO_GENRES = {}


def load_movie_genres():
    """
    Charge le mapping movie_id -> bitmask des genres (int).
    Chaque bit correspond à un genre_id : bit i = genre_id i.
    """
    print("Chargement des genres depuis la base de données...")
    engine = create_engine(DATABASE_URL)
    raw_conn = engine.raw_connection()

    mapping = defaultdict(int)
    try:
        with raw_conn.cursor() as cur:
            cur.execute("SELECT movie_id, genre_id FROM movie_genre")
            for movie_id, genre_id in cur:
                mapping[int(movie_id)] |= (1 << int(genre_id))
    finally:
        raw_conn.close()

    print(f"  -> {len(mapping):,} films chargés avec leurs genres.")
    return dict(mapping)


def submasks_minus_one(mask):
    """
    Génère tous les sous-masques obtenus en retirant exactement un bit.
    """
    m = mask
    while m:
        bit = m & -m          # isole le bit le plus bas
        yield mask ^ bit      # masque sans ce bit
        m ^= bit


def get_mask_for_state(state_str):
    """
    Renvoie le bitmask des genres des films composant l'état.
    state_str : ex "123,456,789"
    """
    mask = 0
    for mid in state_str.split(','):
        try:
            mask |= MOVIE_TO_GENRES.get(int(mid), 0)
        except ValueError:
            continue
    return mask


class SimilarityTestModel:
    def __init__(self, k: int):
        self.k = k

    def fit(self):
        print("=== Test de l'indexation par similarité (compteurs légers, bitmask) ===")
        print(f"k        : {self.k}")

        global MOVIE_TO_GENRES
        MOVIE_TO_GENRES = load_movie_genres()

        self._build_index_and_test()

    def _build_index_and_test(self):
        print("\nExtraction des états et construction de l'index...")
        engine = create_engine(DATABASE_URL)
        raw_conn = engine.raw_connection()

        try:
            with raw_conn.cursor(name='stream_states_cursor') as cur:
                cur.itersize = 10000
                cur.execute("SELECT DISTINCT s FROM transitions")

                inverted_index = defaultdict(int)

                total_states = 0
                max_bucket_size = 0

                sample_states = []

                start = time.perf_counter()

                for (state_str,) in cur:
                    total_states += 1

                    if len(sample_states) < SAMPLE_SIZE:
                        sample_states.append(state_str)
                    else:
                        idx = random.randint(0, total_states - 1)
                        if idx < SAMPLE_SIZE:
                            sample_states[idx] = state_str

                    mask = get_mask_for_state(state_str)

                    if mask.bit_count() <= 1:
                        inverted_index[mask] += 1
                        if inverted_index[mask] > max_bucket_size:
                            max_bucket_size = inverted_index[mask]
                    else:
                        for sub in submasks_minus_one(mask):
                            inverted_index[sub] += 1
                            if inverted_index[sub] > max_bucket_size:
                                max_bucket_size = inverted_index[sub]

                    if total_states % 500000 == 0:
                        elapsed = time.perf_counter() - start
                        print(
                            f"\r  Traitement... {total_states:,} états. "
                            f"Vitesse: {total_states / elapsed:,.0f} états/s. "
                            f"Clés: {len(inverted_index):,}",
                            end="", flush=True)

                print(f"\r  Traitement terminé : {total_states:,} états en "
                      f"{time.perf_counter() - start:.1f}s.      ")

            print("\n=== Statistiques de l'index ===")
            print(f"  - Total d'états distincts lus      : {total_states:,}")
            print(f"  - Nombre de clés (sous-ensembles)  : {len(inverted_index):,}")
            print(f"  - Taille max d'un panier (compteur): {max_bucket_size:,}")

            print("\n=== Simulation des requêtes de similarité ===")
            match_counts = []

            for state_str in sample_states:
                mask = get_mask_for_state(state_str)

                state_matches = 0

                if mask.bit_count() <= 1:
                    state_matches = inverted_index.get(mask, 0)
                else:
                    for sub in submasks_minus_one(mask):
                        state_matches += inverted_index.get(sub, 0)

                match_counts.append(state_matches)

            total_raw_matches = sum(match_counts)

            avg_matches = (
                total_raw_matches / len(match_counts)
                if match_counts else 0
            )

            median_matches = (
                statistics.median(match_counts)
                if match_counts else 0
            )

            min_matches = min(match_counts) if match_counts else 0
            max_matches = max(match_counts) if match_counts else 0

            print(f"  - Requêtes simulées                 : {len(match_counts):,}")
            print(f"  - Nombre moyen de candidats bruts   : {avg_matches:,.0f}")
            print(f"  - Nombre médian de candidats bruts  : {median_matches:,.0f}")
            print(f"  - Minimum de candidats bruts        : {min_matches:,}")
            print(f"  - Maximum de candidats bruts        : {max_matches:,}")

            print("\n  -> La moyenne peut être tirée vers le haut par quelques paniers très gros.")
            print("  -> La médiane donne une meilleure idée du coût typique d'une requête.")

        finally:
            raw_conn.close()


if __name__ == "__main__":
    model = SimilarityTestModel(k=K)
    model.fit()