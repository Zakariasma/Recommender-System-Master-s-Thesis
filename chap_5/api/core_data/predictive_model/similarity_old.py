import sqlite3
import os
import time
from collections import defaultdict
from ast import literal_eval

MODEL_DIR   = "../data/model"
DB_SKIP     = os.path.join(MODEL_DIR, "transitions_skipping.db")
DB_SIM      = os.path.join(MODEL_DIR, "transitions_similarity.db")

OBS_THRESHOLD = 2
K             = 3
BATCH_SIZE    = 10_000
LOG_EVERY     = 100


class SimilarityModel:

    def __init__(self, db_path_movies=None):
        self.db_path_movies = db_path_movies or os.getenv(
            'DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/master'
        )
        self.movie_genres = {}

    def _load_genre_mapping(self):
        import pandas as pd
        from sqlalchemy import create_engine

        engine = create_engine(self.db_path_movies)
        with engine.connect() as conn:
            mg = pd.read_sql("SELECT movie_id, genre_id FROM movie_genre", conn)
        engine.dispose()

        mapping = defaultdict(set)
        for row in mg.itertuples(index=False):
            mapping[row.movie_id].add(row.genre_id)
        return mapping

    def _get_signature(self, s_tuple):
        sig = []
        for movie_id in s_tuple:
            genres = self.movie_genres.get(movie_id, None)
            if not genres:
                return None  # film sans genre → on exclut de l'index
            sig.append(frozenset(genres))
        return tuple(sig)

    def _sim(self, s_tuple, si_tuple):
        score = 0
        for m, (a, b) in enumerate(zip(s_tuple, si_tuple)):
            if a == b:
                score += (m + 1)
        return score

    def _build_signature_index(self, all_states):
        index = defaultdict(list)
        skipped = 0
        for s_str in all_states:
            s_tuple = literal_eval(s_str)
            sig = self._get_signature(s_tuple)
            if sig is None:
                skipped += 1
                continue
            index[sig].append(s_str)
        print(f"  États exclus (film sans genre) : {skipped}")
        return index

    def fit(self):
        conn_skip = sqlite3.connect(DB_SKIP)
        conn_sim  = sqlite3.connect(DB_SIM)
        self._init_db(conn_sim)

        print("Chargement des transitions skipping...")
        tr_old = defaultdict(dict)
        for s, s_, prob in conn_skip.execute("SELECT s, s_, prob FROM transitions"):
            tr_old[s][s_] = prob

        all_states = list(tr_old.keys())
        print(f"  États distincts : {len(all_states)}")

        sparse_states = set()
        for (s,) in conn_skip.execute(
            "SELECT s FROM state_obs WHERE obs_count < ?", (OBS_THRESHOLD,)
        ):
            sparse_states.add(s)
        n_sparse = len(sparse_states)
        print(f"  États peu observés (obs < {OBS_THRESHOLD}) : {n_sparse}")

        conn_skip.close()

        print("Chargement du mapping film → genres...")
        self.movie_genres = self._load_genre_mapping()
        print(f"  Films avec genres : {len(self.movie_genres)}")

        print("Construction de l'index par signature de genres...")
        sig_index = self._build_signature_index(all_states)
        n_sig = len(sig_index)
        print(f"  Signatures distinctes : {n_sig}")
        avg_size = sum(len(v) for v in sig_index.values()) / max(n_sig, 1)
        print(f"  Taille moyenne d'une classe : {avg_size:.1f} états")

        # Étape 1 : états peu observés
        rows = []
        start_time = time.perf_counter()

        for idx, s in enumerate(sparse_states):
            s_tuple = literal_eval(s)
            sig = self._get_signature(s_tuple)

            if sig is None:
                # film sans genre → copie directe
                if s in tr_old:
                    total = sum(tr_old[s].values())
                    for s_, prob in tr_old[s].items():
                        rows.append((s, s_, prob / total))
            else:
                candidates = [si for si in sig_index.get(sig, []) if si != s]

                simcount = defaultdict(float)
                for si in candidates:
                    si_tuple = literal_eval(si)
                    weight = self._sim(s_tuple, si_tuple)
                    if weight == 0:
                        continue
                    for s_, prob in tr_old[si].items():
                        simcount[s_] += weight * prob

                if not simcount:
                    # aucun voisin utile → copie directe
                    if s in tr_old:
                        total = sum(tr_old[s].values())
                        for s_, prob in tr_old[s].items():
                            rows.append((s, s_, prob / total))
                else:
                    total_sim = sum(simcount.values())
                    simcount_norm = {s_: v / total_sim for s_, v in simcount.items()}

                    all_s_ = set(tr_old.get(s, {}).keys()) | set(simcount_norm.keys())
                    mixed = {}
                    for s_ in all_s_:
                        prob_old = tr_old.get(s, {}).get(s_, 0.0)
                        prob_sim = simcount_norm.get(s_, 0.0)
                        mixed[s_] = 0.5 * prob_old + 0.5 * prob_sim

                    total_mix = sum(mixed.values())
                    for s_, prob in mixed.items():
                        rows.append((s, s_, prob / total_mix))

            if len(rows) >= BATCH_SIZE:
                self._flush(conn_sim, rows)
                rows.clear()

            if (idx + 1) % LOG_EVERY == 0:
                self._log_progress("similarity", idx + 1, n_sparse, start_time)

        if rows:
            self._flush(conn_sim, rows)
            rows.clear()

        print(f"\r  [similarity] {n_sparse}/{n_sparse} (100.0%) — terminé" + " " * 20)

        # Étape 2 : états bien observés
        dense_states = [s for s in all_states if s not in sparse_states]
        n_dense = len(dense_states)
        print(f"  États bien observés à copier : {n_dense}")

        start_time = time.perf_counter()
        for idx, s in enumerate(dense_states):
            for s_, prob in tr_old[s].items():
                rows.append((s, s_, prob))
                if len(rows) >= BATCH_SIZE:
                    self._flush(conn_sim, rows)
                    rows.clear()

            if (idx + 1) % LOG_EVERY == 0:
                self._log_progress("copie", idx + 1, n_dense, start_time)

        if rows:
            self._flush(conn_sim, rows)
            rows.clear()

        print(f"\r  [copie] {n_dense}/{n_dense} (100.0%) — terminé" + " " * 20)

        conn_sim.close()
        print("Modèle similarity sauvegardé.")

    def _log_progress(self, label, done, total, start_time):
        elapsed = time.perf_counter() - start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        reste = (total - done) / rate if rate > 0 else 0.0
        print(
            f"\r  [{label}] {done}/{total} "
            f"({100 * done / total:.1f}%) "
            f"— {rate:.1f} états/s "
            f"— reste ~{reste / 60:.1f} min",
            end="", flush=True
        )

    def _flush(self, conn, rows):
        conn.executemany(
            "INSERT INTO transitions (s, s_, prob) VALUES (?, ?, ?)", rows
        )
        conn.commit()

    def _init_db(self, conn):
        conn.execute("DROP TABLE IF EXISTS transitions")
        conn.execute("""CREATE TABLE transitions (
            s    TEXT NOT NULL,
            s_   TEXT NOT NULL,
            prob REAL NOT NULL,
            PRIMARY KEY (s, s_))""")
        conn.commit()

    @classmethod
    def load(cls) -> 'SimilarityModel':
        return cls()

    def get_trmc(self, s: tuple, s_: tuple) -> float:
        s  = tuple(int(x) for x in s)
        s_ = tuple(int(x) for x in s_)
        conn = sqlite3.connect(DB_SIM)
        row = conn.execute(
            "SELECT prob FROM transitions WHERE s = ? AND s_ = ?",
            (str(s), str(s_))
        ).fetchone()
        conn.close()
        return row[0] if row else 0.0


if __name__ == "__main__":
    model = SimilarityModel()
    model.fit()

    a = (8313, 207, 16668)
    b = (207, 16668, 5869)
    print(model.get_trmc(a, b))