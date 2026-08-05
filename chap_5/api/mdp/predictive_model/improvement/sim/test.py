import time
import sqlite3
import random
import os

from chap_5.api.mdp.predictive_model.helper.encoder import encode, decode

SQLITE_TRANSITIONS_PATH = "pg_transitions.sqlite"


def ensure_index():
    """S'assure que l'index existe pour accélérer les requêtes."""
    print("Vérification/Création de l'index sur SQLite...")
    conn = sqlite3.connect(SQLITE_TRANSITIONS_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_transitions_s_k'")
    if not cur.fetchone():
        print("Index absent. Création en cours...")
        cur.execute("CREATE INDEX idx_transitions_s_k ON transitions(s, k);")
        conn.commit()
        print("Index créé avec succès !")
    else:
        print("L'index existe déjà.")
    conn.close()


def test_sqlite_speed():
    print("\n=== Test de vitesse sur SQLite (RAM Max + Fetch Optimisé) ===")

    # 1. Récupérer 10 000 états depuis sim_index.sqlite
    conn_idx = sqlite3.connect("sim_index.sqlite")
    cur_idx = conn_idx.cursor()
    cur_idx.execute("SELECT state FROM state_genres")
    all_blobs = cur_idx.fetchall()
    conn_idx.close()

    sample_blobs = random.sample(all_blobs, 50_000)
    states_to_test = [decode(blob[0]) for blob in sample_blobs]
    unique_keys = list(set(encode(state_tuple) for state_tuple in states_to_test))

    print(f"{len(states_to_test):,} états prêts à être requêtés.")

    # 2. Requêter la base SQLite avec la configuration ultime
    sqlite_conn = sqlite3.connect(SQLITE_TRANSITIONS_PATH)

    # --- OPTIMISATIONS ULTIMES ---
    # On désactive le journal
    sqlite_conn.execute("PRAGMA journal_mode = OFF;")
    # On alloue 1 Go de cache RAM à SQLite (au lieu de 2 Mo par défaut)
    sqlite_conn.execute("PRAGMA cache_size = -1000000;")
    # On force les opérations temporaires en RAM
    sqlite_conn.execute("PRAGMA temp_store = MEMORY;")
    # On mappe le fichier de la base directement en mémoire virtuelle (accélére massivement la lecture de l'index)
    sqlite_conn.execute("PRAGMA mmap_size = 268435456;")  # 256 MB

    cur = sqlite_conn.cursor()

    start_time = time.perf_counter()
    total_transitions = 0

    # CHUNK_SIZE à 999 (limite safe de SQLite pour les variables bind)
    CHUNK_SIZE = 999
    for i in range(0, len(unique_keys), CHUNK_SIZE):
        chunk = unique_keys[i:i + CHUNK_SIZE]
        placeholders = ",".join("?" for _ in chunk)
        query = f"SELECT s_ FROM transitions WHERE s IN ({placeholders}) AND k = 3"

        cur.execute(query, chunk)

        # Au lieu de fetchall() + boucle for, on compte les lignes à la volée.
        # C'est beaucoup plus rapide car ça évite la création d'une grande liste en Python.
        total_transitions += sum(1 for _ in cur)

    elapsed = time.perf_counter() - start_time
    sqlite_conn.close()

    print(f"\n=== Résultats ===")
    print(f"Temps écoulé         : {elapsed:.4f}s")
    print(f"États requêtés       : {len(states_to_test):,}")
    print(f"Transitions trouvées : {total_transitions:,}")

    if elapsed > 0:
        speed = len(states_to_test) / elapsed
        print(f"Vitesse              : {speed:,.0f} états/s")


if __name__ == "__main__":
    ensure_index()
    test_sqlite_speed()