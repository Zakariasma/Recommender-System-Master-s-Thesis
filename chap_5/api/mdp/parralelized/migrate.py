# chap_5/api/mdp/scripts/migrate_sqlite_to_postgres.py
import os
import json
import sqlite3
import pickle
import time
import numpy as np

from chap_5.api.mdp.predictive_model.state_key import encode
from chap_5.api.mdp.config import LIST_SIZE, DATABASE_URL
from chap_5.api.mdp.helper.kv_store import KeyValueStore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "models")

STATES_CACHE = "observed_states_cache.pkl"
SOLVER_STATE_DB = os.path.join(DATA_DIR, "solver_state.db")


def migrate():
    print("Chargement des états depuis le cache...")
    with open(STATES_CACHE, "rb") as f:
        states = pickle.load(f)
    n_states = len(states)
    print(f"  → {n_states:,} états à migrer")

    print("Lecture du checkpoint SQLite...")
    conn = sqlite3.connect(SOLVER_STATE_DB, timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")

    row = conn.execute("SELECT value FROM solver_checkpoint WHERE key='n_states'").fetchone()
    if not row:
        print("Erreur : aucun checkpoint trouvé dans SQLite.")
        conn.close()
        return
    n_states_db = int(row[0])
    if n_states_db != n_states:
        print(f"Erreur : n_states mismatch (cache={n_states}, sqlite={n_states_db})")
        conn.close()
        return

    V_blob = conn.execute("SELECT value FROM solver_checkpoint WHERE key='V'").fetchone()[0]
    pi_blob = conn.execute("SELECT value FROM solver_checkpoint WHERE key='policy_items'").fetchone()[0]
    ps_blob = conn.execute("SELECT value FROM solver_checkpoint WHERE key='policy_scores'").fetchone()[0]
    conn.close()

    # Reconstruction des tableaux numpy
    V_np = np.frombuffer(V_blob, dtype=np.float64)
    pi_np = np.frombuffer(pi_blob, dtype=np.int32).reshape(n_states, LIST_SIZE)
    ps_np = np.frombuffer(ps_blob, dtype=np.float32).reshape(n_states, LIST_SIZE)

    print("Connexion à PostgreSQL...")
    v_store = KeyValueStore(DATABASE_URL, namespace="V")
    policy_store = KeyValueStore(DATABASE_URL, namespace="policy")

    # Optionnel : nettoyer les namespaces existants avant d'écrire
    print("Nettoyage des namespaces existants (V, policy)...")
    try:
        v_store.clear()
        policy_store.clear()
    except AttributeError:
        # Si KeyValueStore n'a pas de méthode clear(), on ignore
        print("  (méthode clear() indisponible, skip)")

    print("Migration V + policy vers PostgreSQL...")
    BATCH_SIZE = 5000
    batch_v = {}
    batch_p = {}
    start = time.time()

    for i, s in enumerate(states):
        key = encode(s)

        # V value — mdp_solver écrit un float via json.dumps implicitement via KeyValueStore
        batch_v[key] = float(V_np[i])

        # Policy value — mdp_solver écrit [(item, score), ...] (liste de tuples)
        policy_list = []
        for j in range(LIST_SIZE):
            if pi_np[i, j] >= 0:
                policy_list.append([int(pi_np[i, j]), float(ps_np[i, j])])
        batch_p[key] = policy_list

        if (i + 1) % BATCH_SIZE == 0:
            v_store.set_many(batch_v)
            policy_store.set_many(batch_p)
            batch_v.clear()
            batch_p.clear()
            elapsed = time.time() - start
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_states - (i + 1)) / speed / 60 if speed > 0 else 0
            print(f"\r  {i+1:,}/{n_states:,} | {speed:,.0f} états/s | ETA: {eta:.1f} min",
                  end="", flush=True)

    if batch_v:
        v_store.set_many(batch_v)
        policy_store.set_many(batch_p)

    v_store.close() if hasattr(v_store, "close") else None
    policy_store.close() if hasattr(policy_store, "close") else None

    print(f"\n\nMigration terminée ! {n_states:,} états écrits dans PostgreSQL.")
    print(f"  Namespace 'V'      : valeurs V(s)")
    print(f"  Namespace 'policy' : listes [[item, score], ...]")


if __name__ == "__main__":
    migrate()