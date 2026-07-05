import os
import json
import sqlite3
import pickle
import numpy as np
from chap_5.api.mdp.predictive_model.state_key import encode
from chap_5.api.mdp.config import LIST_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "models")

STATES_CACHE = "observed_states_cache.pkl"
SOLVER_STATE_DB = os.path.join(DATA_DIR, "solver_state.db")
OUTPUT_DB = os.path.join(DATA_DIR, "solver_policy_readable.db")


def extract():
    print("Chargement des états...")
    with open(STATES_CACHE, "rb") as f:
        states = pickle.load(f)

    print("Lecture du checkpoint binaire...")
    conn = sqlite3.connect(SOLVER_STATE_DB, timeout=30.0)

    V_blob = conn.execute("SELECT value FROM solver_checkpoint WHERE key='V'").fetchone()[0]
    pi_blob = conn.execute("SELECT value FROM solver_checkpoint WHERE key='policy_items'").fetchone()[0]
    ps_blob = conn.execute("SELECT value FROM solver_checkpoint WHERE key='policy_scores'").fetchone()[0]
    n_states = int(conn.execute("SELECT value FROM solver_checkpoint WHERE key='n_states'").fetchone()[0])
    conn.close()

    if n_states != len(states):
        print(f"Erreur: Le nombre d'états dans le checkpoint ({n_states}) ne correspond pas au cache ({len(states)})")
        return

    # Reconstruit les tableaux numpy
    V_np = np.frombuffer(V_blob, dtype=np.float64)
    pi_np = np.frombuffer(pi_blob, dtype=np.int32).reshape(n_states, LIST_SIZE)
    ps_np = np.frombuffer(ps_blob, dtype=np.float32).reshape(n_states, LIST_SIZE)

    print(f"Création de la base de données lisible: {OUTPUT_DB}")
    if os.path.exists(OUTPUT_DB):
        os.remove(OUTPUT_DB)

    out_conn = sqlite3.connect(OUTPUT_DB, timeout=30.0)
    out_conn.execute("PRAGMA journal_mode = WAL")
    out_conn.execute("PRAGMA synchronous = NORMAL")
    out_conn.execute("""
                     CREATE TABLE IF NOT EXISTS kv_store
                     (
                         namespace
                         TEXT
                         NOT
                         NULL,
                         key
                         TEXT
                         NOT
                         NULL,
                         value
                         TEXT
                         NOT
                         NULL,
                         PRIMARY
                         KEY
                     (
                         namespace,
                         key
                     )
                         )
                     """)

    print("Insertion des valeurs V et de la politique...")
    batch_v = []
    batch_p = []
    BATCH_SIZE = 10000

    for i, s in enumerate(states):
        key = encode(s)

        # V value
        batch_v.append(('V', key, json.dumps(float(V_np[i]))))

        # Policy value
        policy_list = []
        for j in range(LIST_SIZE):
            if pi_np[i, j] >= 0:
                policy_list.append([int(pi_np[i, j]), float(ps_np[i, j])])
        batch_p.append(('policy', key, json.dumps(policy_list)))

        if (i + 1) % BATCH_SIZE == 0:
            out_conn.executemany("INSERT OR REPLACE INTO kv_store (namespace, key, value) VALUES (?, ?, ?)", batch_v)
            out_conn.executemany("INSERT OR REPLACE INTO kv_store (namespace, key, value) VALUES (?, ?, ?)", batch_p)
            out_conn.commit()
            batch_v.clear()
            batch_p.clear()
            print(f"\r  {i + 1:,}/{n_states:,} états exportés", end="", flush=True)

    if batch_v:
        out_conn.executemany("INSERT OR REPLACE INTO kv_store (namespace, key, value) VALUES (?, ?, ?)", batch_v)
        out_conn.executemany("INSERT OR REPLACE INTO kv_store (namespace, key, value) VALUES (?, ?, ?)", batch_p)
        out_conn.commit()

    out_conn.close()
    print("\nExportation terminée avec succès !")


if __name__ == "__main__":
    extract()