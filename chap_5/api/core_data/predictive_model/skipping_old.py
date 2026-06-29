import numpy as np
import sqlite3
import os
from collections import defaultdict

sqlite3.register_adapter(np.int32,  int)
sqlite3.register_adapter(np.int64,  int)
sqlite3.register_adapter(np.uint32, int)
sqlite3.register_adapter(np.uint64, int)

NPY_DIR      = "../data/npy"
FLAT_PATH    = os.path.join(NPY_DIR, "sequences_flat.npy")
OFFSETS_PATH = os.path.join(NPY_DIR, "sequences_offsets.npy")
MODEL_DIR    = "../data/model"
DB_PATH      = os.path.join(MODEL_DIR, "transitions_skipping.db")

K          = 3
MAX_SKIP   = 5
BATCH_SIZE = 10_000


class MarkovModelSkipping:

    def __init__(self, k: int = K):
        self.k = k

    def fit(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        flat    = np.load(FLAT_PATH,    mmap_mode='r')
        offsets = np.load(OFFSETS_PATH, mmap_mode='r')
        #total   = len(offsets) - 1
        total = (len(offsets) - 1) // 15  # ← 1/10 du dataset
        print(f"Séquences totales : {total}\n")

        conn = sqlite3.connect(DB_PATH)
        self._init_db(conn)

        counts     = defaultdict(lambda: defaultdict(float))
        obs_counts = defaultdict(int)

        for i in range(total):
            seq = flat[offsets[i]:offsets[i+1]]
            self._process_sequence(seq, counts, obs_counts)

            if (i + 1) % BATCH_SIZE == 0:
                self._flush(conn, counts, obs_counts)
                counts.clear()
                obs_counts.clear()
                pct = 100 * (i + 1) / total
                print(f"\r  {i+1}/{total} ({pct:.1f}%) — {total - i - 1} restantes", end="", flush=True)

        if counts:
            self._flush(conn, counts, obs_counts)

        print(f"\r  {total}/{total} (100.0%) — terminé               ")
        print("Normalisation en cours...")
        self._normalize(conn)
        conn.close()
        print("Modèle sauvegardé.")

    def _process_sequence(self, seq, counts, obs_counts):
        k = self.k
        for pos in range(len(seq) - k):
            s = str(tuple(int(x) for x in seq[pos:pos + k]))

            # Transition directe : s' = (seq[pos+1], ..., seq[pos+k])
            if pos + k < len(seq):
                s_ = str(tuple(int(x) for x in seq[pos + 1:pos + k + 1]))
                counts[s][s_] += 1.0
                obs_counts[s] += 1

            # Skipping : l'item x_j remplace x_{pos+k} à la fin de l'état futur
            # s'_skip = (seq[pos+1], ..., seq[pos+k-1], seq[j])
            for j in range(pos + k + 1, min(len(seq), pos + k + 1 + MAX_SKIP)):
                weight = 1.0 / (2 ** (j - (pos + k)))
                s_skip = str(tuple(int(x) for x in seq[pos + 1:pos + k]) + (int(seq[j]),))
                counts[s][s_skip] += weight

    def _flush(self, conn, counts, obs_counts):
        rows = [(s, s_, c) for s, succ in counts.items() for s_, c in succ.items()]
        conn.executemany("""
            INSERT INTO counts (s, s_, count) VALUES (?, ?, ?)
            ON CONFLICT(s, s_) DO UPDATE SET count = count + excluded.count
        """, rows)
        obs_rows = [(s, n) for s, n in obs_counts.items()]
        conn.executemany("""
            INSERT INTO state_obs (s, obs_count) VALUES (?, ?)
            ON CONFLICT(s) DO UPDATE SET obs_count = obs_count + excluded.obs_count
        """, obs_rows)
        conn.commit()

    def _init_db(self, conn):
        conn.execute("DROP TABLE IF EXISTS counts")
        conn.execute("DROP TABLE IF EXISTS transitions")
        conn.execute("DROP TABLE IF EXISTS state_obs")
        conn.execute("""CREATE TABLE counts (
            s TEXT NOT NULL, s_ TEXT NOT NULL, count REAL NOT NULL,
            PRIMARY KEY (s, s_))""")
        conn.execute("""CREATE TABLE transitions (
            s TEXT NOT NULL, s_ TEXT NOT NULL, prob REAL NOT NULL,
            PRIMARY KEY (s, s_))""")
        conn.execute("""CREATE TABLE state_obs (
            s TEXT NOT NULL PRIMARY KEY,
            obs_count INTEGER NOT NULL DEFAULT 0)""")
        conn.commit()

    def _normalize(self, conn):
        conn.execute("""
            INSERT INTO transitions (s, s_, prob)
            SELECT s, s_, count / SUM(count) OVER (PARTITION BY s)
            FROM counts
        """)
        conn.execute("DROP TABLE counts")
        conn.commit()

    @classmethod
    def load(cls, k: int = K) -> 'MarkovModelSkipping':
        return cls(k)

    def get_trmc(self, s: tuple, s_: tuple) -> float:
        s  = tuple(int(x) for x in s)
        s_ = tuple(int(x) for x in s_)
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT prob FROM transitions WHERE s = ? AND s_ = ?",
            (str(s), str(s_))
        ).fetchone()
        conn.close()
        return row[0] if row else 0.0


if __name__ == "__main__":
    model = MarkovModelSkipping()
    model.fit()

    a = (8313, 207, 16668)
    b = (207, 16668, 5869)
    print(model.get_trmc(a, b))