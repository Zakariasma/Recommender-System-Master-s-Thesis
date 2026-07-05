import numpy as np
import pandas as pd
import time
import os

FULL_HIST_PATH = "../../data/cleaned/historique_full.csv"
NPY_DIR = "../../data/npy"
FLAT_PATH = os.path.join(NPY_DIR, "sequences_flat.npy")
OFFSETS_PATH = os.path.join(NPY_DIR, "sequences_offsets.npy")
USER_IDS_PATH = os.path.join(NPY_DIR, "user_ids.csv")


def benchmark_npy():
    flat = np.load(FLAT_PATH)
    offsets = np.load(OFFSETS_PATH)
    user_ids = pd.read_csv(USER_IDS_PATH)['userId'].values

    start = time.perf_counter()
    for i in range(len(offsets) - 1):
        seq = flat[offsets[i]:offsets[i+1]]
        for item in seq:
            pass
    elapsed = time.perf_counter() - start

    print(f"[NPY]")
    print(f"  Users  : {len(user_ids)}")
    print(f"  Items  : {len(flat)}")
    print(f"  Temps  : {elapsed:.4f}s")
    return elapsed


def benchmark_csv():
    df = pd.read_csv(FULL_HIST_PATH, parse_dates=['timestamp'])
    df = df.sort_values(['userId', 'timestamp']).reset_index(drop=True)

    start = time.perf_counter()
    for user_id, group in df.groupby('userId'):
        seq = group['movie_id'].tolist()
        for item in seq:
            pass
    elapsed = time.perf_counter() - start

    print(f"[CSV]")
    print(f"  Users  : {df['userId'].nunique()}")
    print(f"  Items  : {len(df)}")
    print(f"  Temps  : {elapsed:.4f}s")
    return elapsed


if __name__ == "__main__":
    t_npy = benchmark_npy()
    print()
    t_csv = benchmark_csv()
    print()
    print(f"NPY est {t_csv / t_npy:.1f}x plus rapide que CSV")