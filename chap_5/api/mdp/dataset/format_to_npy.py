import numpy as np
import pandas as pd

from chap_5.api.mdp.config import FULL_HIST_PATH, NPY_DIR, FLAT_PATH, OFFSETS_PATH, USER_IDS_PATH


def build_sequences():
    print("Conversion csv en npy")
    NPY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FULL_HIST_PATH, parse_dates=['timestamp'])
    df = df.sort_values(['userId', 'timestamp']).reset_index(drop=True)

    grouped = df.groupby('userId', sort=True)['movie_id'].apply(list)
    user_ids = list(grouped.index)
    sequences = list(grouped.values)

    flat = np.array([item for seq in sequences for item in seq], dtype=np.int32)
    offsets = np.array([0] + list(np.cumsum([len(s) for s in sequences])), dtype=np.int32)

    np.save(FLAT_PATH, flat)
    np.save(OFFSETS_PATH, offsets)
    pd.DataFrame({'index': range(len(user_ids)), 'userId': user_ids}).to_csv(USER_IDS_PATH, index=False)


if __name__ == "__main__":
    build_sequences()
