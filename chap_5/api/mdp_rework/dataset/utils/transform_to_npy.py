import numpy as np
import pandas as pd

from chap_5.api.mdp.config import FULL_HIST_PATH, NPY_DIR, FLAT_PATH, OFFSETS_PATH, USER_IDS_PATH


def transform_to_npy():
    NPY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FULL_HIST_PATH, parse_dates=['timestamp']).sort_values(['user_id', 'timestamp'])

    grouped = df.groupby('user_id', sort=True)['movie_id'].apply(list)

    flat = np.concatenate(grouped.values).astype(np.int32)
    offsets = np.insert(np.cumsum([len(s) for s in grouped]), 0, 0).astype(np.int32)

    np.save(FLAT_PATH, flat)
    np.save(OFFSETS_PATH, offsets)
    pd.DataFrame({'userId': grouped.index}).to_csv(USER_IDS_PATH, index_label='index')


if __name__ == "__main__":
    transform_to_npy()