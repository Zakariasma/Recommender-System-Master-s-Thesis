import numpy as np
import pandas as pd
import os

FULL_HIST_PATH = "../data/cleaned/historique_full.csv"
NPY_DIR = "../data/npy"
FLAT_PATH = os.path.join(NPY_DIR, "sequences_flat.npy")
OFFSETS_PATH = os.path.join(NPY_DIR, "sequences_offsets.npy")
USER_IDS_PATH = os.path.join(NPY_DIR, "user_ids.csv")


def build_sequences():
    print("Convertion csv en npy")
    os.makedirs(NPY_DIR, exist_ok=True)
    df = pd.read_csv(FULL_HIST_PATH, parse_dates=['timestamp'])

    # On double sort pour préserver l'ordre des sequences
    df = df.sort_values(['userId', 'timestamp'], ascending=[True, True]).reset_index(drop=True)

    grouped = df.groupby('userId', sort=True)['movie_id'].apply(list)
    user_ids = list(grouped.index)
    sequences = list(grouped.values)

    # Deux fichiers :
    # Flat -> sequences brut les une derrieres les autres
    # Offset -> donne la position de commencement de la sequence d'un user
    flat = np.array([item for seq in sequences for item in seq], dtype=np.int32)
    offsets = np.array([0] + list(np.cumsum([len(s) for s in sequences])), dtype=np.int32)

    np.save(FLAT_PATH, flat)
    np.save(OFFSETS_PATH, offsets)
    pd.DataFrame({'index': range(len(user_ids)), 'userId': user_ids}).to_csv(USER_IDS_PATH, index=False)


if __name__ == "__main__":
    build_sequences()