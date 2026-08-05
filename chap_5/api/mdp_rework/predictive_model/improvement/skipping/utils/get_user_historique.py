import numpy as np
from chap_5.api.mdp_rework.config import FLAT_PATH, OFFSETS_PATH, FRACTION


def get_user_historique():
    flat = np.load(FLAT_PATH, mmap_mode='r')
    offsets = np.load(OFFSETS_PATH, mmap_mode='r')
    total = int((len(offsets) - 1) * FRACTION)
    historiques = []
    for i in range(total):
        historique = flat[offsets[i]:offsets[i + 1]]
        historiques.append(historique)
    return historiques