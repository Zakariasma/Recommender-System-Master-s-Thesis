import numpy as np
import os
from collections import Counter

NPY_DIR   = "../data/npy"
FLAT_PATH = os.path.join(NPY_DIR, "sequences_flat.npy")


class PopularityModel:
    """
    p(r) = nombre de fois que r apparaît dans tout l'historique
           / nombre total d'items consommés

    Utilisé par α pour mesurer à quel point un item est populaire.
    Plus p(r) est petit, plus le boost α est grand.
    """

    def __init__(self, flat_path: str = FLAT_PATH):
        flat        = np.load(flat_path, mmap_mode='r')
        total       = len(flat)
        counts      = Counter(int(x) for x in flat)
        self._p     = {item: c / total for item, c in counts.items()}
        self._min   = min(self._p.values()) / 2  # fallback items jamais vus

    def get(self, item: int) -> float:
        return self._p.get(item, self._min)