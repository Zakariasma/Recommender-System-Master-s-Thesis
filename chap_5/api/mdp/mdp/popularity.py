import numpy as np
from collections import Counter

from chap_5.api.mdp.config import FLAT_PATH


class PopularityModel:
    """p(r) = fréquence de l'item r dans tout l'historique."""

    def __init__(self, flat_path=FLAT_PATH):
        flat = np.load(flat_path, mmap_mode='r')
        total = len(flat)
        counts = Counter(int(x) for x in flat)
        self._p = {item: c / total for item, c in counts.items()}
        self._min = min(self._p.values()) / 2

    def get(self, item: int) -> float:
        return self._p.get(item, self._min)
