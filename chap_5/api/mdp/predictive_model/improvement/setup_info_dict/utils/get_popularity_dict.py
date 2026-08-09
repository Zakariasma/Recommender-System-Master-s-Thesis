from collections import Counter
import numpy as np
from chap_5.api.mdp.config import FLAT_PATH, OFFSETS_PATH, FRACTION

def get_item_counts() -> tuple[dict, int]:
    flat = np.load(FLAT_PATH, mmap_mode='r')
    offsets = np.load(OFFSETS_PATH, mmap_mode='r')
    total = int((len(offsets) - 1) * FRACTION)

    item_counts = Counter()
    total_items = 0

    for i in range(total):
        historique = flat[offsets[i]:offsets[i + 1]]
        unique, counts = np.unique(historique, return_counts=True)

        for item, count in zip(unique, counts):
            item_counts[int(item)] += int(count)
            total_items += int(count)

    return dict(item_counts), total_items