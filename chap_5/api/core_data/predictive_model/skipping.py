import numpy as np
import os
from collections import defaultdict

from chap_5.api.core_data.predictive_model.db.db import TransitionDB

NPY_DIR      = "../data/npy"
FLAT_PATH    = os.path.join(NPY_DIR, "sequences_flat.npy")
OFFSETS_PATH = os.path.join(NPY_DIR, "sequences_offsets.npy")


class SkippingModel:

    def __init__(self, k: int, db: TransitionDB, max_skip: int, batch_size: int, fraction: float = None):
        self.k          = k
        self.db         = db
        self.max_skip   = max_skip
        self.batch_size = batch_size
        self.fraction   = fraction

    def fit(self):
        flat    = np.load(FLAT_PATH,    mmap_mode='r')
        offsets = np.load(OFFSETS_PATH, mmap_mode='r')
        total   = len(offsets) - 1

        if self.fraction is not None:
            total = int(total * self.fraction)

        counts = defaultdict(lambda: defaultdict(float))

        for i in range(total):
            seq = flat[offsets[i]:offsets[i+1]]
            self._process_sequence(seq, counts)

            if (i + 1) % self.batch_size == 0:
                self.db.flush_counts(
                    [(s, s_, c) for s, succ in counts.items() for s_, c in succ.items()]
                )
                counts.clear()
                self._log(i + 1, total)

        if counts:
            self.db.flush_counts(
                [(s, s_, c) for s, succ in counts.items() for s_, c in succ.items()]
            )

        print(f"\r  [skipping k={self.k}] terminé" + " " * 30)
        self.db.normalize_and_clean(self.k)

    def _fmt(self, items) -> str:
        return ','.join(str(x) for x in items)

    def _process_sequence(self, seq, counts):
        k = self.k
        for pos in range(len(seq) - k):
            s  = self._fmt(seq[pos:pos + k])
            s_ = self._fmt(seq[pos + 1:pos + k + 1])

            counts[s][s_] += 1.0

            for j in range(pos + k + 1, min(len(seq), pos + k + 1 + self.max_skip)):
                weight = 1.0 / (2 ** (j - (pos + k)))
                s_skip = self._fmt(list(seq[pos + 1:pos + k]) + [int(seq[j])])
                counts[s][s_skip] += weight

    def _log(self, done, total):
        pct = 100 * done / total
        print(f"\r  [skipping k={self.k}] {done}/{total} ({pct:.1f}%)", end="", flush=True)