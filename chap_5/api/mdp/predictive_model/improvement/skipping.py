import numpy as np
from collections import defaultdict

from chap_5.api.mdp.config import FLAT_PATH, OFFSETS_PATH
from chap_5.api.mdp.predictive_model.state_key import encode
from chap_5.api.mdp.predictive_model.transition_store import TransitionStore


class SkippingModel:
    def __init__(self, k: int, store: TransitionStore, max_skip: int, batch_size: int, fraction: float = 1.0):
        self.k = k
        self.store = store
        self.max_skip = max_skip
        self.batch_size = batch_size
        self.fraction = fraction

    def fit(self):
        flat = np.load(FLAT_PATH, mmap_mode='r')
        offsets = np.load(OFFSETS_PATH, mmap_mode='r')
        total = int((len(offsets) - 1) * self.fraction)

        counts = defaultdict(lambda: defaultdict(float))
        for i in range(total):
            seq = flat[offsets[i]:offsets[i + 1]]
            self._process_sequence(seq, counts)
            if (i + 1) % self.batch_size == 0:
                self._flush(counts)
                self._log(i + 1, total)

        self._flush(counts)
        print(f"\r  [skipping k={self.k}] terminé" + " " * 30)
        self.store.normalize_and_clean(self.k)

    def _flush(self, counts):
        if not counts:
            return
        rows = [(s, s_, c) for s, succ in counts.items() for s_, c in succ.items()]
        self.store.flush_counts(rows)
        counts.clear()

    def _process_sequence(self, seq, counts):
        k = self.k
        for pos in range(len(seq) - k):
            s = encode(seq[pos:pos + k])
            s_ = encode(seq[pos + 1:pos + k + 1])
            counts[s][s_] += 1.0

            for j in range(pos + k + 1, min(len(seq), pos + k + 1 + self.max_skip)):
                weight = 1.0 / (2 ** (j - (pos + k)))
                s_skip = encode(list(seq[pos + 1:pos + k]) + [int(seq[j])])
                counts[s][s_skip] += weight

    def _log(self, done, total):
        pct = 100 * done / total
        print(f"\r  [skipping k={self.k}] {done}/{total} ({pct:.1f}%)", end="", flush=True)