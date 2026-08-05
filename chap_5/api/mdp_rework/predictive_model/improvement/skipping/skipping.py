from collections import defaultdict

from chap_5.api.mdp_rework.config import MAX_SKIP
from chap_5.api.mdp_rework.predictive_model.improvement.skipping.data.sql import (
    flush_counts, create_skipping_database, init_skipping_db, normalize_and_clean
)
from chap_5.api.mdp_rework.predictive_model.improvement.skipping.utils.get_user_historique import get_user_historique
from chap_5.api.mdp_rework.shared.debug_log import log_progress, reset_progress


class SkippingModel:
    def __init__(self, k: int):
        self.k = k
        self.counts = defaultdict(lambda: defaultdict(float))
        self.engine = create_skipping_database()
        init_skipping_db(self.engine, self.k)

    def skipping(self):
        historiques = get_user_historique()
        len_historiques = len(historiques)
        reset_progress()
        for i in range(len_historiques):
            hist_to_list = historiques[i].tolist()
            self._process_sequence(hist_to_list, self.counts)
            self.save()
            log_progress(i + 1, len_historiques, 'Skipping')

        if self.counts:
            flush_counts(self.engine, self.counts)
            self.counts.clear()

        normalize_and_clean(self.engine, self.k)

    def save(self):
        n_rows = sum(len(t) for t in self.counts.values())
        if n_rows >= 10_000_000:
            flush_counts(self.engine, self.counts)
            self.counts.clear()

    def _process_sequence(self, histo, counts):
        k = self.k
        histo_len = len(histo)
        if histo_len <= k:
            return

        state = tuple(histo[:k])

        for pos in range(histo_len - k):
            prefix = state[1:]

            next_item = histo[pos + k]
            state_next = prefix + (next_item,)
            counts[state][state_next] += 1.0

            stop_skipping = min(histo_len, pos + k + 1 + MAX_SKIP)
            weight = 0.5
            for j in range(pos + k + 1, stop_skipping):
                state_skip = prefix + (histo[j],)
                counts[state][state_skip] += weight
                weight *= 0.5

            state = state_next