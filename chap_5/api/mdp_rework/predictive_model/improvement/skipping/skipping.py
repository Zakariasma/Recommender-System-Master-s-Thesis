from collections import defaultdict

from chap_5.api.mdp_rework.config import MAX_SKIP
from chap_5.api.mdp_rework.predictive_model.improvement.skipping.data.sql import (
    create_skipping_database, init_skipping_db, normalize_and_clean, open_persistent_connection, flush_counts_conn
)
from chap_5.api.mdp_rework.predictive_model.improvement.skipping.utils.get_user_historique import get_user_historique
from chap_5.api.mdp_rework.shared.debug_log import log_progress, reset_progress


class SkippingModel:
    def __init__(self, k: int):
        self.k = k
        self.counts = defaultdict(lambda: defaultdict(float))
        self.total_rows = 0
        self.db_path = create_skipping_database()
        init_skipping_db(self.db_path, self.k)
        init_skipping_db(self.db_path, self.k)
        self.conn = open_persistent_connection(self.db_path)
        self.conn.execute("PRAGMA synchronous=OFF")

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
                flush_counts_conn(self.conn, self.counts)
                self.counts.clear()
                self.total_rows = 0

            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.close()
            normalize_and_clean(self.db_path, self.k)

    def save(self):
        if self.total_rows >= 10_000_000:
            print(f'save {self.total_rows}')
            flush_counts_conn(self.conn, self.counts)
            self.counts.clear()
            self.total_rows = 0

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

            if state_next not in counts[state]:
                self.total_rows += 1
            counts[state][state_next] += 1.0

            stop_skipping = min(histo_len, pos + k + 1 + MAX_SKIP)
            weight = 0.5
            for j in range(pos + k + 1, stop_skipping):
                state_skip = prefix + (histo[j],)
                if state_skip not in counts[state]:
                    self.total_rows += 1
                counts[state][state_skip] += weight
                weight *= 0.5

            state = state_next