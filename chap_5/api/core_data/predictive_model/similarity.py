from collections import defaultdict

from chap_5.api.core_data.predictive_model.db.db import TransitionDB

LOG_EVERY = 100


class SimilarityModel:

    def __init__(self, k: int, db: TransitionDB, batch_size: int, movie_genres: dict):
        self.k            = k
        self.db           = db
        self.batch_size   = batch_size
        self.movie_genres = movie_genres

    def fit(self):
        tr_old     = self.db.load_skipping_transitions(self.k)
        all_states = list(tr_old.keys())

        print(f"  [similarity k={self.k}] {len(all_states)} états")

        sig_index = self._build_signature_index(all_states)
        rows      = []

        for idx, s in enumerate(all_states):
            s_tuple   = self._parse(s)
            sig       = self._get_signature(s_tuple)
            new_probs = self._compute(s, s_tuple, sig, sig_index, tr_old)

            rows.extend((s, s_, self.k, prob) for s_, prob in new_probs.items())

            if len(rows) >= self.batch_size:
                self.db.flush_similarity(rows)
                rows.clear()

            if (idx + 1) % LOG_EVERY == 0:
                self._log(idx + 1, len(all_states))

        if rows:
            self.db.flush_similarity(rows)

        print(f"\r  [similarity k={self.k}] terminé" + " " * 30)

    def _compute(self, s, s_tuple, sig, sig_index, tr_old):
        if sig is None:
            return self._normalize(tr_old.get(s, {}))

        simcount = defaultdict(float)
        for si in sig_index.get(sig, []):
            if si == s:
                continue
            weight = self._sim(s_tuple, self._parse(si))
            if weight == 0:
                continue
            for s_, prob in tr_old[si].items():
                simcount[s_] += weight * prob

        if not simcount:
            return self._normalize(tr_old.get(s, {}))

        simcount_norm = self._normalize(simcount)
        all_s_        = set(tr_old.get(s, {}).keys()) | set(simcount_norm.keys())
        mixed         = {
            s_: 0.5 * tr_old.get(s, {}).get(s_, 0.0) + 0.5 * simcount_norm.get(s_, 0.0)
            for s_ in all_s_
        }
        return self._normalize(mixed)

    def _parse(self, s_str: str) -> tuple:
        return tuple(int(x) for x in s_str.split(','))

    def _get_signature(self, s_tuple):
        sig = []
        for movie_id in s_tuple:
            genres = self.movie_genres.get(movie_id)
            if not genres:
                return None
            sig.append(frozenset(genres))
        return tuple(sig)

    def _build_signature_index(self, all_states):
        index = defaultdict(list)
        for s_str in all_states:
            sig = self._get_signature(self._parse(s_str))
            if sig is not None:
                index[sig].append(s_str)
        return index

    def _sim(self, s_tuple, si_tuple):
        return sum((m + 1) for m, (a, b) in enumerate(zip(s_tuple, si_tuple)) if a == b)

    def _normalize(self, d: dict) -> dict:
        total = sum(d.values())
        return {} if total == 0 else {k: v / total for k, v in d.items()}

    def _log(self, done, total):
        pct = 100 * done / total
        print(f"\r  [similarity k={self.k}] {done}/{total} ({pct:.1f}%)", end="", flush=True)