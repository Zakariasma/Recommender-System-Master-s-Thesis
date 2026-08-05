import time
import numpy as np
from collections import defaultdict

from chap_5.api.mdp.config import FLAT_PATH, OFFSETS_PATH
from chap_5.api.mdp.predictive_model.helper.sql_query import TransitionStore


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

        start_time = time.perf_counter()
        last_log_time = start_time

        for i in range(total):
            seq = flat[offsets[i]:offsets[i + 1]]
            self._process_sequence(seq, counts)

            now = time.perf_counter()
            if (i + 1) % 10_000 == 0 or now - last_log_time >= 5:
                self._log(i + 1, total, start_time)
                last_log_time = now

            if (i + 1) % self.batch_size == 0:
                print()  # On saute une ligne pour ne pas écraser le log principal
                self._flush(counts)

        self._flush(counts)
        print(f"\r  [skipping k={self.k}] terminé" + " " * 50)
        self.store.normalize_and_clean(self.k)

    def _flush(self, counts):
        if not counts:
            return

        # On compte le nombre total de transitions à insérer pour la barre de progression
        total_transitions = sum(len(succ) for succ in counts.values())
        if total_transitions == 0:
            return

        num_bytes = (self.k * 15 + 7) // 8

        # On va envoyer les données par chunks pour afficher la progression
        chunk_size = 500_000
        processed = 0
        current_rows = []

        flush_start_time = time.perf_counter()

        for s_int, succ in counts.items():
            s_bytes = s_int.to_bytes(num_bytes, 'big')
            for s_next_int, c in succ.items():
                s_next_bytes = s_next_int.to_bytes(num_bytes, 'big')
                current_rows.append((s_bytes, s_next_bytes, c))

                # Quand on atteint la taille du chunk, on envoie en BDD et on log
                if len(current_rows) >= chunk_size:
                    self.store.flush_counts(current_rows)
                    processed += len(current_rows)
                    self._log_flush(processed, total_transitions, flush_start_time)
                    current_rows = []

        # On envoie le reste
        if current_rows:
            self.store.flush_counts(current_rows)
            processed += len(current_rows)
            self._log_flush(processed, total_transitions, flush_start_time)

        # On efface la ligne de progression du flush pour nettoyer la console
        print("\r" + " " * 80, end="\r", flush=True)
        counts.clear()

    def _process_sequence(self, seq, counts):
        k = self.k
        seq_len = len(seq)
        if seq_len <= k:
            return

        BITS = 15
        MASK = (1 << (k * BITS)) - 1

        # Micro-optimisation : convertir en liste Python accélère les accès dans la boucle
        seq_list = seq.tolist()

        # 1. Initialisation du premier état sous forme d'entier
        state_int = 0
        for i in range(k):
            state_int = (state_int << BITS) | seq_list[i]

        # 2. Boucle principale
        for pos in range(seq_len - k):
            next_item = seq_list[pos + k]

            # Transition directe (fenêtre glissante)
            state_next_int = ((state_int << BITS) | next_item) & MASK
            counts[state_int][state_next_int] += 1.0

            # Préparation des skips
            stop = min(seq_len, pos + k + 1 + self.max_skip)

            weight = 0.5
            for j in range(pos + k + 1, stop):
                if weight < 1e-15:
                    break

                skip_item = seq_list[j]
                state_skip_int = ((state_int << BITS) | skip_item) & MASK
                counts[state_int][state_skip_int] += weight
                weight *= 0.5

            state_int = state_next_int

    def _log(self, done, total, start_time):
        pct = 100 * done / total
        elapsed = time.perf_counter() - start_time

        speed = done / elapsed if elapsed > 0 else 0
        remaining = total - done
        eta_seconds = (remaining / speed) if speed > 0 else 0

        eta_min = int(eta_seconds // 60)
        eta_sec = int(eta_seconds % 60)

        print(
            f"\r  [skipping k={self.k}] {done:,}/{total:,} ({pct:.1f}%) | "
            f"{speed:,.0f} séq/s | ETA: {eta_min}m {eta_sec}s   ",
            end="", flush=True
        )

    def _log_flush(self, done, total, start_time):
        pct = 100 * done / total
        elapsed = time.perf_counter() - start_time
        speed = done / elapsed if elapsed > 0 else 0

        print(
            f"\r  [flush k={self.k}] {done:,}/{total:,} lignes insérées ({pct:.1f}%) | {speed:,.0f} lignes/s   ",
            end="", flush=True
        )