import os
import gc
import json
import math
import time
import pickle
import sqlite3
import tempfile
import multiprocessing as mp
import numpy as np
import scipy.sparse as sp
import rust_solver

from chap_5.api.mdp.predictive_model.state_key import encode, decode
from chap_5.api.mdp.config import GAMMA_RL, LIST_SIZE, THRESHOLD, BATCH_SIZE
from chap_5.api.mdp.mdp.reward import RewardModel
from chap_5.api.mdp.parralelized.sqlite_stores import SQLiteTransitionStoreReadOnly
from chap_5.api.mdp.parralelized.predictive_model_sqlite import PredictiveModelSQLite
from chap_5.api.mdp.parralelized.batch_mdp import batch_get_successors, dist_list, dist_single, make_alpha_beta, \
    batch_get_successors_flat

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "models")

TRANSITIONS_DB_PATH = os.path.join(DATA_DIR, "transitions.db")
TRANSITIONS_TABLE = "transitions_skipping"
SOLVER_STATE_DB = os.path.join(DATA_DIR, "solver_state.db")

# =============================================================================
#  GLOBALS : partagés via fork/COW.
# =============================================================================
_global_states = None
_global_state_to_idx = None
_global_rust_state = None  # <-- AJOUTEZ ÇA
_V_raw = None
_pi_raw = None
_ps_raw = None

_ctx = {}


def _init_worker(list_size, transitions_db_path, transitions_table, counter):
    gc.disable()  # CRUCIAL : empêche le GC de dupliquer la mémoire COW

    _ctx["ab"] = make_alpha_beta()
    _ctx["list_size"] = list_size
    _ctx["counter"] = counter
    _ctx["transition_store"] = SQLiteTransitionStoreReadOnly(transitions_db_path, transitions_table)

    # On récupère l'objet Rust global (hérité via COW, 0 Mo de RAM supplémentaire)
    _ctx["rust_state"] = _global_rust_state

    n = len(_global_state_to_idx)
    _ctx["V"] = np.frombuffer(_V_raw, dtype=np.float64)
    _ctx["policy_items"] = np.frombuffer(_pi_raw, dtype=np.int32).reshape(n, list_size)
    _ctx["policy_scores"] = np.frombuffer(_ps_raw, dtype=np.float32).reshape(n, list_size)


def _transition_chunk(task):
    idx, start, end = task
    states = _global_states
    counter = _ctx["counter"]
    list_size = _ctx["list_size"]
    policy_items = _ctx["policy_items"]
    rust_state = _ctx["rust_state"]

    MICRO = 1000
    all_cols, all_data, all_indptr = [], [], [0]

    for chunk_start in range(start, end, MICRO):
        chunk_end = min(chunk_start + MICRO, end)
        states_micro = states[chunk_start:chunk_end]

        if isinstance(states_micro, np.ndarray):
            states_list = [list(s) for s in states_micro]
        else:
            states_list = [list(s) for s in states_micro]

        policy_chunk = []
        for i in range(len(states_list)):
            sidx = chunk_start + i
            items = [int(policy_items[sidx, j]) for j in range(list_size) if policy_items[sidx, j] >= 0]
            policy_chunk.append(items)

        tuples_list = [tuple(s) for s in states_list]

        # On calcule juste les chaînes de préfixes en Python (très rapide)
        # On calcule juste les chaînes de préfixes en Python (très rapide)
        k1_prefixes = [encode(s[-1:]).encode('utf-8') for s in tuples_list]
        k2_prefixes = [encode(s[-2:]).encode('utf-8') for s in tuples_list]
        k3_prefixes = [encode(s[-3:]).encode('utf-8') for s in tuples_list]

        # APPEL RUST : Rust fera tout le travail de matching en mémoire !
        cols, data, indptr = rust_state.build_p_chunk(
            states_list, policy_chunk, k1_prefixes, k2_prefixes, k3_prefixes
        )

        all_cols.extend(cols)
        all_data.extend(data)
        offset = all_indptr[-1]
        all_indptr.extend([p + offset for p in indptr[1:]])

        with counter.get_lock():
            counter.value += len(states_list)

    return idx, np.array(all_cols, dtype=np.int32), np.array(all_data, dtype=np.float64), np.array(all_indptr,
                                                                                                   dtype=np.int32)


def _improve_chunk(task):
    idx, start, end = task
    states = _global_states
    counter = _ctx["counter"]
    list_size = _ctx["list_size"]
    policy_items = _ctx["policy_items"]
    policy_scores = _ctx["policy_scores"]
    V = _ctx["V"]
    rust_state = _ctx["rust_state"]

    stable_local = True
    changed = 0
    MICRO = 1000

    for chunk_start in range(start, end, MICRO):
        chunk_end = min(chunk_start + MICRO, end)
        states_micro = states[chunk_start:chunk_end]

        if isinstance(states_micro, np.ndarray):
            states_list = [list(s) for s in states_micro]
        else:
            states_list = [list(s) for s in states_micro]

        policy_chunk = []
        for i in range(len(states_list)):
            sidx = chunk_start + i
            items = [int(policy_items[sidx, j]) for j in range(list_size) if policy_items[sidx, j] >= 0]
            policy_chunk.append(items)

        tuples_list = [tuple(s) for s in states_list]
        k1_prefixes = [encode(s[-1:]).encode('utf-8') for s in tuples_list]
        k2_prefixes = [encode(s[-2:]).encode('utf-8') for s in tuples_list]
        k3_prefixes = [encode(s[-3:]).encode('utf-8') for s in tuples_list]

        # APPEL RUST
        is_stable, has_changed, new_policies, new_scores = rust_state.improve_chunk(
            states_list, policy_chunk, k1_prefixes, k2_prefixes, k3_prefixes,
            np.ascontiguousarray(V), list_size
        )

        if not is_stable:
            stable_local = False
        changed += has_changed

        for i in range(len(states_list)):
            sidx = chunk_start + i
            for j in range(list_size):
                if j < len(new_policies[i]):
                    policy_items[sidx, j] = new_policies[i][j]
                    policy_scores[sidx, j] = new_scores[i][j]
                else:
                    policy_items[sidx, j] = -1
                    policy_scores[sidx, j] = 0.0

        with counter.get_lock():
            counter.value += len(states_list)

    return idx, stable_local, changed, end - start


def _chunkify(n_states, n_chunks):
    size = max(1, n_states // n_chunks)
    return [(i, min(i + size, n_states)) for i in range(0, n_states, size)]


def get_observed_states_cached(pred_model, k=1, cache_path="observed_states_cache.pkl", force_refresh=False):
    if not force_refresh and os.path.exists(cache_path):
        start = time.time()
        with open(cache_path, "rb") as f:
            states = pickle.load(f)
        print(f"États chargés depuis le cache '{cache_path}' en {time.time() - start:.1f}s ({len(states)} états)")
        return states

    print("Cache absent ou refresh forcé, calcul depuis SQLite...")
    start = time.time()
    states = pred_model.get_observed_states(k)
    print(f"Calcul terminé en {(time.time() - start) / 60:.1f} min ({len(states)} états)")

    dir_ = os.path.dirname(os.path.abspath(cache_path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print(f"Cache sauvegardé dans '{cache_path}'")
    return states


class ProgressTracker:
    def __init__(self, path="solver_progress.json"):
        self.path = path
        self._data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {"meta": {}}

    def _save(self):
        dir_ = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f)
            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def get_meta(self, key, default=None):
        return self._data["meta"].get(key, default)

    def set_meta(self, key, value):
        self._data["meta"][key] = value
        self._save()


class MDPSolverParallel:
    def __init__(self, gamma=GAMMA_RL, list_size=LIST_SIZE, threshold=THRESHOLD,
                 batch_size=BATCH_SIZE, n_workers=None,
                 progress_path="solver_progress.json",
                 transitions_db_path=TRANSITIONS_DB_PATH,
                 transitions_table=TRANSITIONS_TABLE,
                 solver_state_db=SOLVER_STATE_DB):
        self.gamma = gamma
        self.list_size = list_size
        self.threshold = threshold
        self.batch_size = batch_size
        self.n_workers = n_workers or os.cpu_count()
        self.progress = ProgressTracker(progress_path)
        self.transitions_db_path = transitions_db_path
        self.transitions_table = transitions_table
        self.solver_state_db = solver_state_db
        self._n_states = 0
        self.R = None  # vecteur de récompenses, calculé une seule fois

    def _setup_shared_memory(self, states):
        global _global_states, _global_state_to_idx, _global_rust_state, _V_raw, _pi_raw, _ps_raw

        n = len(states)
        self._n_states = n
        L = self.list_size

        print(f"  Construction du mapping ({n:,} états)...")
        start = time.time()
        _global_states = states
        _global_state_to_idx = {s: i for i, s in enumerate(states)}
        print(f"  Mapping Python construit en {time.time() - start:.1f}s")

        print(f"  Construction du cache Rust...")
        start = time.time()
        caches = self._build_trans_cache_arrays(self.transitions_db_path, self.transitions_table)

        # On récupère les poids K, le modèle de popularité et gamma
        from chap_5.api.mdp.parralelized.batch_mdp import K_WEIGHTS, make_alpha_beta
        ab = make_alpha_beta()
        pop_dict = ab.popularity._p
        pop_min = ab.popularity._min
        gamma = ab.gamma

        k1_s, k1_i, k1_p = caches[1]
        k2_s, k2_i, k2_p = caches[2]
        k3_s, k3_i, k3_p = caches[3]

        k1_s_u8 = np.ascontiguousarray(k1_s.view(np.uint8)).ravel()
        k2_s_u8 = np.ascontiguousarray(k2_s.view(np.uint8)).ravel()
        k3_s_u8 = np.ascontiguousarray(k3_s.view(np.uint8)).ravel()

        _global_rust_state = rust_solver.SolverState(
            _global_state_to_idx,
            k1_s_u8, k1_i, k1_p,
            k2_s_u8, k2_i, k2_p,
            k3_s_u8, k3_i, k3_p,
            K_WEIGHTS,
            pop_dict,  # <-- AJOUT
            pop_min,  # <-- AJOUT
            gamma  # <-- AJOUT
        )
        print(f"  Objet Rust construit en {time.time() - start:.1f}s")

        print(f"  Allocation mémoire partagée...")
        _V_raw = mp.RawArray('d', n)
        _pi_raw = mp.RawArray('i', n * L)
        _ps_raw = mp.RawArray('f', n * L)

        V_np = np.frombuffer(_V_raw, dtype=np.float64)
        V_np.fill(0.0)
        pi_np = np.frombuffer(_pi_raw, dtype=np.int32).reshape(n, L)
        pi_np.fill(-1)
        ps_np = np.frombuffer(_ps_raw, dtype=np.float32).reshape(n, L)
        ps_np.fill(0.0)

        self._load_checkpoint()
        self.R = self._compute_rewards(states)

    def _compute_rewards(self, states):
        print("  Calcul du vecteur de récompenses (une seule fois)...")
        start = time.time()
        reward_model = RewardModel()
        R = np.empty(len(states), dtype=np.float64)
        for i, s in enumerate(states):
            R[i] = reward_model.get(s[-1], default=0.0) / 100.0
        print(f"  Récompenses calculées en {time.time() - start:.1f}s")
        return R

    def _load_checkpoint(self):
        if not os.path.exists(self.solver_state_db):
            return

        conn = sqlite3.connect(self.solver_state_db, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")

        try:
            conn.execute("CREATE TABLE IF NOT EXISTS solver_checkpoint (key TEXT PRIMARY KEY, value BLOB)")
            row = conn.execute("SELECT value FROM solver_checkpoint WHERE key = 'n_states'").fetchone()

            if row and int(row[0]) == self._n_states:
                print("  Chargement checkpoint...")
                start = time.time()
                row = conn.execute("SELECT value FROM solver_checkpoint WHERE key = 'V'").fetchone()
                if row:
                    np.frombuffer(_V_raw, dtype=np.float64)[:] = np.frombuffer(row[0], dtype=np.float64)

                row = conn.execute("SELECT value FROM solver_checkpoint WHERE key = 'policy_items'").fetchone()
                if row:
                    np.frombuffer(_pi_raw, dtype=np.int32)[:] = np.frombuffer(row[0], dtype=np.int32)

                print(f"  Checkpoint chargé en {time.time() - start:.1f}s")
        except Exception:
            pass
        finally:
            conn.close()

    def _checkpoint(self):
        print("  Checkpoint...", end="", flush=True)
        start = time.time()
        conn = sqlite3.connect(self.solver_state_db, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("CREATE TABLE IF NOT EXISTS solver_checkpoint (key TEXT PRIMARY KEY, value BLOB)")

        V_np = np.frombuffer(_V_raw, dtype=np.float64)
        pi_np = np.frombuffer(_pi_raw, dtype=np.int32)
        ps_np = np.frombuffer(_ps_raw, dtype=np.float32)

        conn.execute("BEGIN")
        conn.execute("INSERT OR REPLACE INTO solver_checkpoint (key, value) VALUES ('V', ?)", (V_np.tobytes(),))
        conn.execute("INSERT OR REPLACE INTO solver_checkpoint (key, value) VALUES ('policy_items', ?)",
                     (pi_np.tobytes(),))
        conn.execute("INSERT OR REPLACE INTO solver_checkpoint (key, value) VALUES ('policy_scores', ?)",
                     (ps_np.tobytes(),))
        conn.execute("INSERT OR REPLACE INTO solver_checkpoint (key, value) VALUES ('n_states', ?)",
                     (str(self._n_states),))
        conn.execute("COMMIT")
        conn.close()
        print(f" {time.time() - start:.1f}s")

    def _pool(self, counter):
        gc.disable()
        try:
            pool = mp.Pool(processes=self.n_workers, initializer=_init_worker,
                           initargs=(self.list_size, self.transitions_db_path, self.transitions_table,
                                     counter))
        finally:
            gc.enable()
        return pool

    def _run_chunks(self, chunks, phase, worker_fn, total_states):
        tasks = [(i, start, end) for i, (start, end) in enumerate(chunks)]
        start_time = time.time()
        counter = mp.Value("l", 0)

        with self._pool(counter) as pool:
            async_result = pool.map_async(worker_fn, tasks)
            while not async_result.ready():
                time.sleep(1.0)
                elapsed = time.time() - start_time
                done = counter.value
                speed = done / elapsed if elapsed > 0 else 0
                remaining = total_states - done
                eta_min = (remaining / speed / 60) if speed > 0 else 0
                print(f"\r  [{phase}] {done:,}/{total_states:,} | {speed:,.0f} états/s | ETA: {eta_min:.1f} min",
                      end="", flush=True)
            results = async_result.get()
        print()
        return results, time.time() - start_time

    def _build_transition_matrix(self, states, iteration):
        """
        Construit la matrice de transition creuse P (n x n) en parallèle.
        Évite l'explosion de RAM en utilisant NumPy pour les tableaux locaux.
        """
        n = len(states)
        chunks = _chunkify(n, self.n_workers * 4)
        results, elapsed = self._run_chunks(chunks, f"build-P-i{iteration}", _transition_chunk, n)

        global_indptr = np.zeros(n + 1, dtype=np.int32)
        cols_list = []
        data_list = []
        current_ptr = 0

        # Reconstruction de la matrice globale à partir des chunks
        for r in results:
            idx, cols, data, indptr = r
            if len(data) == 0: continue
            start_idx = chunks[idx][0]
            chunk_len = len(indptr) - 1
            # On décale le indptr local avec le pointeur global courant
            global_indptr[start_idx: start_idx + chunk_len + 1] = indptr + current_ptr
            cols_list.append(cols)
            data_list.append(data)
            current_ptr += len(data)

        if cols_list:
            global_cols = np.concatenate(cols_list)
            global_data = np.concatenate(data_list)
            P = sp.csr_matrix((global_data, global_cols, global_indptr), shape=(n, n))
        else:
            P = sp.csr_matrix((n, n))

        print(f"  [BuildP] iter {iteration} — {elapsed:.1f}s | {P.nnz:,} transitions non-nulles "
              f"({P.nnz / max(1, n):.1f} / état)")
        return P

    def _build_trans_cache_arrays(self, transitions_db_path, transitions_table):
        """Charge la DB de 11 Go dans des tableaux NumPy compacts (~3 Go de RAM)."""
        print("  Chargement des transitions en tableaux NumPy...")
        start = time.time()
        conn = sqlite3.connect(f"file:{os.path.abspath(transitions_db_path)}?mode=ro", uri=True, timeout=30.0)

        caches = {}
        for k in [1, 2, 3]:
            cursor = conn.execute(f"SELECT s, s_, prob FROM {transitions_table} WHERE k = ?", (k,))

            s_list = []
            item_list = []
            prob_list = []

            while True:
                rows = cursor.fetchmany(100000)
                if not rows:
                    break
                for s, s_, prob in rows:
                    s_list.append(s.encode('utf-8'))
                    item_list.append(decode(s_)[-1])
                    prob_list.append(prob)

            s_arr = np.array(s_list, dtype='S32')
            item_arr = np.array(item_list, dtype=np.int32)
            prob_arr = np.array(prob_list, dtype=np.float32)

            caches[k] = (s_arr, item_arr, prob_arr)
            print(f"    k={k}: {len(s_arr):,} transitions")

        conn.close()
        print(f"  Tableaux construits en {time.time() - start:.1f}s")
        return caches

    def policy_evaluation(self, states, iteration):
        """
        Résout V = R + gamma * P @ V par itération de point fixe.
        """
        P = self._build_transition_matrix(states, iteration)
        V = np.frombuffer(_V_raw, dtype=np.float64)
        R = self.R

        sweep = 0
        sweep_times, deltas = [], []

        while True:
            t0 = time.time()
            V_new = R + self.gamma * (P @ V)
            delta = float(np.max(np.abs(V_new - V)))
            V[:] = V_new
            elapsed = time.time() - t0

            sweep += 1
            sweep_times.append(elapsed)
            deltas.append(delta)
            avg_time = sum(sweep_times) / len(sweep_times)

            remaining_sweeps_str, remaining_time_str = "?", "?"
            if len(deltas) >= 2 and 0 < delta < deltas[-2]:
                ratio = delta / deltas[-2]
                if 0 < ratio < 1:
                    remaining_sweeps = max(0, math.log(self.threshold / delta) / math.log(ratio))
                    remaining_time_str = f"{(remaining_sweeps * avg_time):.2f}s"
                    remaining_sweeps_str = f"{remaining_sweeps:.0f}"

            print(f"  [Eval] iter {iteration} sweep {sweep} — delta={delta:.6f} | "
                  f"{elapsed * 1000:.1f}ms | restant: ~{remaining_sweeps_str} sweeps (~{remaining_time_str})")

            if delta < self.threshold:
                break

        self.progress.set_meta(f"eval_sweeps_{iteration}", sweep)

    def policy_improvement(self, states, iteration):
        chunks = _chunkify(len(states), self.n_workers * 4)
        results, elapsed = self._run_chunks(chunks, f"improve-i{iteration}", _improve_chunk, len(states))
        stable = all(r[1] for r in results)
        total_changed = sum(r[2] for r in results)
        print(f"  [Improve] iter {iteration} — {elapsed:.1f}s | {total_changed:,} policies modifiées")
        return stable

    def solve(self, states, max_iterations=100):
        self._setup_shared_memory(states)

        start_iter = self.progress.get_meta("iteration", default=0)
        phase = self.progress.get_meta("phase", default="eval")

        for i in range(start_iter, max_iterations):
            self.progress.set_meta("iteration", i)
            print(f"\n=== Itération {i + 1}/{max_iterations} ({len(states):,} états, {self.n_workers} cœurs) ===")

            if phase == "eval":
                self.policy_evaluation(states, iteration=i)
                self.progress.set_meta("phase", "improve")
                phase = "improve"

            stable = self.policy_improvement(states, iteration=i)
            self.progress.set_meta("phase", "eval")
            phase = "eval"

            self._checkpoint()

            print(f"Itération {i + 1} — politique stable: {stable}")
            if stable:
                self.progress.set_meta("done", True)
                break

        self._checkpoint()


if __name__ == "__main__":
    pred_model = PredictiveModelSQLite(TRANSITIONS_DB_PATH, TRANSITIONS_TABLE)
    states = get_observed_states_cached(pred_model)
    print(f"Nombre d'états à traiter : {len(states):,}")
    MDPSolverParallel().solve(states)