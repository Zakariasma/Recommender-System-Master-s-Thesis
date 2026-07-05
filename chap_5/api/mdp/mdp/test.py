import os
import gc
import time
import pickle
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sqlalchemy import create_engine, text

from chap_5.api.mdp.predictive_model.state_key import encode, decode
from chap_5.api.mdp.config import DATABASE_URL, GAMMA_RL, LIST_SIZE, THRESHOLD
from chap_5.api.mdp.mdp.reward import RewardModel
from chap_5.api.mdp.helper.kv_store import KeyValueStore
from chap_5.api.mdp.parralelized.batch_mdp import make_alpha_beta

# =============================================================================
#  Globalaux partagés via fork/COW
# =============================================================================
_global_states = None
_global_state_to_idx = None
_V_raw = None
_pi_raw = None
_ctx = {}

EPSILON = 1e-6


class PostgresBatchStore:
    """Remplace SQLiteTransitionStoreReadOnly pour lire par gros paquets."""

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def batch_get_successors(self, prefixes: list, k: int) -> dict:
        if not prefixes:
            return {}

        grouped = defaultdict(dict)
        # On découpe par 900 pour éviter que l'IN clause de Postgres ne soit trop grande
        with self.engine.connect() as conn:
            for i in range(0, len(prefixes), 900):
                batch = prefixes[i:i + 900]
                in_clause = ",".join(f"'{p}'" for p in batch)
                query = text(f"SELECT s, s_, prob FROM transitions WHERE k = {k} AND s IN ({in_clause})")
                rows = conn.execute(query).fetchall()
                for s, s_, prob in rows:
                    grouped[s][s_] = prob
        return grouped


def _init_worker(list_size, counter):
    gc.disable()
    _ctx["list_size"] = list_size
    _ctx["counter"] = counter

    # Chaque worker a sa propre connexion à Postgres
    _ctx["store"] = PostgresBatchStore(DATABASE_URL)
    _ctx["ab"] = make_alpha_beta()

    n = len(_global_state_to_idx)
    _ctx["V"] = np.frombuffer(_V_raw, dtype=np.float64)
    _ctx["policy_items"] = np.frombuffer(_pi_raw, dtype=np.int32).reshape(n, list_size)


def _transition_chunk(task):
    idx, start, end = task
    states = _global_states[start:end]
    store = _ctx["store"]
    ab = _ctx["ab"]
    policy_items = _ctx["policy_items"]
    list_size = _ctx["list_size"]
    counter = _ctx["counter"]

    cols, data, indptr = [], [], [0]

    K_WEIGHTS = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}

    # Batch des préfixes pour k=1, 2, 3
    prefixes_by_k = defaultdict(set)
    for s in states:
        for k in K_WEIGHTS:
            prefixes_by_k[k].add(encode(s[-k:]))

    raw_by_k = {}
    for k, prefixes in prefixes_by_k.items():
        raw_by_k[k] = store.batch_get_successors(list(prefixes), k)

    for i, s in enumerate(states):
        sidx = start + i
        R_list = [int(policy_items[sidx, j]) for j in range(list_size) if policy_items[sidx, j] >= 0]

        # On reconstruit les successeurs fusionnés
        merged = defaultdict(float)
        for k, weight in K_WEIGHTS.items():
            prefix = encode(s[-k:])
            for s_str, prob in raw_by_k[k].get(prefix, {}).items():
                merged[decode(s_str)[-1]] += weight * prob
        total = sum(merged.values())
        successors = {item: p / total for item, p in merged.items()} if total else {}

        # --- OPTIMISATION O(N^2) -> O(N) ---
        # Au lieu d'appeler dist_list qui recalcule la weighted_sum pour chaque beta,
        # on pré-calcule tout ici.
        n_succ = len(successors)
        if n_succ <= 1:
            dist = {}
            for r in R_list:
                dist[r] = ab.alpha(r) * successors.get(r, 0.0)
            for r_prime, q in successors.items():
                if r_prime not in R_list:
                    dist[r_prime] = (1.0 - EPSILON) * q
        else:
            alpha_map = {r: ab.alpha(r) for r in successors}
            weighted_sum = sum(alpha_map[r] * q for r, q in successors.items())

            dist = {}
            for r in R_list:
                dist[r] = alpha_map.get(r, ab.alpha(r)) * successors.get(r, 0.0)

            for r_prime, q in successors.items():
                if r_prime not in R_list:
                    if q == 0:
                        beta_r = 1.0 - EPSILON
                    else:
                        beta_r = alpha_map[r_prime] + (1.0 - weighted_sum) / ((n_succ - 1) * q)
                        beta_r = max(EPSILON, min(1.0 - EPSILON, beta_r))
                    dist[r_prime] = beta_r * q

        total_dist = sum(dist.values())
        if total_dist > 0 and abs(total_dist - 1.0) > 1e-9:
            dist = {k: v / total_dist for k, v in dist.items()}
        # ------------------------------------

        count = 0
        for r, p in dist.items():
            s_next = s[1:] + (r,)
            if s_next in _global_state_to_idx:
                cols.append(_global_state_to_idx[s_next])
                data.append(p)
                count += 1
        indptr.append(indptr[-1] + count)

        with counter.get_lock():
            counter.value += 1

    return idx, np.array(cols, dtype=np.int32), np.array(data, dtype=np.float64), np.array(indptr, dtype=np.int32)


def _improve_chunk(task):
    idx, start, end = task
    states = _global_states[start:end]
    store = _ctx["store"]
    ab = _ctx["ab"]
    V = _ctx["V"]
    list_size = _ctx["list_size"]
    counter = _ctx["counter"]

    new_policies, new_scores = [], []
    K_WEIGHTS = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}

    # Batch des préfixes
    prefixes_by_k = defaultdict(set)
    for s in states:
        for k in K_WEIGHTS:
            prefixes_by_k[k].add(encode(s[-k:]))

    raw_by_k = {}
    for k, prefixes in prefixes_by_k.items():
        raw_by_k[k] = store.batch_get_successors(list(prefixes), k)

    for i, s in enumerate(states):
        merged = defaultdict(float)
        for k, weight in K_WEIGHTS.items():
            prefix = encode(s[-k:])
            for s_str, prob in raw_by_k[k].get(prefix, {}).items():
                merged[decode(s_str)[-1]] += weight * prob
        total = sum(merged.values())
        successors = {item: p / total for item, p in merged.items()} if total else {}

        # --- OPTIMISATION O(N^2) -> O(N) ---
        n_succ = len(successors)
        if n_succ <= 1:
            scores = {}
            for r, q_r in successors.items():
                s_next = s[1:] + (r,)
                v_next = V[_global_state_to_idx[s_next]] if s_next in _global_state_to_idx else 0.0
                alpha_r = ab.alpha(r)
                scores[r] = (alpha_r * q_r - (1.0 - EPSILON) * q_r) * v_next
        else:
            alpha_map = {r: ab.alpha(r) for r in successors}
            weighted_sum = sum(alpha_map[r] * q for r, q in successors.items())

            scores = {}
            for r, q_r in successors.items():
                s_next = s[1:] + (r,)
                v_next = V[_global_state_to_idx[s_next]] if s_next in _global_state_to_idx else 0.0

                alpha_r = alpha_map[r]
                if q_r == 0:
                    beta_r = 1.0 - EPSILON
                else:
                    beta_r = alpha_r + (1.0 - weighted_sum) / ((n_succ - 1) * q_r)
                    beta_r = max(EPSILON, min(1.0 - EPSILON, beta_r))

                delta_prob = (alpha_r * q_r) - (beta_r * q_r)
                scores[r] = delta_prob * v_next
        # ------------------------------------

        top = sorted(scores.items(), key=lambda x: -x[1])[:list_size]
        new_policies.append([k for k, _ in top])
        new_scores.append([v for _, v in top])

        with counter.get_lock():
            counter.value += 1

    return idx, new_policies, new_scores


def get_observed_states_cached(k=3, cache_path="observed_states_cache.pkl", force_refresh=False):
    """Charge les états depuis Postgres et met en cache dans un .pkl local."""
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Chargement des états k={k} depuis PostgreSQL...")
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT s FROM transitions WHERE k = :k"), {"k": k}).fetchall()
    states = [decode(row[0]) for row in rows]

    with open(cache_path, "wb") as f:
        pickle.dump(states, f)
    return states


class MDPSolverParallelClean:
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or os.cpu_count()
        self.list_size = LIST_SIZE
        self.gamma = GAMMA_RL
        self.threshold = THRESHOLD
        self.R = None
        self.v_store = KeyValueStore(DATABASE_URL, namespace="V")
        self.policy_store = KeyValueStore(DATABASE_URL, namespace="policy")

    def _setup_shared_memory(self, states):
        global _global_states, _global_state_to_idx, _V_raw, _pi_raw
        n = len(states)

        _global_states = states
        _global_state_to_idx = {s: i for i, s in enumerate(states)}

        _V_raw = mp.RawArray('d', n)
        _pi_raw = mp.RawArray('i', n * self.list_size)

        V_np = np.frombuffer(_V_raw, dtype=np.float64)
        V_np.fill(0.0)
        pi_np = np.frombuffer(_pi_raw, dtype=np.int32).reshape(n, self.list_size)
        pi_np.fill(-1)

        self._load_checkpoint_postgres(n, pi_np)

        print("  Calcul du vecteur de récompenses...")
        reward_model = RewardModel()
        self.R = np.array([reward_model.get(s[-1], default=0.0) / 100.0 for s in states], dtype=np.float64)

    def _load_checkpoint_postgres(self, n, pi_np):
        print("  Vérification des checkpoints dans Postgres...")
        # Si la policy existe déjà dans Postgres, on la charge
        if self.policy_store.get(encode(_global_states[0]), default=None):
            print("  Chargement de la policy existante...")
            for i, s in enumerate(_global_states):
                policy = self.policy_store.get(encode(s), default=[])
                for j, item in enumerate(policy[:self.list_size]):
                    pi_np[i, j] = item[0] if isinstance(item, list) else item[0]
        else:
            print("  Aucun checkpoint trouvé, démarrage à froid.")

    def _checkpoint_postgres(self):
        print("  Sauvegarde (Checkpoint) dans Postgres...", end="", flush=True)
        start = time.time()
        V_np = np.frombuffer(_V_raw, dtype=np.float64)
        pi_np = np.frombuffer(_pi_raw, dtype=np.int32).reshape(len(_global_states), self.list_size)

        BATCH_SIZE = 5000
        batch_v, batch_p = {}, {}

        for i, s in enumerate(_global_states):
            key = encode(s)
            batch_v[key] = float(V_np[i])

            policy_list = []
            for j in range(self.list_size):
                if pi_np[i, j] >= 0:
                    policy_list.append([int(pi_np[i, j])])  # On ne sauve que l'item dans Postgres pour le recommender
            batch_p[key] = policy_list

            if len(batch_v) >= BATCH_SIZE:
                self.v_store.set_many(batch_v)
                self.policy_store.set_many(batch_p)
                batch_v, batch_p = {}, {}

        if batch_v:
            self.v_store.set_many(batch_v)
            self.policy_store.set_many(batch_p)
        print(f" Terminé en {time.time() - start:.1f}s")

    def _pool(self, counter):
        gc.disable()
        pool = mp.Pool(self.n_workers, initializer=_init_worker, initargs=(self.list_size, counter))
        gc.enable()
        return pool

    def policy_evaluation(self, states):
        n = len(states)
        chunk_size = max(1, n // (self.n_workers * 4))
        chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
        tasks = [(i, s, e) for i, (s, e) in enumerate(chunks)]

        counter = mp.Value("l", 0)
        start_time = time.time()

        with self._pool(counter) as pool:
            results = pool.map_async(_transition_chunk, tasks)
            while not results.ready():
                time.sleep(1.0)
                elapsed = time.time() - start_time
                done = counter.value
                speed = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / speed / 60 if speed > 0 else 0
                print(f"\r  [Build P] {done:,}/{n:,} | {speed:,.0f} états/s | ETA: {eta:.1f} min", end="", flush=True)
            print()
            results = results.get()

        indptr = np.zeros(n + 1, dtype=np.int32)
        cols_list, data_list = [], []
        ptr = 0
        for idx, cols, data, chunk_indptr in results:
            start_idx = chunks[idx][0]
            length = len(chunk_indptr) - 1
            indptr[start_idx: start_idx + length + 1] = chunk_indptr + ptr
            cols_list.append(cols)
            data_list.append(data)
            ptr += len(data)

        if cols_list:
            P = sp.csr_matrix((np.concatenate(data_list), np.concatenate(cols_list), indptr), shape=(n, n))
        else:
            P = sp.csr_matrix((n, n))

        V = np.frombuffer(_V_raw, dtype=np.float64)
        sweep = 0
        while True:
            t0 = time.time()
            V_new = self.R + self.gamma * (P @ V)
            delta = float(np.max(np.abs(V_new - V)))
            V[:] = V_new
            elapsed_ms = (time.time() - t0) * 1000
            print(f"  [Eval] sweep {sweep} — delta={delta:.6f} | {elapsed_ms:.1f}ms")
            sweep += 1
            if delta < self.threshold: break

    def policy_improvement(self, states):
        n = len(states)
        chunk_size = max(1, n // (self.n_workers * 4))
        chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
        tasks = [(i, s, e) for i, (s, e) in enumerate(chunks)]

        counter = mp.Value("l", 0)
        start_time = time.time()

        with self._pool(counter) as pool:
            results = pool.map_async(_improve_chunk, tasks)
            while not results.ready():
                time.sleep(1.0)
                elapsed = time.time() - start_time
                done = counter.value
                speed = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / speed / 60 if speed > 0 else 0
                print(f"\r  [Improve] {done:,}/{n:,} | {speed:,.0f} états/s | ETA: {eta:.1f} min", end="", flush=True)
            print()
            results = results.get()

        pi_np = np.frombuffer(_pi_raw, dtype=np.int32).reshape(n, self.list_size)
        stable = True

        for idx, new_p, new_s in results:
            start_idx = chunks[idx][0]
            for i in range(len(new_p)):
                sidx = start_idx + i
                old_items = [int(pi_np[sidx, j]) for j in range(self.list_size) if pi_np[sidx, j] >= 0]
                if new_p[i] != old_items:
                    stable = False
                for j in range(self.list_size):
                    pi_np[sidx, j] = new_p[i][j] if j < len(new_p[i]) else -1
        return stable

    def solve(self, states, max_iterations=100):
        self._setup_shared_memory(states)
        for i in range(max_iterations):
            print(f"\n=== Itération {i + 1}/{max_iterations} ({len(states):,} états) ===")
            self.policy_evaluation(states)
            stable = self.policy_improvement(states)

            # Sauvegarde directe dans Postgres à chaque itération !
            self._checkpoint_postgres()

            print(f"Itération {i + 1} — politique stable: {stable}")
            if stable: break


if __name__ == "__main__":
    # On force le rafraîchissement du cache pour bien récupérer les états de taille 3
    states = get_observed_states_cached(k=3, force_refresh=True)
    print(f"Nombre d'états à traiter : {len(states):,}")

    solver = MDPSolverParallelClean()
    solver.solve(states)