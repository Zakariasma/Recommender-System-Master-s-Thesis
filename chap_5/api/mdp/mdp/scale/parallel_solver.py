import os
import gc
import time
import pickle
import multiprocessing as mp

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sqlalchemy import create_engine, text

from chap_5.api.mdp.config import DATABASE_URL, GAMMA_BOOST
from chap_5.api.mdp.mdp.mdp_solver import MDPSolver
from chap_5.api.mdp.predictive_model.helper.encoder import encode, decode
from chap_5.api.mdp.mdp.alpha_beta import AlphaBeta
from chap_5.api.mdp.mdp.popularity import PopularityModel
import rust_solver

# =============================================================================
#  GLOBALS : partagés via fork/COW.
# =============================================================================
_global_states = None
_global_state_to_idx = None
_global_rust_state = None
_V_raw = None
_pi_raw = None
_ctx = {}


def _init_worker(list_size, counter):
    gc.disable()
    _ctx["list_size"] = list_size
    _ctx["counter"] = counter
    _ctx["rust_state"] = _global_rust_state

    n = len(_global_state_to_idx)
    _ctx["V"] = np.frombuffer(_V_raw, dtype=np.float64)
    _ctx["policy_items"] = np.frombuffer(_pi_raw, dtype=np.int32).reshape(n, list_size)


def _transition_chunk(task):
    idx, start, end = task
    states = _global_states[start:end]
    rust_state = _ctx["rust_state"]
    list_size = _ctx["list_size"]
    policy_items = _ctx["policy_items"]
    counter = _ctx["counter"]

    states_list = [list(s) for s in states]
    tuples_list = [tuple(s) for s in states_list]

    policy_chunk = []
    for i in range(len(states_list)):
        sidx = start + i
        items = [int(policy_items[sidx, j]) for j in range(list_size) if policy_items[sidx, j] >= 0]
        policy_chunk.append(items)

    k1_prefixes = [encode(s[-1:]).encode('utf-8') for s in tuples_list]
    k2_prefixes = [encode(s[-2:]).encode('utf-8') for s in tuples_list]
    k3_prefixes = [encode(s[-3:]).encode('utf-8') for s in tuples_list]

    cols, data, indptr = rust_state.build_p_chunk(
        states_list, policy_chunk, k1_prefixes, k2_prefixes, k3_prefixes
    )

    with counter.get_lock():
        counter.value += len(states_list)

    return idx, np.array(cols, dtype=np.int32), np.array(data, dtype=np.float64), np.array(indptr, dtype=np.int32)


def _improve_chunk(task):
    idx, start, end = task
    states = _global_states[start:end]
    rust_state = _ctx["rust_state"]
    list_size = _ctx["list_size"]
    policy_items = _ctx["policy_items"]
    V = _ctx["V"]
    counter = _ctx["counter"]

    states_list = [list(s) for s in states]
    tuples_list = [tuple(s) for s in states_list]

    policy_chunk = []
    for i in range(len(states_list)):
        sidx = start + i
        items = [int(policy_items[sidx, j]) for j in range(list_size) if policy_items[sidx, j] >= 0]
        policy_chunk.append(items)

    k1_prefixes = [encode(s[-1:]).encode('utf-8') for s in tuples_list]
    k2_prefixes = [encode(s[-2:]).encode('utf-8') for s in tuples_list]
    k3_prefixes = [encode(s[-3:]).encode('utf-8') for s in tuples_list]

    is_stable, has_changed, new_policies, new_scores = rust_state.improve_chunk(
        states_list, policy_chunk, k1_prefixes, k2_prefixes, k3_prefixes,
        np.ascontiguousarray(V), list_size
    )

    with counter.get_lock():
        counter.value += len(states_list)

    return idx, is_stable, has_changed, new_policies, new_scores


def get_observed_states_cached(k=3, cache_path="observed_states_cache.pkl", force_refresh=False):
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


class MDPSolverParallelRust(MDPSolver):
    def __init__(self, mdp, reward_model, n_workers=None):
        super().__init__(mdp, reward_model)
        self.n_workers = n_workers or os.cpu_count()

    def _setup_shared_memory(self, states):
        global _global_states, _global_state_to_idx, _global_rust_state, _V_raw, _pi_raw
        n = len(states)
        L = self.list_size

        _global_states = states
        _global_state_to_idx = {s: i for i, s in enumerate(states)}

        print("  Chargement des transitions depuis Postgres vers Rust...")
        engine = create_engine(DATABASE_URL)
        caches = {}
        for k in [1, 2, 3]:
            s_list, item_list, prob_list = [], [], []
            query = text("SELECT s, s_, prob FROM transitions WHERE k = :k")
            for chunk_df in pd.read_sql(query, engine, params={"k": k}, chunksize=200000):
                for s, s_, prob in zip(chunk_df['s'], chunk_df['s_'].str.split(',').str[-1].astype(int),
                                       chunk_df['prob']):
                    s_list.append(s.encode('utf-8'))
                    item_list.append(s_)
                    prob_list.append(prob)

            caches[k] = (
                np.array(s_list, dtype='S32').view(np.uint8).ravel(),
                np.array(item_list, dtype=np.int32),
                np.array(prob_list, dtype=np.float32)
            )
            print(f"    k={k}: {len(s_list):,} transitions chargées")

        K_WEIGHTS = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}
        ab = AlphaBeta(PopularityModel(), GAMMA_BOOST)

        k1_s, k1_i, k1_p = caches[1]
        k2_s, k2_i, k2_p = caches[2]
        k3_s, k3_i, k3_p = caches[3]

        _global_rust_state = rust_solver.SolverState(
            _global_state_to_idx,
            k1_s, k1_i, k1_p,
            k2_s, k2_i, k2_p,
            k3_s, k3_i, k3_p,
            K_WEIGHTS,
            ab.popularity._p,
            ab.popularity._min,
            ab.gamma
        )

        _V_raw = mp.RawArray('d', n)
        _pi_raw = mp.RawArray('i', n * L)

        V_np = np.frombuffer(_V_raw, dtype=np.float64)
        V_np.fill(0.0)
        pi_np = np.frombuffer(_pi_raw, dtype=np.int32).reshape(n, L)
        pi_np.fill(-1)

        print("  Chargement de V et Pi depuis Postgres...")
        keys = [encode(s) for s in states]
        V_dict = self.v_store.get_many(keys)
        P_dict = self.policy_store.get_many(keys)

        for i, s in enumerate(states):
            k = encode(s)
            V_np[i] = V_dict.get(k, 0.0)
            policy = P_dict.get(k, [])
            for j, item in enumerate(policy[:L]):
                pi_np[i, j] = item[0] if isinstance(item, list) else item[0]

        self.R = np.array([self.reward(s) for s in states], dtype=np.float64)

    def _run_chunks(self, chunks, phase, worker_fn, total_states):
        tasks = [(i, start, end) for i, (start, end) in enumerate(chunks)]
        start_time = time.time()
        counter = mp.Value("l", 0)

        gc.disable()
        pool = mp.Pool(self.n_workers, initializer=_init_worker, initargs=(self.list_size, counter))
        gc.enable()

        with pool:
            async_result = pool.map_async(worker_fn, tasks)
            while not async_result.ready():
                time.sleep(1.0)
                elapsed = time.time() - start_time
                done = counter.value
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total_states - done) / speed / 60 if speed > 0 else 0
                print(f"\r  [{phase}] {done:,}/{total_states:,} | {speed:,.0f} états/s | ETA: {eta:.1f} min", end="",
                      flush=True)
            print()
            results = async_result.get()
        return results, time.time() - start_time

    def policy_evaluation(self, states):
        n = len(states)
        chunks = [(i, min(i + max(1, n // (self.n_workers * 4)), n)) for i in
                  range(0, n, max(1, n // (self.n_workers * 4)))]

        results, elapsed = self._run_chunks(chunks, f"build-P", _transition_chunk, n)

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

        P = sp.csr_matrix((np.concatenate(data_list), np.concatenate(cols_list), indptr),
                          shape=(n, n)) if cols_list else sp.csr_matrix((n, n))
        V_np = np.frombuffer(_V_raw, dtype=np.float64)

        sweep = 0
        while True:
            t0 = time.time()
            V_new = self.R + self.gamma * (P @ V_np)
            delta = float(np.max(np.abs(V_new - V_np)))
            V_np[:] = V_new
            print(f"  [Eval] sweep {sweep} — delta={delta:.6f} | {(time.time() - t0) * 1000:.1f}ms")
            sweep += 1
            if delta < self.threshold: break

        batch_v = {}
        for i, s in enumerate(states):
            batch_v[encode(s)] = float(V_np[i])
            if len(batch_v) >= 5000:
                self.v_store.set_many(batch_v)
                batch_v = {}
        if batch_v: self.v_store.set_many(batch_v)

    def policy_improvement(self, states):
        n = len(states)
        chunks = [(i, min(i + max(1, n // (self.n_workers * 4)), n)) for i in
                  range(0, n, max(1, n // (self.n_workers * 4)))]

        results, elapsed = self._run_chunks(chunks, f"improve", _improve_chunk, n)

        pi_np = np.frombuffer(_pi_raw, dtype=np.int32).reshape(n, self.list_size)
        batch_p = {}
        stable = True

        for idx, is_stable, has_changed, new_p, new_s in results:
            if not is_stable: stable = False
            start_idx = chunks[idx][0]
            for i in range(len(new_p)):
                s = states[start_idx + i]
                sidx = start_idx + i
                for j in range(self.list_size):
                    pi_np[sidx, j] = new_p[i][j] if j < len(new_p[i]) else -1
                batch_p[encode(s)] = [[item] for item in new_p[i]]
                if len(batch_p) >= 5000:
                    self.policy_store.set_many(batch_p)
                    batch_p = {}
        if batch_p: self.policy_store.set_many(batch_p)
        return stable

    def solve(self, states, max_iterations=100):
        self._setup_shared_memory(states)
        for i in range(max_iterations):
            print(f"\n=== Itération {i + 1}/{max_iterations} ({len(states):,} états) ===")
            self.policy_evaluation(states)
            stable = self.policy_improvement(states)
            print(f"Itération {i + 1} — politique stable: {stable}")
            if stable: break


if __name__ == "__main__":
    from chap_5.api.mdp.mdp.mdp_transition import MDPTransition
    from chap_5.api.mdp.mdp.reward import RewardModel
    from chap_5.api.mdp.predictive_model.predictive_model import PredictiveModel

    pred_model = PredictiveModel()
    pred_model.improve_model()

    mdp = MDPTransition(pred_model)
    reward_model = RewardModel()

    states = get_observed_states_cached(k=3, force_refresh=False)
    print(f"Nombre d'états à traiter : {len(states):,}")

    solver = MDPSolverParallelRust(mdp, reward_model)
    solver.solve(states)