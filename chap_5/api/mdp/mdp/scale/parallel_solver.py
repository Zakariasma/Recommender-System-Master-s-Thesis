import os
import time
import pickle
import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
from sqlalchemy import create_engine, text

from chap_5.api.mdp.config import DATABASE_URL
from chap_5.api.mdp.mdp.mdp_solver import MDPSolver
from chap_5.api.mdp.mdp.scale.pool_manager import chunkify, run_parallel_tasks
from chap_5.api.mdp.mdp.scale.worker import transition_chunk_worker, init_worker, improve_chunk_worker
from chap_5.api.mdp.predictive_model.state_key import encode, decode


class MDPSolverParallel(MDPSolver):
    """Solveur Parallèle héritant de MDPSolver pour réutiliser la logique de base."""

    def __init__(self, mdp, reward_model, n_workers=None):
        super().__init__(mdp, reward_model)
        self.n_workers = n_workers or os.cpu_count()
        self._V_raw = None
        self._pi_raw = None

    def policy_evaluation(self, states: list):
        n = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}

        self._V_raw = mp.RawArray('d', n)
        V_np = np.frombuffer(self._V_raw, dtype=np.float64)

        self._pi_raw = mp.RawArray('i', n * self.list_size)
        pi_np = np.frombuffer(self._pi_raw, dtype=np.int32).reshape(n, self.list_size)
        pi_np.fill(-1)

        # On charge V et Pi depuis Postgres
        for i, s in enumerate(states):
            V_np[i] = self.get_v(s)
            policy = self.get_policy(s)
            for j, item in enumerate(policy[:self.list_size]):
                pi_np[i, j] = item[0] if isinstance(item, list) else item[0]

        R = np.array([self.reward(s) for s in states], dtype=np.float64)

        chunks = chunkify(n, self.n_workers * 4)
        tasks = [(i, s, e) for i, (s, e) in enumerate(chunks)]

        initargs = (states, state_to_idx, self._V_raw, self._pi_raw, self.list_size)
        results = run_parallel_tasks(transition_chunk_worker, tasks, n, self.n_workers, init_worker, initargs)

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

        sweep = 0
        while True:
            t0 = time.time()
            V_new = R + self.gamma * (P @ V_np)
            delta = float(np.max(np.abs(V_new - V_np)))
            V_np[:] = V_new
            elapsed_ms = (time.time() - t0) * 1000
            print(f"  [Eval] sweep {sweep} — delta={delta:.6f} | {elapsed_ms:.1f}ms")
            sweep += 1
            if delta < self.threshold: break

        # Sauvegarde de V dans Postgres
        batch_v = {}
        for i, s in enumerate(states):
            batch_v[encode(s)] = float(V_np[i])
            if len(batch_v) >= 5000:
                self.v_store.set_many(batch_v)
                batch_v = {}
        if batch_v: self.v_store.set_many(batch_v)

    def policy_improvement(self, states: list) -> bool:
        n = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}

        chunks = chunkify(n, self.n_workers * 4)
        tasks = [(i, s, e) for i, (s, e) in enumerate(chunks)]

        initargs = (states, state_to_idx, self._V_raw, self._pi_raw, self.list_size)
        results = run_parallel_tasks(improve_chunk_worker, tasks, n, self.n_workers, init_worker, initargs)

        pi_np = np.frombuffer(self._pi_raw, dtype=np.int32).reshape(n, self.list_size)
        batch_p = {}
        stable = True

        for idx, new_p, new_s in results:
            start_idx = chunks[idx][0]
            for i in range(len(new_p)):
                s = states[start_idx + i]
                sidx = start_idx + i

                old_items = [int(pi_np[sidx, j]) for j in range(self.list_size) if pi_np[sidx, j] >= 0]
                if new_p[i] != old_items:
                    stable = False

                # Mise à jour en mémoire partagée pour la prochaine itération
                for j in range(self.list_size):
                    pi_np[sidx, j] = new_p[i][j] if j < len(new_p[i]) else -1

                batch_p[encode(s)] = [[item] for item in new_p[i]]
                if len(batch_p) >= 5000:
                    self.policy_store.set_many(batch_p)
                    batch_p = {}

        if batch_p: self.policy_store.set_many(batch_p)
        return stable


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


if __name__ == "__main__":
    from chap_5.api.mdp.mdp.mdp_transition import MDPTransition
    from chap_5.api.mdp.mdp.reward import RewardModel
    from chap_5.api.mdp.predictive_model.predictive_model import PredictiveModel

    pred_model = PredictiveModel()
    pred_model.improve_model()

    mdp = MDPTransition(pred_model)
    reward_model = RewardModel()

    states = get_observed_states_cached(k=3, force_refresh=True)
    print(f"Nombre d'états à traiter : {len(states):,}")

    solver = MDPSolverParallel(mdp, reward_model)
    solver.solve(states)