import sqlite3
import time
from chap_5.api.core_data.mdp.mdp_transition import MDPTransition
from chap_5.api.core_data.mdp.reward import RewardModel


class MDPSolverDB:
    """
    Résout le MDP via policy iteration, en stockant V(s) et policy(s)
    dans SQLite au lieu de dictionnaires en RAM.
    """

    def __init__(self, mdp: MDPTransition, reward_model: RewardModel,
                 db_path: str = "mdp_solver.db",
                 gamma: float = 0.9, list_size: int = 3, threshold: float = 1e-4,
                 batch_size: int = 5000):
        self.mdp = mdp
        self.reward_model = reward_model
        self.gamma = gamma
        self.list_size = list_size
        self.threshold = threshold
        self.batch_size = batch_size

        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS V (
                s TEXT PRIMARY KEY,
                value REAL NOT NULL DEFAULT 0.0
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS policy (
                s TEXT PRIMARY KEY,
                items TEXT NOT NULL DEFAULT ''
            )
        """)
        self.conn.commit()

    def _s_key(self, s: tuple) -> str:
        return ','.join(str(x) for x in s)

    def get_V(self, s: tuple) -> float:
        row = self.conn.execute(
            "SELECT value FROM V WHERE s = ?", (self._s_key(s),)
        ).fetchone()
        return row[0] if row else 0.0

    def get_policy(self, s: tuple) -> list:
        row = self.conn.execute(
            "SELECT items FROM policy WHERE s = ?", (self._s_key(s),)
        ).fetchone()
        if row and row[0]:
            return [int(x) for x in row[0].split(',')]
        return []

    def reward(self, s: tuple) -> float:
        last_item = s[-1]
        score = self.reward_model.get(last_item, default=0.0)
        return score / 100.0

    def policy_evaluation(self, states: list):
        sweep = 0
        total = len(states)

        while True:
            delta = 0.0
            start = time.time()
            batch = []

            for idx, s in enumerate(states):
                R = self.reward(s)
                policy_r = self.get_policy(s)
                dist = self.mdp.tr_mdp_list(s, policy_r)

                v_new = R + self.gamma * sum(
                    p * self.get_V(s_next) for s_next, p in dist.items()
                )
                v_old = self.get_V(s)
                delta = max(delta, abs(v_new - v_old))

                batch.append((self._s_key(s), v_new))

                if len(batch) >= self.batch_size:
                    self._flush_V(batch)
                    batch = []

                    elapsed = time.time() - start
                    speed = (idx + 1) / elapsed if elapsed > 0 else 0
                    remaining_sec = (total - (idx + 1)) / speed if speed > 0 else 0
                    remaining_min = remaining_sec / 60

                    print(
                        f"\r  sweep {sweep} — {idx+1}/{total} états "
                        f"({(idx+1)/total*100:.1f}%) | ETA: {remaining_min:.1f} min",
                        end=""
                    )

            if batch:
                self._flush_V(batch)

            elapsed = time.time() - start
            print(f"\n  [Eval] sweep {sweep} — delta={delta:.6f} — {elapsed/60:.1f} min")

            sweep += 1
            if delta < self.threshold:
                break

    def _flush_V(self, batch: list):
        self.conn.executemany(
            "INSERT INTO V (s, value) VALUES (?, ?) "
            "ON CONFLICT(s) DO UPDATE SET value = excluded.value",
            batch
        )
        self.conn.commit()

    def gain(self, s: tuple, r: int) -> float:
        s_next = s[1:] + (r,)
        p_with = self.mdp.tr_mdp(s, r).get(s_next, 0.0)
        p_without = self.mdp.get_successors(s).get(r, 0.0)
        return (p_with - p_without) * self.get_V(s_next)

    def policy_improvement(self, states: list) -> bool:
        stable = True
        start = time.time()
        batch = []
        total = len(states)

        for idx, s in enumerate(states):
            successors = self.mdp.get_successors(s)
            scores = {r: self.gain(s, r) for r in successors}
            top_items = sorted(scores, key=scores.get, reverse=True)[:self.list_size]

            old_policy = self.get_policy(s)
            if old_policy != top_items:
                stable = False

            items_str = ','.join(str(x) for x in top_items)
            batch.append((self._s_key(s), items_str))

            if len(batch) >= self.batch_size:
                self._flush_policy(batch)
                batch = []

                elapsed = time.time() - start
                speed = (idx + 1) / elapsed if elapsed > 0 else 0
                remaining_sec = (total - (idx + 1)) / speed if speed > 0 else 0
                remaining_min = remaining_sec / 60

                print(
                    f"\r  improvement — {idx+1}/{total} états "
                    f"({(idx+1)/total*100:.1f}%) | ETA: {remaining_min:.1f} min",
                    end=""
                )

        if batch:
            self._flush_policy(batch)

        elapsed = time.time() - start
        print(f"\n  [Improve] {elapsed/60:.1f} min")

        return stable

    def _flush_policy(self, batch: list):
        self.conn.executemany(
            "INSERT INTO policy (s, items) VALUES (?, ?) "
            "ON CONFLICT(s) DO UPDATE SET items = excluded.items",
            batch
        )
        self.conn.commit()

    def solve(self, states: list, max_iterations: int = 100):
        for i in range(max_iterations):
            print(f"\n=== Itération {i+1}/{max_iterations} ({len(states)} états) ===")
            self.policy_evaluation(states)
            stable = self.policy_improvement(states)
            print(f"Itération {i+1} — politique stable : {stable}")
            if stable:
                break

        self.conn.close()


if __name__ == "__main__":
    import os
    from chap_5.api.core_data.predictive_model.predictive_model import PredictiveModel

    db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/master')

    pred_model = PredictiveModel(db_url, fraction=1/15)
    mdp = MDPTransition(pred_model)
    reward_model = RewardModel(db_url)

    states = list(pred_model.get_observed_states())
    print(f"Nombre d'états à traiter : {len(states)}")

    solver = MDPSolverDB(mdp, reward_model, db_path="mdp_solver.db",
                          gamma=0.9, list_size=3, batch_size=5000)
    solver.solve(states)