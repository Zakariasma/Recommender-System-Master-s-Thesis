import time

from chap_5.api.mdp.helper.kv_store import KeyValueStore
from chap_5.api.mdp.predictive_model.helper.encoder import encode
from chap_5.api.mdp.config import DATABASE_URL, GAMMA_RL, LIST_SIZE, THRESHOLD, BATCH_SIZE
from chap_5.api.mdp.mdp.mdp_transition import MDPTransition
from chap_5.api.mdp.mdp.reward import RewardModel


class MDPSolver:
    """Policy iteration : V et policy vivent dans Postgres, pas en RAM."""

    def __init__(self, mdp: MDPTransition, reward_model: RewardModel,
                 gamma: float = GAMMA_RL, list_size: int = LIST_SIZE,
                 threshold: float = THRESHOLD, batch_size: int = BATCH_SIZE):
        self.mdp = mdp
        self.reward_model = reward_model
        self.gamma = gamma
        self.list_size = list_size
        self.threshold = threshold
        self.batch_size = batch_size
        self.v_store = KeyValueStore(DATABASE_URL, namespace="V")
        self.policy_store = KeyValueStore(DATABASE_URL, namespace="policy")

    def get_v(self, s: tuple) -> float:
        return self.v_store.get(encode(s), default=0.0)

    def get_policy(self, s: tuple) -> list:
        return self.policy_store.get(encode(s), default=[])

    def reward(self, s: tuple) -> float:
        return self.reward_model.get(s[-1], default=0.0) / 100.0

    def policy_evaluation(self, states: list):
        sweep = 0
        while True:
            delta, batch = 0.0, {}
            start = time.time()

            for idx, s in enumerate(states):
                policy_items = [item for item, _ in self.get_policy(s)]
                dist = self.mdp.tr_mdp_list(s, policy_items)
                v_new = self.reward(s) + self.gamma * sum(
                    p * self.get_v(s_next) for s_next, p in dist.items()
                )
                delta = max(delta, abs(v_new - self.get_v(s)))
                batch[encode(s)] = v_new

                if len(batch) >= self.batch_size:
                    self.v_store.set_many(batch)
                    batch = {}
                    self._log("Eval", sweep, idx, len(states), start)

            self.v_store.set_many(batch)
            print(f"\n  [Eval] sweep {sweep} — delta={delta:.6f}")
            sweep += 1
            if delta < self.threshold:
                break

    def _delta_prob(self, s: tuple, r: int) -> float:
        successors = self.mdp.get_successors(s)
        q_r = successors.get(r, 0.0)

        if q_r == 0.0:
            return 0.0

        p_with = self.mdp.ab.alpha(r) * q_r
        p_without = self.mdp.ab.beta(r, successors) * q_r

        return p_with - p_without

    def gain(self, s: tuple, r: int) -> float:
        """Gain de valeur future apporté par la recommandation de r."""
        s_next = s[1:] + (r,)
        return self._delta_prob(s, r) * self.get_v(s_next)

    def policy_improvement(self, states: list) -> bool:
        stable, batch = True, {}
        start = time.time()

        for idx, s in enumerate(states):
            successors = self.mdp.get_successors(s)

            scores = {r: self.gain(s, r) for r in successors}
            top = sorted(scores.items(), key=lambda x: -x[1])[:self.list_size]

            if [item for item, _ in top] != [item for item, _ in self.get_policy(s)]:
                stable = False
            batch[encode(s)] = top

            if len(batch) >= self.batch_size:
                self.policy_store.set_many(batch)
                batch = {}
                self._log("Improve", None, idx, len(states), start)

        self.policy_store.set_many(batch)
        print(f"\n  [Improve] {(time.time() - start) / 60:.1f} min")
        return stable

    def solve(self, states: list, max_iterations: int = 100):
        for i in range(max_iterations):
            print(f"\n=== Itération {i + 1}/{max_iterations} ({len(states)} états) ===")
            self.policy_evaluation(states)
            stable = self.policy_improvement(states)
            print(f"Itération {i + 1} — politique stable : {stable}")
            if stable:
                break

    def _log(self, label, sweep, idx, total, start):
        elapsed = time.time() - start
        speed = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (total - (idx + 1)) / speed / 60 if speed > 0 else 0
        prefix = f"sweep {sweep} — " if sweep is not None else ""
        print(f"\r  {label} — {prefix}{idx + 1}/{total} ({(idx + 1) / total * 100:.1f}%) | ETA: {eta:.1f} min", end="")


if __name__ == "__main__":
    from chap_5.api.mdp.predictive_model.predictive_model import PredictiveModel

    pred_model = PredictiveModel()
    pred_model.improve_model()

    mdp = MDPTransition(pred_model)
    reward_model = RewardModel()
    states = pred_model.get_observed_states()
    print(f"Nombre d'états à traiter : {len(states)}")

    MDPSolver(mdp, reward_model).solve(states)