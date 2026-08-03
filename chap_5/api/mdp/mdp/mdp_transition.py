from chap_5.api.mdp.mdp.alpha_beta import AlphaBeta
from chap_5.api.mdp.mdp.popularity import PopularityModel
from chap_5.api.mdp.config import GAMMA_BOOST

class MDPTransition:

    def __init__(self, predictive_model, gamma: float = GAMMA_BOOST):
        self.model = predictive_model
        self.ab = AlphaBeta(PopularityModel(), gamma)

    def get_successors(self, s: tuple) -> dict:
        return self.model.get_successors(s)

    def tr_mdp_list(self, s: tuple, R: list) -> dict:
        successors = self.get_successors(s)
        if not R:
            return {s[1:] + (r_prime,): q for r_prime, q in successors.items()}

        result = {}
        for r in R:
            result[s[1:] + (r,)] = self.ab.alpha(r) * successors.get(r, 0.0)
        for r_prime, q in successors.items():
            if r_prime not in R:
                result[s[1:] + (r_prime,)] = self.ab.beta(r_prime, successors) * q

        return self._normalize(result)

    def _normalize(self, dist: dict) -> dict:
        total = sum(dist.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            return {k: v / total for k, v in dist.items()}
        return dist