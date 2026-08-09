from chap_5.api.mdp.config import GAMMA_BOOST

EPSILON = 1e-5


class AlphaBeta:
    def __init__(self, item_counts: dict, total_items: int):
        self.n = len(item_counts)
        self.gamma = GAMMA_BOOST
        self.p_r = {item: count / total_items for item, count in item_counts.items()}

    def compute_alpha(self, r: int) -> float:
        p = self.p_r.get(r, 0.0)
        if p == 0.0:
            return 1.0 + self.gamma
        return (self.gamma + p) / p

    def compute_beta(self, alpha: float, p_s_r: float, sum_alpha_p: float) -> float:
        if self.n <= 1 or p_s_r == 0.0:
            return EPSILON
        beta = alpha + (1.0 - sum_alpha_p) / ((self.n - 1) * p_s_r)
        if beta < 0:
            beta = EPSILON
        else:
            beta = max(EPSILON, beta - EPSILON)
        return min(beta, 1.0 - EPSILON)