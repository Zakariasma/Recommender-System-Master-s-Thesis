from chap_5.api.core_data.mdp.popularity import PopularityModel


class AlphaBeta:

    def __init__(self, popularity: PopularityModel, gamma: float):
        self.popularity = popularity
        self.gamma = gamma

    def alpha(self, r: int) -> float:
        p_r = self.popularity.get(r)
        return (self.gamma + p_r) / p_r

    def beta(self, r: int, successors: dict) -> float:
        """
        β_{s,r} = α_{s,r} +
            (1 - Σ_{r'} α_{s,r'} p(s·r'|s)) /
            ((n - 1) p(s·r|s))
        """
        EPSILON = 1e-6

        n = len(successors)
        if n <= 1:
            return 1.0 - EPSILON

        alpha_r = self.alpha(r)
        q_r = successors.get(r, 0.0)

        if q_r == 0:
            return 1.0 - EPSILON

        weighted_sum = 0.0
        for r_prime, q in successors.items():
            weighted_sum += self.alpha(r_prime) * q

        numerator = 1.0 - weighted_sum
        denominator = (n - 1) * q_r

        beta_r = alpha_r + (numerator / denominator)

        if beta_r >= 1.0:
            return 1.0 - EPSILON
        if beta_r <= 0.0:
            return EPSILON

        return beta_r