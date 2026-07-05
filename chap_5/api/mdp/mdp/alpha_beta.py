from chap_5.api.mdp.mdp.popularity import PopularityModel

EPSILON = 1e-6


class AlphaBeta:
    def __init__(self, popularity: PopularityModel, gamma: float):
        self.popularity = popularity
        self.gamma = gamma

    def alpha(self, r: int) -> float:
        p_r = self.popularity.get(r)
        return (self.gamma + p_r) / p_r

    def beta(self, r: int, successors: dict) -> float:
        n = len(successors)
        if n <= 1:
            return 1.0 - EPSILON

        q_r = successors.get(r, 0.0)
        if q_r == 0:
            return 1.0 - EPSILON

        weighted_sum = sum(self.alpha(r_prime) * q for r_prime, q in successors.items())
        beta_r = self.alpha(r) + (1.0 - weighted_sum) / ((n - 1) * q_r)

        return min(max(beta_r, EPSILON), 1.0 - EPSILON)

    def beta_fast(self, alpha_r: float, q_r: float, n_succ: int, weighted_sum: float) -> float:
        """
        Version optimisée O(1).
        Au lieu de recalculer 'weighted_sum' et 'alpha(r)' à chaque appel,
        on les passe en paramètres car ils ont déjà été calculés en amont
        dans la boucle principale du worker. Cela évite la complexité O(N^2).
        """
        if n_succ <= 1:
            return 1.0 - EPSILON
        if q_r == 0:
            return 1.0 - EPSILON

        beta_r = alpha_r + (1.0 - weighted_sum) / ((n_succ - 1) * q_r)
        return min(max(beta_r, EPSILON), 1.0 - EPSILON)
