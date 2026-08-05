from chap_5.api.mdp_rework.config import GAMMA_BOOST

EPSILON = 1e-6


class AlphaBeta:
    def __init__(self):
        pass

    def alpha(self, popularity: float) -> float:
        return (GAMMA_BOOST + popularity) / popularity

