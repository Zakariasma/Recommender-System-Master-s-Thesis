from chap_5.api.core_data.mdp.alpha_beta import AlphaBeta
from chap_5.api.core_data.mdp.popularity import PopularityModel
from chap_5.api.core_data.predictive_model.predictive_model import PredictiveModel


class MDPTransition:
    """
    tr_MDP(s, r, s') = modification de tr_predict(s, s')
    via facteurs α (boost) et β (diminution).
    """

    def __init__(self, predictive_model: PredictiveModel, gamma: float = 1 / 1000):
        self.model = predictive_model
        self.ab = AlphaBeta(PopularityModel(), gamma)

    def get_successors(self, s: tuple) -> dict:
        return self.model.get_successors(s)

    def tr_mdp(self, s: tuple, r: int) -> dict:

        successors = self.get_successors(s)

        alpha_r = self.ab.alpha(r)
        beta_r = self.ab.beta(r, successors)

        result = {}

        q_r = successors.get(r, 0.0)

        # transition boostée (item recommandé)
        result[s[1:] + (r,)] = alpha_r * q_r

        # transitions atténuées (autres items)
        for r_prime, q in successors.items():
            if r_prime == r:
                continue
            result[s[1:] + (r_prime,)] = beta_r * q

        # normalisation (sécurité numérique)
        total = sum(result.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            result = {k: v / total for k, v in result.items()}

        return result

    def tr_mdp_list(self, s: tuple, R: list) -> dict:
        """
        tr_MDP(s, R, ·) : distribution de transition quand une LISTE R
        d'items est recommandée simultanément.

        Hypothèse d'indépendance (section 4.3.3) :
        - item r ∈ R  → traité comme recommandé seul (boost α, atténuation β sur les autres)
        - item r ∉ R  → probabilité inchangée du modèle prédictif de base
        """
        successors = self.get_successors(s)

        if not R:
            result = {}
            for r_prime, q in successors.items():
                result[s[1:] + (r_prime,)] = q
            return result

        result = {}

        for r in R:
            q_r = successors.get(r, 0.0)
            alpha_r = self.ab.alpha(r)
            result[s[1:] + (r,)] = alpha_r * q_r

        for r_prime, q in successors.items():
            if r_prime in R:
                continue
            beta_r = self.ab.beta(r_prime, successors)
            result[s[1:] + (r_prime,)] = beta_r * q

        total = sum(result.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            result = {k: v / total for k, v in result.items()}

        return result

if __name__ == "__main__":
    import os
    db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/master')

    pred_model = PredictiveModel(db_url, fraction=1/15)

    mdp = MDPTransition(pred_model)

    s = (995, 13870, 10)
    r = list(mdp.get_successors(s).keys())[1]  # premier successeur connu

    print(f"r = {r}")
    print(f"α({r}) = {mdp.ab.alpha(r):.4f}")
    print(f"β(s, {r}) = {mdp.ab.beta(r, mdp.get_successors(s)):.10f}")

    dist = mdp.tr_mdp(s, r)
    print(len(dist))
    print(f"\ntr_MDP(s, r={r}, ·) :")
    for s_next, prob in sorted(dist.items(), key=lambda x: -x[1])[:10]:
        print(f"  {s_next} → {prob:.4f}")
    print(f"\nSomme des probas : {sum(dist.values()):.6f}")