from chap_5.api.mdp.config import DATABASE_URL, LIST_SIZE, BOLTZMANN_TEMPERATURE
from chap_5.api.mdp.helper.kv_store import KeyValueStore
from chap_5.api.mdp.predictive_model.state_key import encode
from chap_5.api.mdp.predictive_model.predictive_model import PredictiveModel
from chap_5.api.mdp.serving.boltzmann import boltzmann, sample


class Recommender:
    """Point d'entrée unique pour servir une recommandation à partir d'un état utilisateur."""

    def __init__(self, predictive_model: PredictiveModel, temperature: float = BOLTZMANN_TEMPERATURE):
        self.model = predictive_model
        self.policy_store = KeyValueStore(DATABASE_URL, namespace="policy")
        self.temperature = temperature

    def recommend(self, s: tuple, n: int = LIST_SIZE) -> list:
        key = encode(s)
        policy = self.policy_store.get(key, default=None)
        successors = self.model.get_successors(s)
        print(policy)
        print(successors)

        if not successors and not policy:
            return []

        recommended_items = []
        pool = successors.copy()

        # Fixe le 1er item -> celui de la policy
        if policy:
            top_item = policy[0][0]  # Le 1er élément de la policy (le meilleur selon le MDP)
            recommended_items.append(top_item)

            # On retire du pool pour pas le recommander 2 fois
            if top_item in pool:
                del pool[top_item]

        # Boltzmann sur le reste
        if len(recommended_items) < n and pool:
            distribution = boltzmann(pool, self.temperature)
            rest = sample(distribution, n - len(recommended_items))
            recommended_items.extend(rest)

        return recommended_items


if __name__ == "__main__":
    pred_model = PredictiveModel()
    recommender = Recommender(pred_model)
    print(recommender.recommend((2774,)))