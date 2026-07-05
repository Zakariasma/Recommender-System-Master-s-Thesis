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
        self.v_store = KeyValueStore(DATABASE_URL, namespace="V")  # <-- AJOUT pour récupérer V(s)
        self.temperature = temperature

    def recommend(self, s: tuple, n: int = LIST_SIZE) -> list:
        key = encode(s)
        policy = self.policy_store.get(key, default=None)
        successors = self.model.get_successors(s)


        if not successors and not policy:
            return []

        recommended_items = []
        pool = successors.copy()

        # 1. Fixe le 1er item -> celui de la policy
        if policy:
            top_item = policy[0][0]  # Le 1er élément de la policy (le meilleur selon le MDP)
            recommended_items.append(top_item)

            # On retire du pool pour pas le recommander 2 fois
            if top_item in pool:
                del pool[top_item]

        # 2. Boltzmann sur le reste, en utilisant V(s') au lieu de la proba de transition
        if len(recommended_items) < n and pool:
            # On calcule les clés des états futurs pour chaque successeur possible
            next_states_keys = {r: encode(s[1:] + (r,)) for r in pool.keys()}

            # On récupère toutes les valeurs V en une seule requête SQL (très rapide)
            v_dict = self.v_store.get_many(list(next_states_keys.values()))

            # On construit le pool pour Boltzmann : {item: V(s_next)}
            # On ne garde QUE les items dont on connaît V (ceux observés pendant l'entraînement)
            v_pool = {}
            for r, k_next in next_states_keys.items():
                v_next = v_dict.get(k_next)
                if v_next is not None:
                    v_pool[r] = v_next

            if v_pool:
                # C'est bien V(s') qui est passé à Boltzmann, conformément au papier !
                distribution = boltzmann(v_pool, self.temperature)
                rest = sample(distribution, n - len(recommended_items))
                recommended_items.extend(rest)
            else:
                # Sécurité : si aucun des états futurs n'a été observé par le MDP,
                # on retombe sur les probabilités naturelles de transition.
                distribution = boltzmann(pool, self.temperature)
                rest = sample(distribution, n - len(recommended_items))
                recommended_items.extend(rest)

        return recommended_items


if __name__ == "__main__":
    pred_model = PredictiveModel()
    recommender = Recommender(pred_model)
    print(recommender.recommend((2774, 5896, 1456)))