from chap_5.api.mdp.dataset.init_dataset import init_dataset
from chap_5.api.mdp.mdp.mdp_solver import MDPSolver
from chap_5.api.mdp.mdp.mdp_transition import MDPTransition
from chap_5.api.mdp.mdp.reward import RewardModel
from chap_5.api.mdp.predictive_model.predictive_model import PredictiveModel
from chap_5.api.mdp.config import BOOTSTRAP


def boostrap_app():
    init_dataset()

    if not BOOTSTRAP:
        print("\n CALCUL DU MODÈLE MDP")
        pred_model = PredictiveModel()
        pred_model.improve_model()
        mdp = MDPTransition(pred_model)
        reward_model = RewardModel()
        states = pred_model.get_observed_states()
        MDPSolver(mdp, reward_model).solve(states)
    else:
        print("\nBOOTSTRAP=True : Les données MDP ont été chargées")


if __name__ == "__main__":
    boostrap_app()