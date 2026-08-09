from chap_5.api.mdp.config import K
from chap_5.api.mdp.predictive_model.predictive_model import PredictiveModel
from chap_5.api.mdp.mdp.mdp_solver import MDPSolver

def init_mdp():
    print("\nModèle Prédictif")
    predictive_model = PredictiveModel(k=K)
    predictive_model._build_info_index()

    print("\nRésolution du MDP")
    solver = MDPSolver()
    solver.solve_mdp()