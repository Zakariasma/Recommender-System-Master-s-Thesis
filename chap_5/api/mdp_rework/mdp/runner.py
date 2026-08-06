from chap_5.api.mdp_rework.config import K
from chap_5.api.mdp_rework.predictive_model.predictive_model import PredictiveModel
from chap_5.api.mdp_rework.mdp.mdp_solver import MDPSolver

if __name__ == "__main__":
    # 1. Construction du modèle prédictif et génération des tables full_info
    print("=== Phase 1: Modèle Prédictif ===")
    predictive_model = PredictiveModel(k=K)
    predictive_model._build_info_index()

    # 2. Résolution du MDP
    print("\n=== Phase 2: Résolution du MDP ===")
    solver = MDPSolver()
    solver.solve_mdp()