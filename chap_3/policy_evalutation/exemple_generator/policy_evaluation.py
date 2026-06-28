import json

with open("portes_mdp.json", "r", encoding="utf-8") as f:
    model = json.load(f)

gamma = model["gamma"]
states = model["states"]

S = [s for s, info in states.items() if info.get("type") != "terminal"]


def expected_return_for_action(state, action, V_current):
    action_data = states[state]["actions"][action]

    hole = action_data["hole_transition"]
    safe = action_data["safe_transition"]

    hole_return = hole["prob"] * (
        hole["reward"] + gamma * V_current[hole["next_state"]]
    )

    safe_return = safe["prob"] * sum(
        item["prob"] * (item["reward"] + gamma * V_current[safe["next_state"]])
        for item in safe["reward_distribution"]
    )

    return hole_return + safe_return


def policy_evaluation(policy, threshold=0.1):
    V = {s: 0.0 for s in states}
    iteration = 0
    history = []

    while True:
        iteration += 1
        delta = 0.0
        V_new = V.copy()

        for s in S:
            new_v = sum(
                action_prob * expected_return_for_action(s, action, V)
                for action, action_prob in policy.items()
            )
            V_new[s] = new_v
            delta = max(delta, abs(V[s] - new_v))

        V = V_new
        history.append((iteration, V["s_0"]))

        if delta < threshold:
            break

    return V, history, iteration


policy = {
    "gauche": 0.5,
    "droite": 0.5,
}

V, history, iteration = policy_evaluation(policy, threshold=0.1)

print("Nombre d'iterations :", iteration)
for k, val in history:
    print(f"iteration {k}: {val}")