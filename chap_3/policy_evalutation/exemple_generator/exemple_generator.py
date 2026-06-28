import json
import random
import networkx as nx

from chap_3.policy_evalutation.exemple_generator.models.action import ActionModel
from chap_3.policy_evalutation.exemple_generator.models.mdp import MDPModel
from chap_3.policy_evalutation.exemple_generator.models.reward_prob import RewardProb
from chap_3.policy_evalutation.exemple_generator.models.state import StateModel
from chap_3.policy_evalutation.exemple_generator.models.transition import Transition

SEED = 42
LEVELS = 9
GAMMA = 0.9
ROOT = "s_0"
TERMINAL_HOLE = "terminal_hole"
TERMINAL_EXIT = "terminal_exit"
ACTIONS = ("gauche", "droite")
REWARD_POOL = [
    5, 10, 20, 50, 100, 200, 500, 1000,
    5000, 10000, 50000, 100000, 1000000,
    10000000, 1000000000
]


def random_reward_distribution():
    n = random.choice([2, 3])
    rewards = random.sample(REWARD_POOL, n)
    weights = [random.random() for _ in range(n)]
    total = sum(weights)
    probs = [round(w / total, 2) for w in weights]
    probs[-1] = round(probs[-1] + (1 - sum(probs)), 2)
    return [RewardProb(reward=r, prob=p) for r, p in zip(rewards, probs)]


def add_action_edges(graph, state, action, next_state, safe_prob, hole_prob, reward_distribution):
    graph.add_edge(
        state,
        next_state,
        action=action,
        transition_type="safe",
        prob=safe_prob,
        reward_distribution=[item.model_dump() for item in reward_distribution],
    )
    graph.add_edge(
        state,
        TERMINAL_HOLE,
        action=action,
        transition_type="hole",
        prob=hole_prob,
        reward=0,
    )


def build_graph():
    random.seed(SEED)
    graph = nx.MultiDiGraph()

    graph.add_node(ROOT, level=0)
    graph.add_node(TERMINAL_HOLE, type="terminal", description="Tomber dans un trou, perte du butin")
    graph.add_node(TERMINAL_EXIT, type="terminal", description="Fin du parcours")

    graph.add_node("s_1_0_g", level=1)
    graph.add_node("s_1_0_d", level=1)

    add_action_edges(
        graph, ROOT, "gauche", "s_1_0_g", 0.9, 0.1,
        [RewardProb(reward=1000, prob=0.3), RewardProb(reward=10, prob=0.7)]
    )
    add_action_edges(
        graph, ROOT, "droite", "s_1_0_d", 0.1, 0.9,
        [
            RewardProb(reward=1_000_000, prob=0.8),
            RewardProb(reward=10_000_000, prob=0.19),
            RewardProb(reward=1_000_000_000, prob=0.01),
        ]
    )

    frontier = ["s_1_0_g", "s_1_0_d"]

    for level in range(1, LEVELS):
        new_frontier = []

        for state in frontier:
            for action in ACTIONS:
                hole_prob = round(random.uniform(0.1, 0.9), 2)
                safe_prob = round(1 - hole_prob, 2)

                if level < LEVELS - 1:
                    next_state = f"s_{level + 1}_{len(new_frontier)}_{action[0]}"
                    graph.add_node(next_state, level=level + 1)
                    new_frontier.append(next_state)
                else:
                    next_state = TERMINAL_EXIT

                add_action_edges(
                    graph,
                    state,
                    action,
                    next_state,
                    safe_prob,
                    hole_prob,
                    random_reward_distribution(),
                )

        frontier = new_frontier

    return graph


def graph_to_model(graph: nx.MultiDiGraph) -> MDPModel:
    states = {}

    for node, attrs in graph.nodes(data=True):
        state_actions = {}

        if attrs.get("type") != "terminal":
            outgoing = graph.out_edges(node, keys=True, data=True)

            grouped = {}
            for _, target, _, edge_attrs in outgoing:
                action = edge_attrs["action"]
                grouped.setdefault(action, {})
                grouped[action][edge_attrs["transition_type"]] = (target, edge_attrs)

            for action, transitions in grouped.items():
                safe_target, safe_attrs = transitions["safe"]
                hole_target, hole_attrs = transitions["hole"]

                state_actions[action] = ActionModel(
                    safe_transition=Transition(
                        prob=safe_attrs["prob"],
                        next_state=safe_target,
                        reward_distribution=[
                            RewardProb(**item) for item in safe_attrs["reward_distribution"]
                        ],
                    ),
                    hole_transition=Transition(
                        prob=hole_attrs["prob"],
                        next_state=hole_target,
                        reward=hole_attrs["reward"],
                    ),
                )

        states[node] = StateModel(
            level=attrs.get("level"),
            type=attrs.get("type"),
            description=attrs.get("description"),
            actions=state_actions or None,
        )

    return MDPModel(
        description="Arbre de portes pour MDP avec niveaux aléatoires.",
        initial_state=ROOT,
        rule_if_hole="Si on tombe dans un trou, on perd tout le butin et on va dans un etat terminal.",
        gamma=GAMMA,
        states=states,
    )


if __name__ == "__main__":
    graph = build_graph()
    model = graph_to_model(graph)

    with open("portes_mdp.json", "w", encoding="utf-8") as f:
        json.dump(model.model_dump(exclude_none=True), f, ensure_ascii=False, indent=2)