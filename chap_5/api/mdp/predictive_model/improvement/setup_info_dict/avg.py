import numpy as np

from chap_5.api.mdp.predictive_model.improvement.setup_info_dict.data.sql import (
    create_full_info_database,
    init_full_info_db,
)
from chap_5.api.mdp.predictive_model.improvement.skipping.data.sql import (
    create_skipping_database,
    retrieve_distinct_states,
    retrieve_skipping_full_info_for_batch,
)
from chap_5.api.mdp.predictive_model.improvement.similarity.data.sql import (
    create_similarity_database,
    retrieve_similarity_full_info_for_batch,
)


BATCH_SIZE = 10_000
K_VALUES = [1, 2, 3]


def count_unique_successors(successors):
    """Compte les successeurs uniques."""
    if not successors:
        return 0

    return len({tuple(successor) for successor in successors})


skipping_engine = create_skipping_database()
similarity_engine = create_similarity_database()

# Récupération des états observés
observed_states = retrieve_distinct_states(skipping_engine, k=3)
batch_states = observed_states[:BATCH_SIZE]
batch_set = set(batch_states)

print(f"États analysés : {len(batch_states):,}")

# Récupération des données du batch
raw_skipping = retrieve_skipping_full_info_for_batch(
    skipping_engine,
    batch_set,
)
raw_similarity = retrieve_similarity_full_info_for_batch(
    similarity_engine,
    batch_set,
)

successors_by_k = {k: [] for k in K_VALUES}
all_successors_per_state = []

for state in batch_states:
    skip_data = raw_skipping.get(state, {})
    sim_data = raw_similarity.get(state, {})

    state_total_successors = set()

    for k in K_VALUES:
        skip_successors = skip_data.get(k, ([], []))[0]
        sim_successors = sim_data.get(k, ([], []))[0]

        successors = {
            tuple(successor)
            for successor in list(skip_successors) + list(sim_successors)
        }

        successors_by_k[k].append(len(successors))
        state_total_successors.update(successors)

    all_successors_per_state.append(len(state_total_successors))

# Résultats par k
for k in K_VALUES:
    values = successors_by_k[k]
    print(
        f"k={k} : moyenne = {np.mean(values):.2f} successeurs uniques par état "
        f"(min={np.min(values)}, max={np.max(values)})"
    )

# Résultat global, union des successeurs de k=1, k=2 et k=3
print(
    f"Global : moyenne = {np.mean(all_successors_per_state):.2f} "
    f"successeurs uniques par état "
    f"(min={np.min(all_successors_per_state)}, "
    f"max={np.max(all_successors_per_state)})"
)