import numpy as np
from scipy.stats import pearsonr

films = ['K-PAX', 'Life of Brian', 'Memento', 'Notorious', 'Shutter Island', 'The Dictator', 'Usual Suspects']

ratings = {
    'ALICE': [4, 2, 5, None, 5, None, 3],
    'BOB': [None, 4, None, 2, None, 5, None],
    'CINDY': [5, 2, 4, None, 4, None, 4],
    'DAVID': [None, 3, 4, None, 2, 5, 5],
}

users = list(ratings.keys())


def get_co_rated(u1, u2):
    return [(r1, r2) for r1, r2 in zip(ratings[u1], ratings[u2]) if r1 is not None and r2 is not None]


def pearson_sim(u1, u2):
    pairs = get_co_rated(u1, u2)

    # Pearson nécessite au moins 2 points communs (Explique pq dans mémoire)
    if len(pairs) < 2:
        return None

    notes_u1 = [p[0] for p in pairs]
    notes_u2 = [p[1] for p in pairs]
    correlation, _ = pearsonr(notes_u1, notes_u2)
    return correlation


def moyenne(user):
    return np.mean([n for n in ratings[user] if n is not None])


# Calcul matrice similarité de Pearson + print
sim_matrix = {u1: {u2: (1.0 if u1 == u2 else pearson_sim(u1, u2)) for u2 in users} for u1 in users}
print(f"{'':8}" + "".join(f"{u:8}" for u in users))
for u1 in users:
    row = f"{u1:8}" + "".join(
        f"{sim_matrix[u1][u2]:8.3f}" if sim_matrix[u1][u2] is not None else f"{'N/A':>8}" for u2 in users)
    print(row)


def predict_rating(user, film_idx):
    num, den = 0, 0
    for other in users:
        if other == user:
            continue

        sim = sim_matrix[user][other]
        note = ratings[other][film_idx]

        # Si corrélation valide et voisin a vu le film
        if sim is not None and note is not None:
            num += sim * (note - moyenne(other))
            den += abs(sim)

    if den == 0:
        return None

    return round(moyenne(user) + (num / den), 2)



print("\nPrédictions :")
for user in users:
    vus = [films[i] for i in range(len(films)) if ratings[user][i] is not None]
    print(f"\n{user} (a vu: {vus})")

    preds = []
    for i, note in enumerate(ratings[user]):
        if note is None:
            pred = predict_rating(user, i)
            if pred is not None:
                preds.append((films[i], pred))

    preds.sort(key=lambda x: x[1], reverse=True)
    for film, score in preds:
        print(f"  {film}: {score:.2f}")