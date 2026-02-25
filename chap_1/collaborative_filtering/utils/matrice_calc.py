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


def pearson_sim(u1, u2):
    # Prend deux notes d'utilisateurs et retourne liste de pair de notes
    # Zip →  (4,N) (2,4) (5,N)...
    pairs = [(n1, n2) for n1, n2 in zip(ratings[u1], ratings[u2]) if n1 is not None and n2 is not None]
    # Pearson a besoin de len >= 2 sinon crash, explique pq dans mémoire Table 1.9
    if len(pairs) < 2:
        return None
    corr, _ = pearsonr(*zip(*pairs))
    return corr


def moyenne(user):
    # Avg de toutes les notes
    return np.mean([n for n in ratings[user] if n is not None])


pearson_matrix = {
    u1: {u2: 1.0 if u1 == u2 else pearson_sim(u1, u2) for u2 in users}
    for u1 in users
}


def predict_rating(user, film_idx):
    num, den = 0, 0
    for other in users:
        sim = pearson_matrix[user][other]
        note = ratings[other][film_idx]

        # On skip si c'est user lui meme ou pas assez de film co-notés ou voisin a pas noté de film
        if other == user or sim is None or note is None:
            continue
        num += sim * (note - moyenne(other))
        den += abs(sim)
    k = 1 / den if den > 0 else None # Evite div par 0
    return round(moyenne(user) + k * num, 2) if k is not None else None


# Matrice de Pearson
print(f"{'':8}" + "".join(f"{u:8}" for u in users))
for u1 in users:
    row = f"{u1:8}" + "".join(
        f"{pearson_matrix[u1][u2]:8.3f}" if pearson_matrix[u1][u2] is not None else f"{'N/A':>8}"
        for u2 in users
    )
    print(row)

# Prédit note, recommande avec notes élevé (heuristique par exemple > 3.5)
for user in users:
    vus = [(films[i], ratings[user][i]) for i in range(len(films)) if ratings[user][i] is not None]
    print(f"\n{user} (noté: {vus})")
    predictions = sorted(
        [(films[i], predict_rating(user, i)) for i in range(len(films)) if
         ratings[user][i] is None and predict_rating(user, i) is not None],
        key=lambda x: x[1], reverse=True
    )
    for film, score in predictions:
        print(f"  {film}: {score:.2f}")
