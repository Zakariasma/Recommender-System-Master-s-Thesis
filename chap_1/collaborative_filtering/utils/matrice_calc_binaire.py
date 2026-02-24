import numpy as np

films_list = ['K-PAX', 'Life of Brian', 'Memento', 'Notorious', 'Shutter Island', 'The Dictator', 'Usual Suspects']

ratings = {
    'ALICE': [1, 0, 1, 0, 0, 0, 1],
    'BOB':   [0, 1, 0, 1, 0, 1, 0],
    'CINDY': [1, 0, 0, 0, 1, 0, 1],
    'DAVID': [0, 1, 1, 0, 0, 0, 1],
}

users = list(ratings.keys())


def cosine_sim(u1, u2):
    r1 = ratings[u1]
    r2 = ratings[u2]
    dot   = sum(a * b for a, b in zip(r1, r2))
    norm1 = np.sqrt(sum(a ** 2 for a in r1))
    norm2 = np.sqrt(sum(b ** 2 for b in r2))
    return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0


cosine_matrix = {
    u1: {u2: 1.0 if u1 == u2 else cosine_sim(u1, u2) for u2 in users}
    for u1 in users
}


def predict_score(user, film_idx):
    num, den = 0, 0
    for other in users:
        sim  = cosine_matrix[user][other]
        note = ratings[other][film_idx]
        if other == user or note == 0:
            continue
        num += sim * note
        den += abs(sim)
    k = 1 / den if den > 0 else None
    return round(k * num, 3) if k is not None else None


# Matrice cosinus
for u1 in users:
    for u2 in users:
        print(f"{u1} / {u2}: {cosine_matrix[u1][u2]:.3f}")

# Recommandations
for user in users:
    vus = [films_list[i] for i in range(len(films_list)) if ratings[user][i] == 1]
    print(f"\n{user} (vu: {vus})")
    predictions = sorted(
        [(films_list[i], predict_score(user, i)) for i in range(len(films_list))
         if ratings[user][i] == 0 and predict_score(user, i) is not None],
        key=lambda x: x[1], reverse=True
    )
    for film, score in predictions:
        print(f"  {film}: {score:.3f}")
