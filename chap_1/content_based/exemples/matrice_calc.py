import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

K_PAX = np.array([1, 1, 0, 0, 0])
LIFE_OF_BRIAN = np.array([0, 0, 1, 0, 0])
MEMENTO = np.array([0, 0, 0, 1, 0])
NOTORIOUS = np.array([0, 0, 0, 0, 1])
SHUTTER_ISLAND = np.array([0, 1, 0, 1, 0])
THE_DICTATOR = np.array([0, 0, 1, 0, 0])
USUAL_SUSPECTS = np.array([0, 0, 0, 0, 1])

films = {
    'K-PAX': K_PAX,
    'Life of Brian': LIFE_OF_BRIAN,
    'Memento': MEMENTO,
    'Notorious': NOTORIOUS,
    'Shutter Island': SHUTTER_ISLAND,
    'The Dictator': THE_DICTATOR,
    'Usual Suspects': USUAL_SUSPECTS
}

# Les historiques utilisent le nom du film comme clé
utilisateurs = {
    'ALICE': {'K-PAX': 5, 'Memento': 3, 'Usual Suspects': 4},
    'BOB': {'Life of Brian': 4, 'Notorious': 2, 'The Dictator': 5},
    'CINDY': {'K-PAX': 4, 'Shutter Island': 5, 'Usual Suspects': 4},
    'DAVID': {'Life of Brian': 3, 'Memento': 4, 'Usual Suspects': 5}
}


def calc_ponder_vecteur(user_history):
    vecteurs = [films[name] for name in user_history.keys()]
    notes = list(user_history.values())
    return np.average(vecteurs, axis=0, weights=notes)


def cos_sim(vecteur1, vecteur2):
    v1 = np.array(vecteur1).reshape(1, -1)
    v2 = np.array(vecteur2).reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]


def get_movie_user_history(utilisateur):
    return list(utilisateurs[utilisateur].keys())


def calc_users_profils():
    profils = {}
    for utilisateur, history in utilisateurs.items():
        profils[utilisateur] = calc_ponder_vecteur(history)
    return profils


profils = calc_users_profils()
print("--- Profils Utilisateurs (Moyenne Pondérée) ---")
for p in profils:
    print(f"{p}: {np.round(profils[p], 2)}")


cos_sim_result = {}
print("\n--- Recommandations ---")
for user in profils:
    user_history = get_movie_user_history(user)
    cos_sim_result[user] = {}

    for film_name, film_vec in films.items():
        if film_name not in user_history:
            result = cos_sim(profils[user], film_vec)
            cos_sim_result[user][film_name] = result
        else:
            cos_sim_result[user][film_name] = None

    print(f"\n{user} (a vu: {user_history})")
    recos = sorted(cos_sim_result[user].items(), key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
    for film, score in recos:
        if score is not None:
            print(f"  {film}: {score:.3f}")
        else:
            print(f"  {film}: Déjà vu")