import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calc_ponder_vecteur(liste_vecteurs):
    return np.mean(liste_vecteurs, axis=0)

def cos_sim(vecteur1, vecteur2):
    v1 = np.array(vecteur1).reshape(1, -1)
    v2 = np.array(vecteur2).reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]

K_PAX = [1, 1, 0, 0, 0]
LIFE_OF_BRIAN = [0, 0, 1, 0, 0]
MEMENTO = [0, 0, 0, 1, 0]
NOTORIOUS = [0, 0, 0, 0, 1]
SHUTTER_ISLAND = [0, 1, 0, 1, 0]
THE_DICTATOR = [0, 0, 1, 0, 0]
USUAL_SUSPECTS = [0, 0, 0, 0, 1]

films = {
    'K-PAX': K_PAX,
    'Life of Brian': LIFE_OF_BRIAN,
    'Memento': MEMENTO,
    'Notorious': NOTORIOUS,
    'Shutter Island': SHUTTER_ISLAND,
    'The Dictator': THE_DICTATOR,
    'Usual Suspects': USUAL_SUSPECTS
}

ALICE_VUS = [K_PAX, MEMENTO, USUAL_SUSPECTS]
BOB_VUS = [LIFE_OF_BRIAN, NOTORIOUS, THE_DICTATOR]
CINDY_VUS = [K_PAX, SHUTTER_ISLAND, USUAL_SUSPECTS]
DAVID_VUS = [LIFE_OF_BRIAN, MEMENTO, USUAL_SUSPECTS]

utilisateurs = {
    'ALICE': ALICE_VUS,
    'BOB': BOB_VUS,
    'CINDY': CINDY_VUS,
    'DAVID': DAVID_VUS
}


def calc_users_profils():
    profils = {}
    for utilisateur, vus in utilisateurs.items():
        result_user_profil = calc_ponder_vecteur(vus)
        profils[utilisateur] = result_user_profil
    return profils


# Calcul matrice user profils
profils = calc_users_profils()
for p in profils:
    print(f"{p}: {profils[p]}\n")






