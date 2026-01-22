import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def plot_normalized(axes, user_A, user_B, item):
    user_A_norm = normalize(user_A)
    user_B_norm = normalize(user_B)
    item_norm = normalize(item)

    axes.quiver(0, 0, user_A_norm[0], user_A_norm[1], angles='xy', scale_units='xy', scale=1, color='red', label='User A')
    axes.quiver(0, 0, user_B_norm[0], user_B_norm[1], angles='xy', scale_units='xy', scale=1, color='blue', label='User B')
    axes.quiver(0, 0, item_norm[0], item_norm[1], angles='xy', scale_units='xy', scale=1, color='green', label='Item')
    axes.set_title("Vecteurs normalisés")
    axes.set_xlim(0, 1.2)
    axes.set_ylim(0, 1.2)
    axes.set_aspect('equal')
    axes.grid()
    axes.legend()