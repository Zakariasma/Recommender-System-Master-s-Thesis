import numpy as np

def plot_non_normalized(axes, user_A, user_B, item):
    axes.quiver(0, 0, user_A[0], user_A[1], angles='xy', scale_units='xy', scale=1, color='red', label='User A')
    axes.quiver(0, 0, user_B[0], user_B[1], angles='xy', scale_units='xy', scale=1, color='blue', label='User B')
    axes.quiver(0, 0, item[0], item[1], angles='xy', scale_units='xy', scale=1, color='green', label='Item')
    axes.set_title("Vecteurs non normalisés")
    axes.set_xlim(0, 12)
    axes.set_ylim(0, 6)
    axes.set_aspect('equal')
    axes.grid()
    axes.legend()