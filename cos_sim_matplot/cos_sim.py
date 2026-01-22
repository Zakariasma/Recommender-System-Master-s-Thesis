import numpy as np
import matplotlib.pyplot as plt
from normalized import plot_normalized
from non_normalized import plot_non_normalized

user_A = np.array([2, 1])
user_B = np.array([10, 5])
item = np.array([5, 2])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

plot_non_normalized(axes[0], user_A, user_B, item)
plot_normalized(axes[1], user_A, user_B, item)

plt.show()