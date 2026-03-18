import numpy as np
import matplotlib.pyplot as plt

np.random.seed(13)
data = np.random.normal(loc=6,size=1000)
_, axes = plt.subplots(1,2, figsize=(10, 5))
axes[0].hist(data, density=True, edgecolor='blue', color='skyblue', alpha=0.4)
axes[0].plot([3, 4, 5], [0, 0, 0.4], color='red', linewidth=2)
axes[0].plot([7, 5], [0.4, 0.4], color='red', linewidth=2)
axes[0].plot([7, 8, 9], [0.4, 0, 0], color='red', linewidth=2)
axes[1].plot([0, 0.7, 0.8, 0.9, 2], [0, 0, 1, 0, 0], color='red', linewidth=2)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, None)
plt.tight_layout()
plt.show()