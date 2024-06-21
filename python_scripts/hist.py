from matplotlib import pyplot as plt
import numpy as np

x = np.random.normal(0, 1, 55)

plt.hist(x, 10, density=True, color="blue", alpha=0.5, edgecolor="black")
plt.xticks([])
plt.yticks([])

plt.axis("off")
plt.show()