import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

x = np.linspace(0, 1, 100)
plt.plot(x, stats.norm.pdf(x, 0.43299443, 0.11952501), color="green")
plt.plot(x, stats.norm.pdf(x, 0.5825024, 0.16159964), color="red")
plt.show()
