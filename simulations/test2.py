import numpy as np
mu, sigma = 0, 0.4 # mean and standard deviation
s = np.random.normal(mu, sigma)
print(s)
"""

import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()"""