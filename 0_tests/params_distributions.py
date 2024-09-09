import numpy as np
from numpy.random import default_rng
from scipy import stats
from matplotlib import pyplot as plt
from lime.model import gaussian_model

rnd = np.random.RandomState(12345)

seed_val = 1234
rnd_size = 1000
# rng = default_rng(12345)
# rng = np.random.RandomState(12345)
rnd = np.random.RandomState()
print(rnd.normal(0, 10, 3))
rnd = np.random.RandomState()
print(rnd.normal(0, 10, 3))


# rng = np.random.default_rng(0)
# print(rng.normal(0, 10, 3))
#
# print(rng.normal(0, 10, 3))
#
# rng = np.random.default_rng(0)
# print(rng.normal(0, 10, 3))

# x = np.linspace(0, 5, 200)
# pdf = stats.cauchy.pdf(x, scale=0.5)
# label = r'Cauchy $\beta = {}$'.format(0.5)

x = np.linspace(-5, 5, 200)
pdf = stats.norm.pdf(x, loc=-1, scale=1)
label = r'Normal $\mu = {}$, $\sigma={}$'.format(-1, 1)

rnd_array = stats.norm.rvs(loc=-1, scale=1, size=rnd_size, random_state=seed_val)
# print(rnd_array)


fig, ax = plt.subplots()
ax.plot(x, pdf, label=label)
ax.hist(rnd_array, density=True, histtype='stepfilled', alpha=0.2)
ax.legend()
ax.update({'xlabel': 'x', 'ylabel': 'f(x)', 'title': 'Distribution'})
plt.show()







