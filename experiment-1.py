import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spstats
import scipy.integrate as spint
from tqdm import tqdm

np.random.seed(1243)
data = np.hstack([np.random.normal(loc=4, scale=2, size=500), np.random.normal(loc=10, scale=2, size=500)])
# data = np.hstack([np.random.chisquare(df=5, size=500), np.random.uniform(low=10, high=15, size=500)])

plt.hist(data)
# plt.title('1:1-смесь распределений chi-square(5) и Uniform[10;15]')
plt.title('1:1-смесь распределений N(4,2) и N(10,2)')
plt.show()

def a_part_good(x): # FS(0,4,11,15)
  if x < 0:
    return 0
  if x > 15:
    return 0
  if 4 <= x <= 11:
    return 1
  if 0 <= x < 4:
    return x / 4
  if 11 < x <= 15:
    return (15 - x) / 4

def a_part_bad(x): # FS(0,2,6,10)
  if x < 0:
    return 0
  if x > 10:
    return 0
  if 2 <= x <= 6:
    return 1
  if 0 <= x < 2:
    return x / 2
  if 6 < x <= 10:
    return (10 - x) / 4
  
def f_hist(data):
  heights, bins = np.histogram(data, density=True)
  def f(x):
    if x < bins[0]:
      return 0
    if x >= bins[-1]:
      return 0
    for i in range(len(heights)):
      if bins[i] <= x < bins[i + 1]:
        return heights[i]
    return -1
  return f

def plot_function(f, min, max):
  step = (max - min) / 40
  xs = np.arange(min, max + step, step)
  ys = [f(x) for x in xs]
  plt.plot(xs, ys)

def get_distributions(data):
  mean = np.mean(data)
  std = np.std(data)
  min = np.min(data)
  max = np.max(data)

  normals = [spstats.norm(loc, scale).pdf for loc in np.arange(min, max, (max - min) / 20) for scale in np.arange(0.1, 2 * std, std / 10)]
  expons = [spstats.expon(loc, scale).pdf for loc in np.arange(min, max, (max - min) / 20) for scale in np.arange(0.1, 2 * std, std / 10)]
  uniform = [spstats.uniform(loc, scale).pdf 
             for loc in np.arange(min - (max - min) / 2, min + (max - min) / 2, (max + min) / 20) 
             for scale in np.arange((max - min) / 2, 3 * (max - min) / 2, (max - min) / 10)]
  print(len(normals), len(expons), len(uniform))
  return normals + expons + uniform

ds = get_distributions(data)
f_h = f_hist(data)

def euclide_distance(f, g, min, max):
  return spint.quad(lambda x: (f(x) - g(x)) ** 2, min, max)[0]

def manhattan_distance(f, g, min, max):
  xs = np.arange(min, max, (max-min)/1000)
  ys = [np.abs(f(x) - g(x)) for x in xs]
  return np.max(ys)

def s_sigmoid(dist_func, f, g, min, max):
  return 2 / (1 + np.exp(dist_func(f,g,min,max)))

def s_exp(dist_func, f, g, min, max):
  return np.exp(-dist_func(f,g,min,max))

euclide_dists = []
manhattan_dists = []
for pdf in tqdm(ds):
  euclide_dists.append(euclide_distance(pdf, f_h, data.min(), data.max()))
  manhattan_dists.append(manhattan_distance(pdf, f_h, data.min(), data.max()))

euclide_dists = np.array(euclide_dists)
manhattan_dists = np.array(manhattan_dists)

euclide_similarities = [
    np.exp(-euclide_dists),
    2 / (1 + np.exp(euclide_dists)),
    np.min(euclide_dists) / euclide_dists
]

manhattan_similarities = [
    np.exp(-manhattan_dists),
    2 / (1 + np.exp(manhattan_dists)),
    np.min(manhattan_dists) / manhattan_dists
]

integrals_good = []
integrals_bad = []
for pdf in tqdm(ds):
  integrals_good.append(spint.quad(lambda x: pdf(x) * a_part_good(x), 0, 16)[0])
  integrals_bad.append(spint.quad(lambda x: pdf(x) * a_part_bad(x), 0, 16)[0])

# s_F vs s_exp vs s_sigmoid
_, axes = plt.subplots(1,3,figsize=(15,5))

axes[0].scatter(integrals_good[:400], euclide_similarities[2][:400])
axes[0].scatter(integrals_good[400:800], euclide_similarities[2][400:800])
axes[0].scatter(integrals_good[800:], euclide_similarities[2][800:])
axes[0].set_title('s_F')
axes[0].legend(['normal', 'exponential', 'uniform'])
#plt.show()

axes[1].scatter(integrals_good[:400], euclide_similarities[0][:400])
axes[1].scatter(integrals_good[400:800], euclide_similarities[0][400:800])
axes[1].scatter(integrals_good[800:], euclide_similarities[0][800:])
axes[1].set_title('s_exp')
axes[1].legend(['normal', 'exponential', 'uniform'])
#axes[0,1].show()

axes[2].scatter(integrals_good[:400], manhattan_similarities[1][:400])
axes[2].scatter(integrals_good[400:800], manhattan_similarities[1][400:800])
axes[2].scatter(integrals_good[800:], manhattan_similarities[1][800:])
axes[2].set_title('s_sigmoid')
axes[2].legend(['normal', 'exponential', 'uniform'])

plt.tight_layout()
plt.show()

# good vs bad + manhattan vs euclide
_, axes = plt.subplots(2,2,figsize=(10,10))

axes[0,0].scatter(integrals_good[:400], euclide_similarities[2][:400])
axes[0,0].scatter(integrals_good[400:800], euclide_similarities[2][400:800])
axes[0,0].scatter(integrals_good[800:], euclide_similarities[2][800:])
axes[0,0].set_title('"хорошая" A-часть,\nевклидово расстояние')
axes[0,0].legend(['normal', 'exponential', 'uniform'])

axes[0,1].scatter(integrals_bad[:400], euclide_similarities[2][:400])
axes[0,1].scatter(integrals_bad[400:800], euclide_similarities[2][400:800])
axes[0,1].scatter(integrals_bad[800:], euclide_similarities[2][800:])
axes[0,1].set_title('"посредственная" A-часть,\nевклидово расстояние')
axes[0,1].legend(['normal', 'exponential', 'uniform'])

axes[1,0].scatter(integrals_good[:400], manhattan_similarities[2][:400])
axes[1,0].scatter(integrals_good[400:800], manhattan_similarities[2][400:800])
axes[1,0].scatter(integrals_good[800:], manhattan_similarities[2][800:])
axes[1,0].set_title('"хорошая" A-часть,\nманхэттенское расстояние')
axes[1,0].legend(['normal', 'exponential', 'uniform'])

axes[1,1].scatter(integrals_bad[:400], manhattan_similarities[2][:400])
axes[1,1].scatter(integrals_bad[400:800], manhattan_similarities[2][400:800])
axes[1,1].scatter(integrals_bad[800:], manhattan_similarities[2][800:])
axes[1,1].set_title('"посредственная" A-часть,\nманхэттенское расстояние')
axes[1,1].legend(['normal', 'exponential', 'uniform'])

plt.tight_layout()
plt.show()