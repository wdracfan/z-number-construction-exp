import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spstats
import scipy.integrate as spint
from tqdm import tqdm

from utils.fuzzy import FS
from utils.histogram import histogram_function
from utils.logging import log_with_timestamp

def get_distributions(data: np.ndarray) -> dict[str, list]:
  std = np.std(data)
  min = np.min(data)
  max = np.max(data)

  min_std = 0.1 if std > 0.1 else 10 ** np.floor((np.log10(std)))

  normals = [spstats.norm(loc, scale) for loc in np.arange(min, max, (max - min) / 20) for scale in np.arange(min_std, 2 * std, std / 10)]
  expons = [spstats.expon(loc, scale) for loc in np.arange(min, max, (max - min) / 20) for scale in np.arange(min_std, 2 * std, std / 10)]
  uniforms = [spstats.uniform(loc, scale) 
            for loc in np.arange(min - (max - min) / 2, min + (max - min) / 2, (max - min) / 20) 
            for scale in np.arange((max - min) / 2, 3 * (max - min) / 2, (max - min) / 10)]
  return {
    'normal': normals,
    'exponential': expons,
    'uniform': uniforms
  }

# @log_with_timestamp
def calculate_integrals(a_part: FS, distributions: dict[str, list]):
  integrals = {}
  mf = a_part.membership_function()
  for key in distributions:
    integrals[key] = []
    for distribution in tqdm(distributions[key]):
      integrals[key].append(spint.quad(lambda x: distribution.pdf(x) * mf(x), a_part.a, a_part.d)[0])
    integrals[key] = np.array(integrals[key])
  return integrals

def convert_distances_into_similarities(distances: dict[str, list], p: float = 2) -> dict[str, dict[str, list]]:
  return {
    # 'exp': {key: (np.exp(-distances[key])) ** p for key in distances},
    # 'sigmoid': {key: (2 / (1 + np.exp(distances[key]))) ** p for key in distances},
    'F-based': {key: (np.min(distances[key]) / distances[key]) ** p for key in distances}
  }

#@log_with_timestamp
def calculate_distances(data: np.ndarray, distributions: dict[str, list], functions: list[str] = ['euclide']):
  histogram = histogram_function(data)

  def euclide_distance(f, g, l, r):
    return spint.quad(lambda x: (f(x) - g(x)) ** 2, l, r)[0]

  def chebyshev_distance(f, g, l, r):
    xs = np.arange(l, r, (r - l) / 1000)
    ys = [np.abs(f(x) - g(x)) for x in xs]
    return np.max(ys)

  def manhattan_distance(f, g, l, r):
    return spint.quad(lambda x: np.abs(f(x) - g(x)), l, r)[0]
  
  # Calculate distances
  euclide_dists = {}
  manhattan_dists = {}
  chebyshev_dists = {}
  for key in distributions:
    if 'euclide' in functions:
      euclide_dists[key] = []
    if 'manhattan' in functions:
      manhattan_dists[key] = []
    if 'chebyshev' in functions:
      chebyshev_dists[key] = []
    for distribution in tqdm(distributions[key]):
      if 'euclide' in functions:
        euclide_dists[key].append(euclide_distance(distribution.pdf, histogram, data.min(), data.max()))
      if 'manhattan' in functions:
        manhattan_dists[key].append(manhattan_distance(distribution.pdf, histogram, data.min(), data.max()))
      if 'chebyshev' in functions:
        chebyshev_dists[key].append(chebyshev_distance(distribution.pdf, histogram, data.min(), data.max()))
    if 'euclide' in functions: 
      euclide_dists[key] = np.array(euclide_dists[key])
    if 'manhattan' in functions:
      manhattan_dists[key] = np.array(manhattan_dists[key])
    if 'chebyshev' in functions:
      chebyshev_dists[key] = np.array(chebyshev_dists[key])

  return euclide_dists, manhattan_dists, chebyshev_dists

@log_with_timestamp
def poss_distr_triangularity(data: np.ndarray, description: str, prefix: str, a_part_good: FS, a_part_bad: FS, 
                             compare_similarities: bool, compare_p: bool):
  # Plot data
  plt.hist(data)
  plt.title(description)
  plt.savefig(f'../experiments/poss-distr-triangularity/{prefix}-histogram.png')
  plt.close()

  # Generate distributions
  distributions = get_distributions(data)

  # Get distances
  euclide_dists, manhattan_dists, chebyshev_dists = calculate_distances(data, distributions)

  # Get integrals (good and bad)
  integrals_good = calculate_integrals(a_part_good, distributions)
  integrals_bad = calculate_integrals(a_part_bad, distributions)

  p = 2

  # Convert distances into similarities
  euclide_similarities = convert_distances_into_similarities(euclide_dists, p)
  manhattan_similarities = convert_distances_into_similarities(manhattan_dists, p)
  chebyshev_similarities = convert_distances_into_similarities(chebyshev_dists, p)

  if compare_similarities:
    _, axes = plt.subplots(1, 3, figsize=(15,5))

    s_labels = ('F-based', 'exp', 'sigmoid')
    for i in range(len(s_labels)):
      for key in euclide_dists:
        axes[i].scatter(integrals_good[key], euclide_similarities[s_labels[i]][key])
      axes[i].set_title(s_labels[i])
      axes[i].legend(euclide_dists.keys())

    plt.tight_layout()
    plt.savefig(f'../experiments/poss-distr-triangularity/{prefix}-s-comparison.png')
    plt.close()

  if compare_p:
    _, axes = plt.subplots(3, 3, figsize=(15,15))

    distances = [
      (euclide_dists, 'евклидово'), 
      (manhattan_dists, 'манхэттенское'),
      (chebyshev_dists, 'чебышёвское')
    ]

    for i in range(len(distances)):
      distance, label = distances[i]
      for p in [1, 2, 3]:
        euclide_similarities_p = convert_distances_into_similarities(distance, p)
        for key in euclide_dists:
          axes[i, p-1].scatter(integrals_good[key], euclide_similarities_p['F-based'][key])
        axes[i, p-1].set_title(f'{label} расстояние,\np = {p}')
        axes[i, p-1].legend(euclide_dists.keys())

    plt.tight_layout()
    plt.savefig(f'../experiments/poss-distr-triangularity/{prefix}-p-comparison.png')
    plt.close()

  # Construct B-part
  distance_functions = [
    (euclide_similarities, 'евклидово'), 
    (manhattan_similarities, 'манхэттенское'),
    (chebyshev_similarities, 'чебышёвское')
  ]
  a_parts = [(integrals_good, 'хорошая'), (integrals_bad, 'посредственная')]
  _, axes = plt.subplots(3, 2, figsize=(5*2,5*3))

  for i in range(len(distance_functions)):
    similarities, distance_label = distance_functions[i]
    for j in range(len(a_parts)):
      integrals, a_part_label = a_parts[j]
      for key in integrals:
        axes[i, j].scatter(integrals[key],similarities['F-based'][key])
      axes[i, j].set_title(f'"{a_part_label}" A-часть,\n{distance_label} расстояние')
      axes[i, j].legend(euclide_dists.keys())
  
  plt.tight_layout()
  plt.savefig(f'../experiments/poss-distr-triangularity/{prefix}-b-parts.png')
  plt.close()

@log_with_timestamp
def experiment():
  print("------ NORMAL DATA ------")
  # Generate data
  np.random.seed(1243)
  data = np.hstack([np.random.normal(loc=4, scale=2, size=500), np.random.normal(loc=10, scale=2, size=500)])

  # A-part
  a_part_good = FS(0, 4, 11, 15)
  a_part_bad = FS(0, 2, 6, 10)

  poss_distr_triangularity(data, '1:1-смесь распределений N(4,2) и N(10,2)', 'normal', a_part_good, a_part_bad, compare_similarities=True, compare_p=True)

  print("------ MIXED DATA ------")
  # Generate data
  np.random.seed(1243)
  data = np.hstack([np.random.chisquare(df=5, size=500), np.random.uniform(low=10, high=15, size=500)])

  # A-part
  a_part_good = FS(0, 2, 14, 16)
  a_part_bad = FS(4, 8, 12, 15)

  poss_distr_triangularity(data, '1:1-смесь распределений chi-square(5) и Uniform[10;15]', 'mixed', a_part_good, a_part_bad, compare_similarities=False, compare_p=False)