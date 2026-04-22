import numpy as np
import matplotlib.pyplot as plt

from fake_data_approximation.experiment import find_center_a1, find_center_a2, find_margins_b1, find_margins_b2, find_approx_c1
from poss_distr_triangularity.experiment import get_distributions, calculate_distances, calculate_integrals, convert_distances_into_similarities
from utils.fuzzy import FS

def plot_triangular_approximation(axis, title, l, m, r, xs, ys):
  axis.scatter(xs, ys)
  axis.plot([m, l], [1, 0], color='red', lw=3)
  axis.plot([m, r], [1, 0], color='red', lw=3)
  axis.set_title(title)
  return l, m, r

def plot_different_approximations(data: np.ndarray, a_part: FS, p: float):
  # Generate distributions
  distributions = get_distributions(data)

  # Get distances
  euclide_dists, manhattan_dists, chebyshev_dists = calculate_distances(data, distributions)

  # Get integrals
  integrals = np.concatenate(list(calculate_integrals(a_part, distributions).values()))

  # Convert distances into similarities
  euclide_similarities = np.concatenate(list(convert_distances_into_similarities(euclide_dists, p)['F-based'].values()))
  #manhattan_similarities = np.concatenate(list(convert_distances_into_similarities(manhattan_dists, p)['F-based'].values()))
  #chebyshev_similarities = np.concatenate(list(convert_distances_into_similarities(chebyshev_dists, p)['F-based'].values()))

  xs = np.array(integrals)
  ys = np.array(euclide_similarities)

  centers = {
    'A1': find_center_a1,
    'A2': find_center_a2
  }

  margins = {
    'B1': find_margins_b1, 
    'B2': find_margins_b2
  }

  _, axes = plt.subplots(len(centers) * len(margins) + 1, 2, figsize=(2 * 5, (len(centers) * len(margins) + 1) * 5))

  for i in range(len(centers)):
    for j in range(len(margins)):
      center = centers[list(centers.keys())[i]]
      margin = margins[list(margins.keys())[j]]
      l, m, r = plot_triangular_approximation(
        axes[i * len(centers) + j, 0], 
        f'{list(centers.keys())[i]} + {list(margins.keys())[j]}, порог t = 0',
        *margin(xs, ys, center(xs, ys), thr=0),
        xs, ys)
      print(f'{list(centers.keys())[i]} + {list(margins.keys())[j]}, порог t = 0', r-l)
      l, m, r = plot_triangular_approximation(
        axes[i * len(centers) + j, 1], 
        f'{list(centers.keys())[i]} + {list(margins.keys())[j]}, порог t = 0.4',
        *margin(xs, ys, center(xs, ys), thr=0.4),
        xs, ys)
      print(f'{list(centers.keys())[i]} + {list(margins.keys())[j]}, порог t = 0.4', r-l)
  l, m, r = plot_triangular_approximation(
    axes[-1, 0], 
    f'C1, порог t = 0',
    *find_approx_c1(xs, ys, thr=0),
    xs, ys)
  print(f'C1, порог t = 0', r-l)
  l, m, r = plot_triangular_approximation(
    axes[-1, 1], 
    f'C1, порог t = 0.4',
    *find_approx_c1(xs, ys, thr=0.4),
    xs, ys)
  print(f'C1, порог t = 0', r-l)

  # plt.title('Результаты построения треугольной аппроксимации B-части различными способами')
  plt.tight_layout()
  plt.savefig(f'../experiments/real-data-approximation/graph.png')
  plt.close()
      
def experiment():
  np.random.seed(1243)
  data = np.hstack([np.random.chisquare(df=5, size=500), np.random.uniform(low=10, high=15, size=500)])
  a_part = FS(0, 2, 14, 16)
  #data = np.hstack([np.random.normal(loc=4, scale=2, size=500), np.random.normal(loc=10, scale=2, size=500)])
  #a_part = FS(0, 2, 6, 10)
  for p in [2]:
    plot_different_approximations(data, a_part, p)