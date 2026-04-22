import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint
import scipy.stats as spstats
from tqdm import tqdm

import pandas as pd

from fake_data_approximation.experiment import find_center_a1, find_center_a2, find_margins_b1, find_margins_b2, find_approx_c1
from poss_distr_triangularity.experiment import get_distributions, calculate_distances, convert_distances_into_similarities
from b_part_construction.experiment import cumulativeness
from utils.fuzzy import FS
from utils.latex import table_to_latex

from math import erf

def phi(z):
    return (1.0 + erf(z / np.sqrt(2.0))) / 2.0

def _normal_integral(a, b, c, d, mu, sigma):
  f = lambda x: 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

  def xint(x0, x1):
    return mu * (phi((x1 - mu) / sigma) - phi((x0 - mu) / sigma)) + sigma ** 2 * (f(x0) - f(x1))
  
  def noxint(x0, x1):
    return phi((x1 - mu) / sigma) - phi((x0 - mu) / sigma)
  
  result = 0
  
  if a != b:
    result += xint(a, b) / (b - a) - a / (b - a) * noxint(a, b)
  if b != c:
    result += noxint(b, c)
  if c != d:
    result += d / (d - c) * noxint(c, d) - xint(c, d) / (d - c)

  return result

def _expon_integral(a, b, c, d, mu, l):
  def xint(x0, x1):
    x0 = max(x0, mu)
    x1 = max(x1, mu)
    return (l * x0 + 1) / l * np.exp(-l * (x0 - mu)) - (l * x1 + 1) / l * np.exp(-l * (x1 - mu))
  
  def noxint(x0, x1):
    x0 = max(x0, mu)
    x1 = max(x1, mu)
    return np.exp(-l * (x0 - mu)) - np.exp(-l * (x1 - mu))

  result = 0
  if a != b:
    result += xint(a, b) / (b - a) - a / (b - a) * noxint(a, b)
  if b != c:
    result += noxint(b, c)
  if c != d:
    result += d / (d - c) * noxint(c, d) - xint(c, d) / (d - c)
  return result

def _uniform_integral(a, b, c, d, u, v):
  def xint(x0, x1):
    x0 = max(x0, u)
    x1 = min(x1, v)
    if x1 <= x0:
      return 0
    return (x1 ** 2 - x0 ** 2) / (2 * (v - u))
  
  def noxint(x0, x1):
    x0 = max(x0, u)
    x1 = min(x1, v)
    if x1 <= x0:
      return 0
    return (x1 - x0) / (v - u)

  result = 0
  if a != b:
    result += xint(a, b) / (b - a) - a / (b - a) * noxint(a, b)
  if b != c:
    result += noxint(b, c)
  if c != d:
    result += d / (d - c) * noxint(c, d) - xint(c, d) / (d - c)
  return result

def calculate_integrals(a_part: FS, distributions: dict[str, list]):
  integrals = {}
  mf = a_part.membership_function()
  for key in distributions:
    if key == 'normal':
      integrals[key] = [_normal_integral(a_part.a, a_part.b, a_part.c, a_part.d, distribution.stats()[0], np.sqrt(distribution.stats()[1])) for distribution in distributions[key]]
    elif key == 'exponential':
      integrals[key] = [_expon_integral(a_part.a, a_part.b, a_part.c, a_part.d, distribution.stats()[0] - np.sqrt(distribution.stats()[1]), 1 / np.sqrt(distribution.stats()[1])) for distribution in distributions[key]]
    elif key == 'uniform':
      integrals[key] = [_uniform_integral(a_part.a, a_part.b, a_part.c, a_part.d, (distribution.stats()[0] * 2 - np.sqrt(12 * distribution.stats()[1])) / 2, (distribution.stats()[0] * 2 + np.sqrt(12 * distribution.stats()[1])) / 2) for distribution in distributions[key]]
    else: # невозможная ветка
      integrals[key] = []
      for distribution in distributions[key]:
        integrals[key].append(spint.quad(lambda x: distribution.pdf(x) * mf(x), a_part.a, a_part.d)[0])
    integrals[key] = np.array(integrals[key])
  return integrals

def construct_b_part(data: np.ndarray, a_part: FS, distributions, euclide_dists, p: float = 2, scatter = False) -> tuple[int, int, int]:
    # Get integrals
    integrals = np.concatenate(list(calculate_integrals(a_part, distributions).values()))

    # Convert distances into similarities
    euclide_similarities = np.concatenate(list(convert_distances_into_similarities(euclide_dists, p)['F-based'].values()))

    xs = np.array(integrals)
    ys = np.array(euclide_similarities)

    if scatter:
      plt.scatter(xs, ys)
      plt.show()

    l, m, r = find_margins_b1(xs, ys, find_center_a2(xs, ys), thr=0.4)
    l = max(0, l)
    r = min(1, r)
    return FS(l, m, m, r)

def build_z_number(data: np.ndarray,
          u_min=None, u_max=None, u_step=None,
          optimize='specificity', beta=0.5, s_threshold=0.5, c_threshold=0.7,
          defuzzify='peak',
          p=2):
    if u_min == None:
      u_min = min(data)
    if u_max == None:
      u_max = max(data)
    if u_step == None:
      u_step = (u_max - u_min) / 10

    best_score = None
    best_subscore = None
    best_A = None
    best_B = None

    # Generate distributions
    distributions = get_distributions(data)
    # Get distances
    euclide_dists, _, _ = calculate_distances(data, distributions, ['euclide'])

    for a in tqdm(np.linspace(u_min, u_max, int((u_max - u_min) / u_step) + 1)):
      for b in np.linspace(a + u_step, u_max, int((u_max - a - u_step) / u_step) + 1):
        for c in np.linspace(b, u_max, int((u_max - b) / u_step) + 1):
          for d in np.linspace(c + u_step, u_max, int((u_max - c - u_step) / u_step) + 1):
            A = FS(a, b, c, d)
            specificity = A.specificity(u_max - u_min)

            if optimize == 'b' and specificity < s_threshold:
              continue
            B = construct_b_part(data, A, distributions, euclide_dists, p=p)
            print(specificity, A, B)
            if np.isnan(B.a) or np.isnan(B.b) or np.isnan(B.d) or np.isinf(B.a) or np.isinf(B.b) or np.isinf(B.d):
              continue
            b_defuzzified = (B.a + B.b + B.d) / 3 if defuzzify == 'centroid' else B.b

            if optimize == 'specificity' and b_defuzzified < c_threshold:
              continue

            if optimize == 'specificity':
              score = specificity
              subscore = b_defuzzified
            elif optimize == 'b':
              score = b_defuzzified
              subscore = specificity
            elif optimize == 'both':
              score = beta * b_defuzzified + (1 - beta) * specificity
              subscore = score

            if best_score == None or best_score < score or best_score == score and best_subscore < subscore:
              best_score = score
              best_subscore = subscore
              best_A = A
              best_B = B

    return best_A, best_B

def build_z_number_experiment(data, path, **kwargs):
    a, b = build_z_number(data, **kwargs)

    table = [
      ['Cum($\\mathbb X$)', f'{cumulativeness(data, data.min(), data.max(), 'sturges'):.3f}'],
      ['$A$', f'$\\FS({a.a:.3f}; {a.b:.3f}; {a.c:.3f}; {a.d:.3f})$'],
      ['$\\Sp(A)$', f'{a.specificity(data.max() - data.min()):.3f}'],
      ['$B$', f'$\\FS({b.a:.3f}; {b.b:.3f}; {b.d:.3f})$'],
      ['$B^*$', f'{(b.a + b.b + b.d) / 3:.3f}']
    ]

    with open(f'../experiments/e2e/{path}.tex', 'w') as f:
      f.write(table_to_latex(table, []))

    vals, bins = np.histogram(data)
    vals = vals / vals.max()
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].stairs(vals, bins, fill=True)
    axes[0].plot([data.min(), a.a, a.b, a.c, a.d, data.max()], [0, 0, 1, 1, 0, 0], color='red', lw=3)
    axes[0].set_title('A-часть Z-числа')
    axes[1].plot([0, b.a, b.b, b.d, 1], [0, 0, 1, 0, 0], color='red', lw=3)
    axes[1].set_title('B-часть Z-числа')
    plt.tight_layout()
    plt.savefig(f'../experiments/e2e/{path}.png')
    plt.close()

def experiment():
    df = pd.read_csv('datasets/globalAirQuality.csv')
    data = np.array(df[df.city == 'Paris']['no2'])
    
    plt.hist(data)
    plt.title('Концентрация диоксида азота в воздухе Парижа')
    plt.ylabel('мкг/куб.м.')
    plt.savefig('../experiments/e2e/no2/histograms.png')
    plt.close()

    build_z_number_experiment(data, 'no2/optimize_b', optimize='b', s_threshold=0.5, defuzzify='centroid', p=2)
    build_z_number_experiment(data, 'no2/optimize_both', optimize='both', beta=0.5, defuzzify='centroid', p=2)
  
    df = pd.read_csv('datasets/mosquito_Indicator.csv')
    data = np.array(df[(df.date >= '2017-05-01') & (df.date <= '2017-09-30')].mosquito_Indicator)

    plt.hist(data)
    plt.title('Количество комаров на фиксированной территории в Сеуле')
    plt.ylabel('шт.')
    plt.savefig('../experiments/e2e/mosquito/histogram.png')
    plt.close()

    build_z_number_experiment(data, 'mosquito/optimize_spec', optimize='specificity', c_threshold=0.6, defuzzify='centroid', p=3)
    build_z_number_experiment(data, 'mosquito/optimize_both', optimize='both', beta=0.5, defuzzify='centroid', p=3)
    