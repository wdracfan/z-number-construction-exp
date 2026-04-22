import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spstats
import scipy.integrate as spint

from utils.latex import table_to_latex
from utils.fuzzy import FS

# TODO: графики сгенерированных треугольников и мб про шаг в a2
# TODO: параметр t=0.4 оказывает хорошее влияние только на методы b1 и d1 (и c1, но мб случайность)
# TODO: пофиксить минусы в отображении таблицы

# Generates fake triangular data with or without noising of zero-values
def generate_triangle(l, m, r, size=1000, noise_zero_values=False):
  xm = m
  xl = l
  xr = r

  np.random.seed(1243)

  if not noise_zero_values:
    xs = np.random.uniform(size=size) # x-coordinates
    eps = np.random.normal(loc=-0.3, scale=0.25, size=size) # noise
    ys = np.min([(xs - xl) / (xm - xl), (xr - xs) / (xr - xm)], axis=0) + eps
  else:
    xs_triangle = np.random.uniform(low=l, high=r, size=size//5) # x-coordinates
    eps_triangle = np.random.normal(loc=-0.3, scale=0.25, size=size//5) # noise
    ys_triangle = np.min([(xs_triangle - xl) / (xm - xl), (xr - xs_triangle) / (xr - xm)], axis=0) + eps_triangle
    
    xs_middle = np.random.uniform(low=m/2, high=(m+1)/2, size=size//5)
    ys_middle = np.concatenate([
      np.random.normal(loc=0.2, scale=0.1, size=size//10),
      np.random.uniform(low=0, high=0.4, size=size//10)])

    xs_overall = np.random.uniform(low=0, high=1, size=3*size//5)
    ys_overall = np.concatenate([
      np.random.normal(loc=0.05, scale=0.1, size=3*size//10),
      np.random.uniform(low=0, high=0.2, size=3*size//10)])

    xs = np.concatenate([xs_triangle, xs_middle, xs_overall])
    ys = np.concatenate([ys_triangle, ys_middle, ys_overall])
  
  ys[(ys > 1) | (ys < 0)] = 0

  plot_triangular_approximation(l, m, r, xs, ys, 'with' if noise_zero_values else 'no')

  return xs, ys

def plot_triangular_approximation(l, m, r, xs, ys, noise_label: str):
  plt.scatter(xs, ys)
  plt.plot([m, l], [1, 0], color='red')
  plt.plot([m, r], [1, 0], color='red')
  plt.savefig(f'../experiments/fake-data-approximation/{noise_label}-noise.png')
  plt.close()

# Methods described in Section 5

def find_center_a1(xs, ys):
  return xs[np.argsort(ys)[-1]]

def find_center_a2(xs, ys, d=0.1):
  max_mean = 0
  max_center = 0
  for left in np.arange(0, 1 - 2*d, 0.01):
    right = left + 2*d
    mean = np.mean(ys[(xs >= left) & (xs <= right)])
    if mean > max_mean:
      max_mean = mean
      max_center = left + d
  return max_center

def find_margins_b1(xs, ys, c, thr=0):
  xs_less = xs[(xs < c) & (ys > thr)]
  ys_less = ys[(xs < c) & (ys > thr)]
  xs_greater = xs[(xs > c) & (ys > thr)]
  ys_greater = ys[(xs > c) & (ys > thr)]
  xm_less = np.mean(xs_less)
  ym_less = np.mean(ys_less)
  xm_greater = np.mean(xs_greater)
  ym_greater = np.mean(ys_greater)
  l = c - (1 / (1 - ym_less) * (c - xm_less))
  r = c + (1 / (1 - ym_greater) * (xm_greater - c))
  return l, c, r

def find_margins_d1(xs, ys, c, thr=0):
  xs_less = xs[(xs < c) & (ys > thr)]
  ys_less = ys[(xs < c) & (ys > thr)]
  xs_greater = xs[(xs > c) & (ys > thr)]
  ys_greater = ys[(xs > c) & (ys > thr)]
  xm_less = np.median(xs_less)
  ym_less = np.median(ys_less)
  xm_greater = np.median(xs_greater)
  ym_greater = np.median(ys_greater)
  l = c - (1 / (1 - ym_less) * (c - xm_less))
  r = c + (1 / (1 - ym_greater) * (xm_greater - c))
  return l, c, r

def find_margins_b2(xs, ys, c, thr=0):
  xs_less = xs[(xs < c) & (ys > thr)]
  ys_less = ys[(xs < c) & (ys > thr)]
  xs_greater = xs[(xs > c) & (ys > thr)]
  ys_greater = ys[(xs > c) & (ys > thr)]
  # plt.scatter(xs_greater, ys_greater)
  l = np.mean((xs_less - c * ys_less) * (xs_less - c)) / np.mean((1 - ys_less) * (xs_less - c))
  r = np.mean((xs_greater - c * ys_greater) * (xs_greater - c)) / np.mean((1 - ys_greater) * (xs_greater - c))
  return l, c, r

def find_margins_d2(xs, ys, c, thr=0):
  xs_less = xs[(xs < c) & (ys > thr)]
  ys_less = ys[(xs < c) & (ys > thr)]
  xs_greater = xs[(xs > c) & (ys > thr)]
  ys_greater = ys[(xs > c) & (ys > thr)]
  # plt.scatter(xs_greater, ys_greater)
  l = np.mean((xs_less - c * ys_less) / (xs_less - c)) / np.mean((1 - ys_less) / (xs_less - c))
  r = np.mean((xs_greater - c * ys_greater) / (xs_greater - c)) / np.mean((1 - ys_greater) / (xs_greater - c))
  return l, c, r

def find_approx_c1(xs, ys, thr=0):
  min_error = 1e9
  min_args = (0,0,0)
  for c in xs[1:-1]:
    xs_less = xs[(xs < c) & (ys > thr)]
    ys_less = ys[(xs < c) & (ys > thr)]
    xs_greater = xs[(xs > c) & (ys > thr)]
    ys_greater = ys[(xs > c) & (ys > thr)]
    if (len(xs_less) * len(xs_greater) == 0):
      continue
    l = np.mean((xs_less - c * ys_less) * (xs_less - c)) / np.mean((1 - ys_less) * (xs_less - c))
    r = np.mean((xs_greater - c * ys_greater) * (xs_greater - c)) / np.mean((1 - ys_greater) * (xs_greater - c))
    error = np.sum(((xs_less - l) / (c - l) - ys_less) ** 2) + np.sum(((r - xs_greater) / (r - c) - ys_greater) ** 2) 
    if error < min_error:
      min_error = error
      min_args = (l,c,r)
  return min_args

def find_approx_e1(xs, ys, p, thr=0):
  min_error = 1e9
  min_args = (0,0,0)
  for c in xs[1:-1]:
    xs_less = xs[(xs < c) & (ys > thr)]
    ys_less = ys[(xs < c) & (ys > thr)]
    xs_greater = xs[(xs > c) & (ys > thr)]
    ys_greater = ys[(xs > c) & (ys > thr)]
    if (len(xs_less) * len(xs_greater) == 0):
      continue
    l = np.mean((xs_less - c * ys_less) / (c - xs_less) ** (p - 1)) / np.mean((1 - ys_less) / (c - xs_less) ** (p - 1))
    r = np.mean((xs_greater - c * ys_greater) / (xs_greater - c) ** (p - 1)) / np.mean((1 - ys_greater) / (xs_greater - c) ** (p - 1))
    error = np.sum(((xs_less - l) / (c - l) - ys_less) ** 2 / (c - xs_less) ** p) + np.sum(((r - xs_greater) / (r - c) - ys_greater) ** 2 / (xs_greater - c) ** p)
    if error < min_error:
      min_error = error
      min_args = (l,c,r)
  return min_args

def evaluate(l,c,r, ll,cc,rr, method_label: str, t: str):
  mae = np.abs(l-ll) + np.abs(r-rr) + np.abs(c-cc)
  manhattan = np.max([np.abs(FS(l,c,c,r).membership_function()(x) - FS(ll,cc,cc,rr).membership_function()(x)) for x in np.arange(min(l,ll), max(r,rr), 0.001)])
  euclide = spint.quad(lambda x: (FS(l,c,c,r).membership_function()(x) - FS(ll,cc,cc,rr).membership_function()(x)) ** 2, min(l,ll), max(r,rr))[0]
  return [method_label, t, f'{l:.3f}', f'{c:.3f}', f'{r:.3f}', f'{euclide:.3f}', f'{manhattan:.3f}', f'{mae:.3f}']

def experiment():
    triangle_args = (0.43, 0.72, 0.89)

    # No zero-values noise
    xs, ys = generate_triangle(*triangle_args, noise_zero_values=False)

    centers = {
      'A1': find_center_a1,
      'A2': find_center_a2
    }

    margins = {
      'B1': find_margins_b1, 
      'B2': find_margins_b2,
      'D1': find_margins_d1,
      'D2': find_margins_d2
    }

    whole = {
      'C1': find_approx_c1,
      'E1': find_approx_e1
    }

    result_table = []
    for center in centers.keys():
      for margin in list(margins.keys())[:2]:
        for t in ['0', '0.4']:
          result_table.append(evaluate(*margins[margin](xs, ys, centers[center](xs, ys), float(t)), *triangle_args, f'{center} + {margin}', t))
    for method in list(whole.keys())[:1]:
      for t in ['0', '0.4']:
        result_table.append(evaluate(*whole[method](xs, ys, float(t)), *triangle_args, method, t))
    
    with open('../experiments/fake-data-approximation/no-zero-value-noise.tex', 'w') as f:
        f.write(table_to_latex(result_table, bold_max_columns=[5, 6, 7], max_or_min = 'min'))
    
    # With zero-values noise
    xs, ys = generate_triangle(*triangle_args, noise_zero_values=True)

    centers = {
      'A1': find_center_a1,
      'A2': find_center_a2
    }

    margins = {
      'B1': find_margins_b1, 
      'B2': find_margins_b2,
      'D1': find_margins_d1,
      'D2': find_margins_d2
    }

    whole = {
      'C1': find_approx_c1,
      'E1': find_approx_e1
    }

    result_table = []
    for center in centers.keys():
      for margin in list(margins.keys())[:2]:
        for t in ['0', '0.4']:
          result_table.append(evaluate(*margins[margin](xs, ys, centers[center](xs, ys), float(t)), *triangle_args, f'{center} + {margin}', t))
    for method in list(whole.keys())[:1]:
      for t in ['0', '0.2']:
        result_table.append(evaluate(*whole[method](xs, ys, float(t)), *triangle_args, method, t))
    
    with open('../experiments/fake-data-approximation/zero-value-noise.tex', 'w') as f:
        f.write(table_to_latex(result_table, bold_max_columns=[5, 6, 7], max_or_min = 'min'))

    result_table = []
    for center in centers.keys():
      for margin in list(margins.keys())[2:]:
        for t in ['0', '0.4']:
          result_table.append(evaluate(*margins[margin](xs, ys, centers[center](xs, ys), float(t)), *triangle_args, f'{center} + {margin}', t))
    for method in list(whole.keys())[1:]:
      for t in ['0', '0.2']:
        result_table.append(evaluate(*whole[method](xs, ys, 1, float(t)), *triangle_args, method, t))
    
    with open('../experiments/fake-data-approximation/zero-value-noise-modified.tex', 'w') as f:
        f.write(table_to_latex(result_table, bold_max_columns=[5, 6, 7], max_or_min = 'min'))