import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spstats
import scipy.integrate as spint
from tqdm import tqdm

np.random.seed(1243)
data = np.hstack([np.random.normal(loc=4, scale=2, size=500), np.random.normal(loc=10, scale=2, size=500)])

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

euclide_dists = []
for pdf in tqdm(ds):
  euclide_dists.append(euclide_distance(pdf, f_h, data.min(), data.max()))

euclide_dists = np.array(euclide_dists)

euclide_similarities = [
    None, #np.exp(-euclide_dists),
    None, #2 / (1 + np.exp(euclide_dists)),
    np.min(euclide_dists) / euclide_dists
]

integrals_good = []
for pdf in tqdm(ds):
  integrals_good.append(spint.quad(lambda x: pdf(x) * a_part_good(x), 0, 16)[0])

plt.scatter(integrals_good, euclide_similarities[2])

def plot_approx(axis, title, l, m, r, xs, ys):
  axis.scatter(xs, ys)
  axis.plot([m, l], [1, 0], color='red', lw=3)
  axis.plot([m, r], [1, 0], color='red', lw=3)
  axis.set_title(title)

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
  l = np.mean((xs_less - c * ys_less) * (xs_less - c)) / np.mean((1 - ys_less) * (xs_less - c))
  r = np.mean((xs_greater - c * ys_greater) * (xs_greater - c)) / np.mean((1 - ys_greater) * (xs_greater - c))
  return l, c, r

def find_margins_d2(xs, ys, c, thr=0):
  xs_less = xs[(xs < c) & (ys > thr)]
  ys_less = ys[(xs < c) & (ys > thr)]
  xs_greater = xs[(xs > c) & (ys > thr)]
  ys_greater = ys[(xs > c) & (ys > thr)]
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

def find_approx_e1(xs, ys, thr=0):
  min_error = 1e9
  min_args = (0,0,0)
  for c in xs[1:-1]:
    xs_less = xs[(xs < c) & (ys > thr)]
    ys_less = ys[(xs < c) & (ys > thr)]
    xs_greater = xs[(xs > c) & (ys > thr)]
    ys_greater = ys[(xs > c) & (ys > thr)]
    if (len(xs_less) * len(xs_greater) == 0):
      continue
    l = np.mean((xs_less - c * ys_less) / (xs_less - c)) / np.mean((1 - ys_less) / (xs_less - c))
    r = np.mean((xs_greater - c * ys_greater) / (xs_greater - c)) / np.mean((1 - ys_greater) / (xs_greater - c))
    error = np.sum(((xs_less - l) / (c - l) - ys_less) ** 2 / (c - xs_less) ** 2) + np.sum(((r - xs_greater) / (r - c) - ys_greater) ** 2 / (c - xs_greater) ** 2) 
    if error < min_error:
      min_error = error
      min_args = (l,c,r)
  return min_args

xs = np.array(integrals_good)
ys = np.array(euclide_similarities[2])

_, axes = plt.subplots(5,4, figsize=(20,25))

plot_approx(axes[0,0], 'A1 + B1, порог t = 0', *find_margins_b1(xs, ys, find_center_a1(xs, ys), 0.0), xs, ys)
plot_approx(axes[0,1], 'A1 + B1, порог t = 0.4', *find_margins_b1(xs, ys, find_center_a1(xs, ys), 0.4), xs, ys)
plot_approx(axes[0,2], 'A2 + B1, порог t = 0', *find_margins_b1(xs, ys, find_center_a2(xs, ys), 0), xs, ys)
plot_approx(axes[0,3], 'A2 + B1, порог t = 0.4', *find_margins_b1(xs, ys, find_center_a2(xs, ys), 0.4), xs, ys)

plot_approx(axes[1,0], 'A1 + B2, порог t = 0', *find_margins_b2(xs, ys, find_center_a1(xs, ys), 0.0), xs, ys)
plot_approx(axes[1,1], 'A1 + B2, порог t = 0.4', *find_margins_b2(xs, ys, find_center_a1(xs, ys), 0.4), xs, ys)
plot_approx(axes[1,2], 'A2 + B2, порог t = 0', *find_margins_b2(xs, ys, find_center_a2(xs, ys), 0), xs, ys)
plot_approx(axes[1,3], 'A2 + B2, порог t = 0.4', *find_margins_b2(xs, ys, find_center_a2(xs, ys), 0.4), xs, ys)

plot_approx(axes[2,0], 'A1 + D1, порог t = 0', *find_margins_d1(xs, ys, find_center_a1(xs, ys), 0.0), xs, ys)
plot_approx(axes[2,1], 'A1 + D1, порог t = 0.4', *find_margins_d1(xs, ys, find_center_a1(xs, ys), 0.4), xs, ys)
plot_approx(axes[2,2], 'A2 + D1, порог t = 0', *find_margins_d1(xs, ys, find_center_a2(xs, ys), 0), xs, ys)
plot_approx(axes[2,3], 'A2 + D1, порог t = 0.4', *find_margins_d1(xs, ys, find_center_a2(xs, ys), 0.4), xs, ys)

plot_approx(axes[3,0], 'A1 + D2, порог t = 0', *find_margins_d2(xs, ys, find_center_a1(xs, ys), 0.0), xs, ys)
plot_approx(axes[3,1], 'A1 + D2, порог t = 0.4', *find_margins_d2(xs, ys, find_center_a1(xs, ys), 0.4), xs, ys)
plot_approx(axes[3,2], 'A2 + D2, порог t = 0', *find_margins_d2(xs, ys, find_center_a2(xs, ys), 0), xs, ys)
plot_approx(axes[3,3], 'A2 + D2, порог t = 0.4', *find_margins_d2(xs, ys, find_center_a2(xs, ys), 0.4), xs, ys)

plot_approx(axes[4,0], 'C1, порог t = 0', *find_approx_c1(xs, ys, 0.0), xs, ys)
plot_approx(axes[4,1], 'C2, порог t = 0.4', *find_approx_c1(xs, ys, 0.4), xs, ys)

plot_approx(axes[4,2], 'E1, порог t = 0', *find_approx_e1(xs, ys, 0.0), xs, ys)
plot_approx(axes[4,3], 'E2, порог t = 0.4', *find_approx_e1(xs, ys, 0.4), xs, ys)
