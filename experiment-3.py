import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spstats
import scipy.integrate as spint

from time import time

def generate_triangle(l, m, r, size=1000):
  xm = m
  xl = l
  xr = r

  np.random.seed(1243)
  xs = np.random.uniform(size=1000)
  eps = np.random.normal(scale=0.25, size=1000)
  # ДОБАВЛЯТЬ ШУМ В НУЛИ:
  ys = np.max([np.zeros(1000), np.min([(xs - xl) / (xm - xl), (xr - xs) / (xr - xm)], axis=0)], axis=0) + eps
  # НЕ ДОБАВЛЯТЬ ШУМ В НУЛИ:
  #ys = np.min([(xs - xl) / (xm - xl), (xr - xs) / (xr - xm)], axis=0) + eps
  ys[(ys > 1) | (ys < 0)] = 0

  return xs, ys

xs, ys = generate_triangle(0.4, 0.5, 0.7)

plt.scatter(xs, ys)

def plot_approx(l, m, r, xs, ys):
  plt.scatter(xs, ys)
  plt.plot([m, l], [1, 0], color='red')
  plt.plot([m, r], [1, 0], color='red')
  plt.show()

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
  plt.scatter(xs_greater, ys_greater)
  l = np.mean((xs_less - c * ys_less) * (xs_less - c)) / np.mean((1 - ys_less) * (xs_less - c))
  r = np.mean((xs_greater - c * ys_greater) * (xs_greater - c)) / np.mean((1 - ys_greater) * (xs_greater - c))
  return l, c, r

def find_margins_d2(xs, ys, c, thr=0):
  xs_less = xs[(xs < c) & (ys > thr)]
  ys_less = ys[(xs < c) & (ys > thr)]
  xs_greater = xs[(xs > c) & (ys > thr)]
  ys_greater = ys[(xs > c) & (ys > thr)]
  plt.scatter(xs_greater, ys_greater)
  l = np.mean((xs_less - c * ys_less) / (xs_less - c)) / np.mean((1 - ys_less) / (xs_less - c))
  r = np.mean((xs_greater - c * ys_greater) / (xs_greater - c)) / np.mean((1 - ys_greater) / (xs_greater - c))
  return l, c, r

def find_approx(xs, ys, thr=0):
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

ds = np.arange(0.001, 0.5, 0.001)
cs = [find_center_a2(xs, ys, d) for d in ds]
plt.xlabel('Размер "окна" d')
plt.ylabel('Центр c\' аппроксимирующего нечёткого числа')
plt.title('Зависимость c\' от d')
plt.axhline(0.5, color='red', label='"Истинное значение" c')
plt.legend()
plt.plot(ds, cs)

def triangular(l,c,r):
  def f(x):
    if x < l:
      return 0
    if x > r:
      return 0
    if l <= x <= c:
      return (x - l) / (c - l)
    if c <= x <= r:
      return (r - x) / (r - c)
  return f

def evaluate(l,c,r, ll,cc,rr):
  mae = np.abs(l-ll) + np.abs(r-rr) + np.abs(c-cc)
  manhattan = np.max([np.abs(triangular(l,c,r)(x) - triangular(ll,cc,rr)(x)) for x in np.arange(min(l,ll), max(r,rr), 0.001)])
  euclide = spint.quad(lambda x: (triangular(l,c,r)(x) - triangular(ll,cc,rr)(x)) ** 2, min(l,ll), max(r,rr))[0]
  print(f"{l:.3f} & {c:.3f} & {r:.3f} & {euclide:.3f} & {manhattan:.3f} & {mae:.3f}")

evaluate(*find_margins_d1(xs, ys, find_center_a1(xs, ys), 0.0), 0.4, 0.5, 0.7)
evaluate(*find_margins_d1(xs, ys, find_center_a1(xs, ys), 0.4), 0.4, 0.5, 0.7)
#evaluate(*find_margins_b1(xs, ys, find_center_a2(xs, ys), 0.0), 0.4, 0.5, 0.7)
#evaluate(*find_margins_b1(xs, ys, find_center_a2(xs, ys), 0.4), 0.4, 0.5, 0.7)

evaluate(*find_margins_d2(xs, ys, find_center_a1(xs, ys), 0.0), 0.4, 0.5, 0.7)
evaluate(*find_margins_d2(xs, ys, find_center_a1(xs, ys), 0.4), 0.4, 0.5, 0.7)
#evaluate(*find_margins_b2(xs, ys, find_center_a2(xs, ys), 0.0), 0.4, 0.5, 0.7)
#evaluate(*find_margins_b2(xs, ys, find_center_a2(xs, ys), 0.4), 0.4, 0.5, 0.7)

evaluate(*find_approx(xs, ys, 0.0), 0.4, 0.5, 0.7)

evaluate(*find_margins_d1(xs, ys, find_center_a2(xs, ys), 0.0), 0.4, 0.5, 0.7)
evaluate(*find_margins_d1(xs, ys, find_center_a2(xs, ys), 0.4), 0.4, 0.5, 0.7)
evaluate(*find_margins_d2(xs, ys, find_center_a2(xs, ys), 0.0), 0.4, 0.5, 0.7)
evaluate(*find_margins_d2(xs, ys, find_center_a2(xs, ys), 0.4), 0.4, 0.5, 0.7)