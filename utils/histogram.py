import numpy as np

def histogram_function(data):
    heights, bins = np.histogram(data, density=True)
    def h(x):
      if x < bins[0]:
        return 0
      if x >= bins[-1]:
        return 0
      for i in range(len(heights)):
        if bins[i] <= x < bins[i + 1]:
          return heights[i]
      return -1
    return h