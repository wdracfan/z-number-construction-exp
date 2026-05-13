import numpy as np
import matplotlib.pyplot as plt

from e2e.experiment import build_z_number
from utils.latex import table_to_latex

def experiment():
    np.random.seed(1243)
    table = []
    for size in [1000, 5000, 10000, 50000, 100000, 500000, 1000000]:
        data = np.random.normal(2, 3, size=size)

        a, b, t1, t2 = build_z_number(data, optimize='b', s_threshold=0.5, defuzzify='centroid', p=2, u_step='sturges', measure_time=True)

        table.append([f'{size}', f'{t1:.3f}', f'{t2:.3f}'])

    with open(f'../experiments/time_comparison/results.tex', 'w') as f:
        f.write(table_to_latex(table, []))