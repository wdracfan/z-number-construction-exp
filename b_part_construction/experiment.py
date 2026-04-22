import numpy as np
import matplotlib.pyplot as plt

from fake_data_approximation.experiment import find_center_a1, find_center_a2, find_margins_b1, find_margins_b2, find_approx_c1
from poss_distr_triangularity.experiment import get_distributions, calculate_distances, calculate_integrals, convert_distances_into_similarities
from utils.fuzzy import FS
from utils.latex import table_to_latex

def construct_b_part(data: np.ndarray, a_part: FS, p: float = 2) -> tuple[int, int, int]:
    # Generate distributions
    distributions = get_distributions(data)

    # Get distances
    euclide_dists, _, _ = calculate_distances(data, distributions, ['euclide'])

    # Get integrals
    integrals = np.concatenate(list(calculate_integrals(a_part, distributions).values()))

    # Convert distances into similarities
    euclide_similarities = np.concatenate(list(convert_distances_into_similarities(euclide_dists, p)['F-based'].values()))

    xs = np.array(integrals)
    ys = np.array(euclide_similarities)

    l, m, r = find_margins_b1(xs, ys, find_center_a2(xs, ys), thr=0.4)
    return l, m, r

def cumulativeness(data: np.ndarray, l: float, r: float, bins: int) -> float:
    hist, bins = np.histogram(data, bins=bins, range=(l, r))
    interval = bins[1] - bins[0]
    return hist.sum() * interval / ((r - l) * hist.max())
    
def experiment_data_spec():
    np.random.seed(1243)
    bins = 15
    a_part = FS(-7, -6, 6, 7)
    size = 1000

    datas = [
        np.hstack([np.random.uniform(low=-6, high=6, size=size)]),

        np.hstack([np.random.normal(loc=0, scale=3, size=size)]),

        np.hstack([
            np.random.normal(loc=-4, scale=1, size=size//3),
            np.random.normal(loc=-0, scale=1, size=size//3),
            np.random.normal(loc=4, scale=1, size=size//3)
        ]),

        np.hstack([np.random.normal(loc=-3, scale=1, size=size//2), np.random.normal(loc=3, scale=1, size=size//2)])
    ]

    _, axes = plt.subplots(2, 2, figsize=(10,10))
    axes[0][0].hist(datas[0], bins=bins)
    axes[0][0].set_title('Гистограмма выборки 1')
    axes[0][1].hist(datas[1], bins=bins)
    axes[0][1].set_title('Гистограмма выборки 2')
    axes[1][0].hist(datas[2], bins=bins)
    axes[1][0].set_title('Гистограмма выборки 3')
    axes[1][1].hist(datas[3], bins=bins)
    axes[1][1].set_title('Гистограмма выборки 4')
    plt.tight_layout()
    plt.savefig(f'../experiments/b-part-construction/histograms.png')
    plt.close()

    results = [['1'], ['2'], ['3'], ['4']]
    for i in range(len(datas)):
        l, c, r = construct_b_part(datas[i], a_part, p=2)
        spec = cumulativeness(datas[i], -6, 6, bins)
        results[i].append(f'{spec:.3f}')
        results[i].append(f'{r-l:.3f}')

    with open('../experiments/b-part-construction/spec-of-datasets.tex', 'w') as f:
        f.write(table_to_latex(results, []))

def experiment_a_spec():
    np.random.seed(1243)
    data = np.hstack([np.random.chisquare(df=5, size=500), np.random.uniform(low=10, high=15, size=500)])

    # A-части по возрастанию специфичности
    a_parts = [
        FS(-2, 0, 18, 20),
        FS(0, 2, 16, 18),
        FS(0, 5, 13, 18),
        FS(5, 8, 12, 16),
        FS(10, 11, 14, 16)
    ]

    results = [[fs.to_latex_string(), f'{fs.specificity(22):.3f}'] for fs in a_parts]
    
    for i in range(len(a_parts)):
        a_part = a_parts[i]
        l, c, r = construct_b_part(data, a_part, p=2)
        results[i].append(f'{r-l:.3f}')

    with open('../experiments/b-part-construction/spec-of-a-parts.tex', 'w') as f:
        f.write(table_to_latex(results, []))
        