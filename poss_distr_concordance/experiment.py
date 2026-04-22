import matplotlib.pyplot as plt
import scipy.stats as spstats
import numpy as np

from poss_distr_triangularity.experiment import get_distributions, calculate_distances, calculate_integrals, convert_distances_into_similarities

from utils.logging import log_with_timestamp
from utils.fuzzy import FS

def convert_to_latex_table(concordance_results, f_true_label: str) -> str:
    return f'''
{f_true_label} & $d_{{eucl}}$ & {concordance_results['kendall']['euclide'].statistic:.3f} & {concordance_results['kendall']['euclide'].pvalue:3f} & {concordance_results['spearman']['euclide'].statistic:.3f} & {concordance_results['spearman']['euclide'].pvalue:3f} \\\\ 
{f_true_label} & $d_{{manh}}$ & {concordance_results['kendall']['manhattan'].statistic:.3f} & {concordance_results['kendall']['manhattan'].pvalue:3f} & {concordance_results['spearman']['manhattan'].statistic:.3f} & {concordance_results['spearman']['manhattan'].pvalue:3f} \\\\ 
{f_true_label} & $d_{{cheb}}$ & {concordance_results['kendall']['chebyshev'].statistic:.3f} & {concordance_results['kendall']['chebyshev'].pvalue:3f} & {concordance_results['spearman']['chebyshev'].statistic:.3f} & {concordance_results['spearman']['chebyshev'].pvalue:3f} \\\\ \hline
'''

@log_with_timestamp
def check_concordance(data: np.ndarray, draw_stats: bool = False, a_part: FS = None):
    distributions = get_distributions(data)
    
    if draw_stats:
        integrals =  np.concatenate(list(calculate_integrals(a_part, distributions).values()))

    # For each distribution, apply Kolmogorov test to data and get statistic values
    distributions = np.concatenate(list(distributions.values()))
    kstest_values = [-spstats.kstest(data, distribution.cdf).statistic for distribution in distributions]

    if draw_stats:
        plt.scatter(integrals, (1 + np.array(kstest_values)))
        plt.ylabel('Значение статистики Колмогорова')
        plt.savefig('../experiments/poss-distr-concordance/kolmogorov-stats.png')
        plt.close()
    
    euclide_distances, manhattan_distances, chebyshev_distances = calculate_distances(data, distributions)
    euclide_similarities = np.concatenate(list(convert_distances_into_similarities(euclide_distances)['F-based'].values()))
    manhattan_similarities = np.concatenate(list(convert_distances_into_similarities(manhattan_distances)['F-based'].values()))
    chebyshev_similarities = np.concatenate(list(convert_distances_into_similarities(chebyshev_distances)['F-based'].values()))

    # Using Kendall test, check concordance of calculated possibilities and statistics from KS-test
    euclide_kendall_result = spstats.kendalltau(euclide_similarities, kstest_values, alternative='greater')
    manhattan_kendall_result = spstats.kendalltau(manhattan_similarities, kstest_values, alternative='greater')
    chebyshev_kendall_result = spstats.kendalltau(chebyshev_similarities, kstest_values, alternative='greater')

    # Using Spearman test, check concordance of calculated possibilities and statistics from KS-test
    euclide_spearman_result = spstats.spearmanrho(euclide_similarities, kstest_values, alternative='greater')
    manhattan_spearman_result = spstats.spearmanrho(manhattan_similarities, kstest_values, alternative='greater')
    chebyshev_spearman_result = spstats.spearmanrho(chebyshev_similarities, kstest_values, alternative='greater')

    return {
        "kendall": {
            "euclide": euclide_kendall_result,
            "manhattan": manhattan_kendall_result,
            "chebyshev": chebyshev_kendall_result
        },
        "spearman": {
            "euclide": euclide_spearman_result,
            "manhattan": manhattan_spearman_result,
            "chebyshev": chebyshev_spearman_result
        }
    }

@log_with_timestamp
def experiment():
    with open('../experiments/poss-distr-concordance/results.tex', 'w') as f:
        print("------ NORMAL DATA ------")

        np.random.seed(1243)
        data = np.hstack([np.random.normal(loc=4, scale=2, size=500), np.random.normal(loc=10, scale=2, size=500)])
        res = check_concordance(data, draw_stats=True, a_part=FS(0, 4, 11, 15))
        print(res)
        
        f.write(convert_to_latex_table(res, '$f_1$'))

        print("------ MIXED DATA ------")
        
        np.random.seed(1243)
        data = np.hstack([np.random.chisquare(df=5, size=500), np.random.uniform(low=10, high=15, size=500)])
        res = check_concordance(data)
        print(res)
        f.write(convert_to_latex_table(res, '$f_2$'))

        f.write('\end{tabular}')



