from experiments import WrinkleExperiment
from dataloader import AmazonLoader
import numpy as np

RUN=False
PLOT=True


# dataset = AmazonLoader('data/processed', 'amazon_industrial_scientific')
# dataset = AmazonLoader('data/processed', 'amazon_industrial_scientific_v2')
dataset = AmazonLoader('data/processed', 'amazon_industrial_scientific_v3ls')
# dataset.make_pipeline('data/raw/Industrial_and_Scientific_5.json.gz', 5, 100, k, distribution_type='norm', lower_token_filter=5, upper_token_filter=100, start=2)
k = 1

# k = 5
# dataset.make_pipeline('data/raw/Industrial_and_Scientific_5.json.gz', 5, 25, k, distribution_type='binorm', lower_token_filter=5, upper_token_filter=100, start=2)
# dataset.make_pipeline('data/raw/Industrial_and_Scientific_5.json.gz', 5, 100, k, distribution_type='norm', lower_token_filter=5, upper_token_filter=100, start=2)

# dataset.make_pipeline('data/raw/Industrial_and_Scientific_5.json.gz', 5, 500, k, distribution_type='normmix',
#                       lower_token_filter=5, upper_token_filter=100, start=2, nmix=5, density_factor=2,
#                       spread_factor=3.5)
# print("Processing completed")

we = WrinkleExperiment(dataset)
# we.run(k, np.arange(0.5, 1.0, 0.1), outpath='results/wrinkle_results_v2.2_k=')

# we.run(k, np.arange(0.2, 1.0, 0.1), outpath='results/wrinkle_results_v3ls_k=', disk=True)

# we.plot_wrinkle_results('results/wrinkle_results_v2_k={}.csv'.format(k), k, identifier='wrinkle_results_v2.2k{}'.format(k))


# we.plot_wrinkle_results('results/wrinkle_results_v3_k={}.csv'.format(k), k, identifier='wrinkle_results_v3k{}'.format(k))


# we.run(k, np.arange(0.2, 1.0, 0.1), outpath='results/wrinkle_results_v3diskFalse_k=', disk=False)

# we.run(k, np.arange(0.1, 1.0, 0.1), outpath='results/wrinkle_results_v3lsdiskFalse_k=', disk=False)

# we.plot_wrinkle_results('results/wrinkle_results_v2_k={}.csv'.format(k), k, identifier='wrinkle_results_v2.2k{}'.format(k))
if RUN:
    we.run(k, np.arange(0.1, 1.0, 0.1), outpath='results/wrinkle_results_v3lsdiskFalse_k=', disk=False)

if PLOT:
    we.plot_wrinkle_results_subplots('results/wrinkle_results_v3lsdiskFalse_k={}.csv'.format(k), k, identifier='wrinkle_results_v3lsdiskFalsek{}'.format(k))