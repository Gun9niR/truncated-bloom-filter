from experiments import GeneralExperiment
from dataloader import AmazonLoader
import numpy as np

INDUSTRIAL_SCIENTIFIC=True
MUSIC=True
PLOT_BOTH = True

# dataset = AmazonLoader('data/processed', 'amazon_industrial_scientific')
# dataset = AmazonLoader('data/processed', 'amazon_industrial_scientific_v2')

# dataset = AmazonLoader('data/processed', 'amazon_industrial_scientific_v3')

if INDUSTRIAL_SCIENTIFIC:
    k = 1
    dataset = AmazonLoader('data/processed', 'amazon_industrial_scientific_v3ls')
    exp = GeneralExperiment(dataset)
    exp.run(k, np.arange(0.1, 1.0, 0.1), 300, outpath="results/general_results_v3ls_k=")
    exp.plot_general_results("results/general_results_v3ls_k={}.csv".format(k), k, outpath="results", identifier="general_results_v3ls")

if MUSIC:
    k = 1
    dataset = AmazonLoader('data/processed', 'amazon_musical_instruments')
    exp = GeneralExperiment(dataset)
    exp.run(k, np.arange(0.1, 1.0, 0.1), 300, outpath="results/general_results_music=")
    exp.plot_general_results("results/general_results_music={}.csv".format(k), k, outpath="results", identifier="general_results_music")

if PLOT_BOTH:
    k = 1
    GeneralExperiment.plot_general_results_subplots(["results/general_results_v3ls_k={}.csv".format(k),
                                                     "results/general_results_music={}.csv".format(k)], k, outpath="results", identifier="general_results_SUBPLOTS")
# k = 5
# dataset.make_pipeline('data/raw/Industrial_and_Scientific_5.json.gz', 5, 300, k, lower_token_filter=5, upper_token_filter=100, start=5)
# dataset.make_pipeline('data/raw/Industrial_and_Scientific_5.json.gz', 5, 100, k, distribution_type='norm', lower_token_filter=5, upper_token_filter=100, start=2)
# dataset.make_pipeline('data/raw/Industrial_and_Scientific_5.json.gz', 5, 50, k, distribution_type='normmix',
#                       lower_token_filter=5, upper_token_filter=100, start=4, nmix=5, density_factor=2,
#                       spread_factor=3.0)
# dataset.make_pipeline('data/raw/Musical_Instruments_5.json.gz', 5, 50, k, distribution_type='normmix',
#                       lower_token_filter=5, upper_token_filter=100, start=3, nmix=5, density_factor=2,
#                       spread_factor=3.5)
# print("Processing completed")


# exp.run(k, np.arange(0.1, 1.0, 0.1), 200)

# exp.run(k, np.arange(0.2, 1.0, 0.1), 300, outpath="results/general_results_v3_k=")
# exp.run(k, np.arange(0.1, 1.0, 0.1), 300, outpath="results/general_results_v3ls_k=")
# exp.run(k, np.arange(0.2, 1.0, 0.1), 300, outpath="results/general_results_music=")

# exp.plot_general_results("results/general_results_k={}.csv".format(k), k, outpath="results/general_results_v2")
# exp.plot_general_results("results/general_results_v3_k={}.csv".format(k), k, outpath="results", identifier="general_results_v3")







