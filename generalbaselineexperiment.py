from experiments import GeneralExperiment
from dataloader import AmazonLoader
import numpy as np
import matplotlib.pyplot as plt

INDUSTRIAL_SCIENTIFIC=True
MUSIC=True
PLOT_BOTH = True


if INDUSTRIAL_SCIENTIFIC:
    k = 1
    dataset = AmazonLoader('data/processed', 'REVISION_amazon_industrial_scientific_v3ls')
    exp = GeneralExperiment(dataset)
    crs = np.arange(0.1, 1.0, 0.1)
    crs[0] = 0.103
    
    utilities=dataset.read_utilities()
    plt.hist(utilities, bins=100)
    plt.savefig("utilities_hist.png")
    exp.run(k, crs, 300, outpath="REVISION_RESULTS/general_results_v3ls_k=")
    exp.plot_general_results("REVISION_RESULTS/general_results_v3ls_k={}.csv".format(k), k, outpath="REVISION_RESULTS", identifier="general_results_v3ls")

if MUSIC:
    k = 1
    dataset = AmazonLoader('data/processed', 'REVISION_amazon_musical_instruments')
    exp = GeneralExperiment(dataset)
    exp.run(k, np.arange(0.1, 1.0, 0.1), 300, outpath="REVISION_RESULTS/general_results_music=")
    exp.plot_general_results("REVISION_RESULTS/general_results_music={}.csv".format(k), k, outpath="REVISION_RESULTS", identifier="general_results_music")

if PLOT_BOTH:
    k = 1
    GeneralExperiment.plot_general_results_subplots(["REVISION_RESULTS/general_results_v3ls_k={}.csv".format(k),
                                                     "REVISION_RESULTS/general_results_music={}.csv".format(k)], k, outpath="REVISION_RESULTS", identifier="general_results_SUBPLOTS")





