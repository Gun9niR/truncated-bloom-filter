# Optimizing Collections of Bloom Filters within a Space Budget
Welcome to the repository accompanying the paper Optimizing Collections of Bloom Filters within a Space Budget. Every result from the paper can be produced using the code from this repository. 

# Getting Started
1. Clone the repo
2. Make sure that you have Python 3 installed
3. (Optional) Create a virtual environment
4. `pip3 install requirements.txt`

# Data processing replication
If desired, you can replicate our data processing and query generation scripts using the following notebooks: `relational.ipynb` and `dataset_analysis.ipynb`. Note that this will overwrite the data that can be used to reproduce our results, since certain parts are stochastic. However, the overall results would still be practically identical, if one follows the advice given below. 

# Experiment replication
Each experiment can be run in one click. The exact data that was used to produce the results is already in the repository, but there is some inherent randomness due to the probabablistic nature of a Bloom filter. Note that many of the experiments take a long time to run. Additionally, some of the experiments measure latency. Please make sure that you minimize the impact of other running processes on your machine, as this can produce noisy results. The code that produces the figures for each subsection in the evaluation is listed below.    

## Data Skipping
1. Figures 2, 3, 4 -> `python3 skippingexperiment.py`
2. Figure 5 -> `python3 skipping_limitvalue.py` and `python3 skipping_predicate_count.py`
3. Figure 6 -> `python3 hybrid_skipping_experiment.py`

## Full-text search
1. Figures 7, 8 -> `python3 generalbaselineexperiment.py`

## Microbenchmarks
1. Figure 9 -> `python3 tbfexperiments.py`
2. Table 1 -> `python3 optimizationtime.py`

# Results
The plots and saved numerical values are in the folder "REVISION_RESULTS."
