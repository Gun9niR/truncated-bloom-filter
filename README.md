# Optimizing Collections of Bloom Filters within a Space Budget
Welcome to the repository accompanying the paper Optimizing Collections of Bloom Filters within a Space Budget. Every result from the paper can be produced using the code from this repository. 

# Getting Started
1. Clone the repo
2. Make sure that you have Python 3 installed
3. (Optional) Create a virtual environment
4. `pip3 install requirements.txt`
5. Download the following dataset folders [1](https://drive.google.com/drive/folders/1eRfZ1cKL8zXl9aTb3uW8wtZ-AafFULAP?usp=sharing) [2](https://drive.google.com/drive/folders/1-kv1vjf8kWWft5N5vTZunOLRyfFcSyqj?usp=sharing) and place them both in the `data/` directory.

# Data processing replication
If desired, you can replicate our data processing and query generation scripts using the following notebooks: `relational.ipynb` and `dataset_analysis.ipynb`. Note that this will overwrite the data that can be used to reproduce our results, since certain parts are stochastic. However, the overall results would still be practically identical, if one follows the advice given below. 

# Experiment replication
Each experiment can be run in one click. The exact data that was used to produce the results is already in the repository, but there is some inherent randomness due to the probabablistic nature of a Bloom filter. Note that many of the experiments take a long time to run. Additionally, some of the experiments measure latency. Please make sure that you minimize the impact of other running processes on your machine, as this can produce noisy results. The code that produces the figures for each subsection in the evaluation is listed below.    

## Data Skipping
1. Figures 2, 3, 4 -> `python3 skippingexperiment.py`
2. Figure 5 -> `python3 disaggregatedsimulation.py`

## Full-text search
1. Figures 6, 7 -> `python3 generalbaselineexperiment.py`

## Microbenchmarks
1. Figure 8 -> `python3 tbfexperiments.py`
2. Figure 9 -> `python3 optimizationtime.py`
3. Figure 10 -> `python3 utilityrobustness.py`
4. Figure 11 -> `python3 wrinklebaselineexperiment.py`

# Results
The plots and saved numerical values are in the folders that have the substring "results."
