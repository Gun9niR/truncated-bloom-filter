# Optimal Bloom Filter Indexing on a Space Budget
Welcome to the repository accompanying the paper Optimal Bloom Filter Indexing on a Space Budget. Every result from the paper can be reproduced using the code from this repository. 

# Getting Started
1. Clone the repo
2. Make sure that you have Python 3 installed
3. (Optional) Create a virtual environment
4. `pip3 install requirements.txt`
5. Download the following dataset folders [1](https://drive.google.com/drive/folders/1eRfZ1cKL8zXl9aTb3uW8wtZ-AafFULAP?usp=sharing) [2](https://drive.google.com/drive/folders/1-kv1vjf8kWWft5N5vTZunOLRyfFcSyqj?usp=sharing) and place them both in the `data/` directory.

# Data processing replication
If desired, you can replicate our data processing and query generation scripts using the following notebooks: `relational.ipynb` and `dataset_analysis.ipynb`. Note that this will overwrite the data that can be used to exactly reproduce our results, since certain parts are stochastic. However, the overall results would still be practically identical. 

# Experiment replication
Each experiment can be replicated in one click. The exact data that was used to produce the results is already in the repository. Note that many of the experiments take a long time to run. Additionally, some of the experiments measure latency. Please make sure that you minimize the impact of other running processes on your machine, as this can produce noisy results. The code for each subsection in the evaluation is listed below.    

## Data Skipping
1. `python3 skippingexperiment.py`
2. `python3 disaggregatedsimulation.py`

## Full-text search
1. `python3 generalbaselineexperiment.py`

## Microbenchmarks
1. `python3 tbfexperiments.py`
2. `python3 optimizationtime.py`
3. `python3 utilityrobustness.py`
4. `python3 wrinklebaselineexperiment.py`

# Results
The plots and saved numerical values are in the folders that have the substring "results."
