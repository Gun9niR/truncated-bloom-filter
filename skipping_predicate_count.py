from skippingbaselines import *
from relationalquerygeneration import RelationalLoader
import matplotlib.pyplot as plt
import json
import pandas as pd
import time
import itertools
import pickle
from columnindexes import Query

plt.rcParams['figure.dpi'] = 300
plt.style.use("seaborn-v0_8-paper")
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}

# increase marker and line size
plt.rcParams['lines.markersize'] = 7.5
plt.rcParams['lines.linewidth'] = 2

plt.rcParams.update(params)

N_QUERIES = 100

RUN=True
PLOT=True

N_QUERIES = 100


pqpath, qpath, util_path = ('data/access_frequency_skipping_data/Real_Estate/parquet/Real_Estate_Sales_2001-2020.parquet',
                            'data/access_frequency_skipping_data/Real_Estate/queries',
                            'data/access_frequency_skipping_data/Real_Estate/utilities/utilities.npy')

rl = RelationalLoader(pqpath, qpath, util_path)

group_keys_all, column_dtypes, column_names, rg_size  = rl.extract_group_keys()

_, nonalpha_queries = rl.load_queries()

queries = nonalpha_queries[:N_QUERIES]

utilities = rl.load_utilities()

BITS_TO_MB = 8e6


crs = np.arange(0.1, 1.0, 0.1)

k = 45

if RUN:
    fpr = 0.0001
    
    col_idxs = list(range(len(group_keys_all)))
    
    
    results = {float(cr): [] for cr in crs}
    index_sizes = []
    names = []
    
    for i in range(1, len(group_keys_all)+1):
        for comb in itertools.combinations(col_idxs, i):
            group_keys_subset = [group_keys_all[j] for j in comb]
            col_name_subset = [column_names[j] for j in comb]
            col_dtype_subset = [column_dtypes[j] for j in comb]
    
            
            queries_predicate_subset = []
    
                        
                
            
            queries_predicate_subset = [Query([p for p in q.predicates if p.column_name in col_name_subset]) for q in queries]
            
            names.append(', '.join(col_name_subset))
            
            print('Processing', names[-1])
            for cr in crs:
                print('Compression ratio:', cr)
                bs = BloomSkipping(col_name_subset, col_dtype_subset, len(group_keys_subset[0]),
                                utilities, pqpath, rg_size)

                bs.construct_indexes(group_keys_subset, fpr, cr)
                
                for q in queries_predicate_subset:
                    bs.query(q.predicates, k)
                
                results[cr].append(bs.average_stats()['Skip rate'])
    
    with open('REVISION_RESULTS/names.pkl', 'wb') as fp:
        pickle.dump(names, fp)
    
    with open('REVISION_RESULTS/index_sizes.pkl', 'wb') as fp:
        pickle.dump(index_sizes, fp)
         
    with open('REVISION_RESULTS/results.json', 'w') as fp:
        json.dump(results, fp)
        
if PLOT:
    with open('REVISION_RESULTS/names.pkl', 'rb') as fp:
        names = pickle.load(fp)
        
    with open('REVISION_RESULTS/results.json', 'r') as fp:
        results = json.load(fp)
    
    results_parsed = {name: {cr: [] for cr in crs} for name in names}
    
    for j in range(len(names)):
        for cr in crs:
            results_parsed[names[j]][str(cr)] = results[str(cr)][j]
    with open('REVISION_RESULTS/results_parsed.json', 'w') as fp:
        json.dump(results_parsed, fp)
    
    fig = plt.figure(figsize=(15, 10))
    
    markers = ['o', 's', 'd', '^']
    
    for j, name in enumerate(names):
        plt.plot([int(cr*100) for cr in crs], [results_parsed[name][str(cr)] for cr in crs], marker=markers[j], label=name)
    plt.legend()
    plt.xlabel('Compression ratio (%)')
    plt.ylabel('Average skip rate')
    plt.savefig('REVISION_RESULTS/MULTI_PREDICATE_skiprate_indexsize.png', bbox_inches='tight')
        