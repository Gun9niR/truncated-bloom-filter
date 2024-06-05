from skippingbaselines import *
from relationalquerygeneration import RelationalLoader
import matplotlib.pyplot as plt
import json
import pandas as pd
import time
import pickle

plt.rcParams['figure.dpi'] = 300
plt.style.use("seaborn-v0_8-paper")
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'axes.labelweight': 'bold'}

# increase marker and line size
plt.rcParams['lines.markersize'] = 7.5
plt.rcParams['lines.linewidth'] = 2

plt.rcParams.update(params)

N_QUERIES = 100

def row_group_means(utils, rg_size):
    return np.array([np.mean(utils[i:i+rg_size]) for i in range(0, len(utils), rg_size)])

RUN=True
PLOT=False
PLOT_BOTH = True

crs = [0.1, 0.3, 0.5, 0.7, 0.9]
# crs = [0.2, 0.4, 0.6, 0.8]
ks = [10, 20, 30, 40, 50, 60, 70, 80]



pqpath, qpath = ('data/access_frequency_skipping_data/Real_Estate/parquet/Real_Estate_Sales_2001-2020.parquet',
        'data/access_frequency_skipping_data/Real_Estate/queries')

# read the parquet file with all of the row groups
df = pd.read_parquet(pqpath)

rl = RelationalLoader(pqpath, qpath, '')

_, nonalpha_queries = rl.load_queries()

queries = nonalpha_queries[:N_QUERIES]

utility_values_list = []

for k in ks:
    access_frequencies = np.zeros(df.shape[0])
    for q in queries:
        query_str = ' & '.join([f'{p.column_name} == "{p.value}"' for p in q.predicates])
        
        top_k_rows = df.query(query_str).head(k)
        last_idx = np.max(list(top_k_rows.index.values))
        access_frequencies[:last_idx+1] += 1
    utility_values_list.append(access_frequencies)

group_keys_all, column_dtypes, column_names, rg_size  = rl.extract_group_keys()
    
plt.figure(figsize=(15, 5))
for i, utility_values in enumerate(utility_values_list):
    plt.plot(range(1, math.ceil(utility_values.shape[0]/rg_size)+1), row_group_means(utility_values, rg_size), label=f'k={ks[i]}')
    plt.xlabel('Row group')

plt.ylabel('Utility value')
plt.legend()
plt.savefig('REVISION_RESULTS/utility_values_limit.png', bbox_inches='tight')
plt.clf()


if RUN:
    fpr = 0.0001


    
    results = {cr: [] for cr in crs}
    for cr in crs:
        print("CR: {}".format(cr))
        for k, utilities in zip(ks, utility_values_list):
            print("k: {}".format(k))

            bs = BloomSkipping(column_names, column_dtypes, len(group_keys_all[0]),
                            utilities, pqpath, rg_size)

            bs.construct_indexes(group_keys_all, fpr, cr)
        
            for q in queries:
                query_str = ' & '.join([f'{p.column_name} == "{p.value}"' for p in q.predicates])
                bs.query(q.predicates, k)
            results[cr].append(bs.average_stats()['Skip rate'])

    with open('data/limitvalue.json', 'w') as f:
        json.dump(results, f)

if PLOT:
    markers = ['o', 's', 'D', '^', 'v', 'p', 'P', 'X']
    with open('data/limitvalue.json', 'r') as f:
        results = json.load(f)

    plt.figure(figsize=(15, 6))
    for cr, marker in zip(crs, markers):
        plt.plot(ks, results[str(cr)], label=f'CR={int(cr*100)}%', marker=marker)

    plt.xlabel('Limit value (k)')
    plt.ylabel('Average skip rate')

    plt.legend()
    plt.savefig('REVISION_RESULTS/limitvalue.png', bbox_inches='tight')
    
    
if PLOT_BOTH:
    crs = [0.1, 0.3, 0.5, 0.7, 0.9] 
    markers = ['o', 's', 'D', '^', 'v', 'p', 'P', 'X']
    with open('data/limitvalue.json', 'r') as f:
        results = json.load(f)

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(15, 6))
     
    for cr, marker in zip(crs, markers):
        axs[0].plot(ks, results[str(cr)], label=f'CR={int(cr*100)}%', marker=marker)

    axs[0].set_xlabel('Limit value (k)')
    axs[0].set_ylabel('Average skip rate')
    
    
    
    axs[0].legend()
    

    crs = np.arange(0.1, 1.0, 0.1)
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
    
    
    markers = ['v', 'o', 'p']
    
    colors = ['r', 'g', 'b']
    
    for j, name in enumerate(names):
        axs[1].plot([int(cr*100) for cr in crs], [results_parsed[name][str(cr)] for cr in crs], marker=markers[j], label=name, linestyle='dashed')
    axs[1].set_xlabel('Compression ratio (%)')
    axs[1].legend()
    plt.savefig('REVISION_RESULTS/limit_value_predicate_count.png', bbox_inches='tight')

    
        


        
    


