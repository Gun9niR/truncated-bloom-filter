from skippingbaselines import *
from relationalquerygeneration import RelationalLoader
import matplotlib.pyplot as plt
import json
import pandas as pd
import time

def cleanup():
    for filename in os.listdir('.'):
        if filename.endswith('.bin'):
            os.remove(filename)

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


RUN = False
# CONFINTS = False
MULTIPLE_RUNS = False

DOUBLE_PLOT = False

NO_BB_PLOT = True

FPR = 0.0001

# def confidence_interval(stats, z=1.96):
#         return {name: [np.mean(stat)-z*np.std(stat, ddof=1)/np.sqrt(len(stat)), np.mean(stat)+z*np.std(stat, ddof=1)/np.sqrt(len(stat))] for name, stat in stats.items() if name != 'Index size'}


DATASETS = [['data/access_frequency_skipping_data/Real_Estate/parquet/Real_Estate_Sales_2001-2020.parquet',
                'data/access_frequency_skipping_data/Real_Estate/queries',
                'data/access_frequency_skipping_data/Real_Estate/utilities/utilities.npy'],
                ['data/access_frequency_skipping_data/Nasa/parquet/NASA.parquet',
                'data/access_frequency_skipping_data/Nasa/queries',
                'data/access_frequency_skipping_data/Nasa/utilities/utilities.npy']
                ]
DATASET_NAMES = ['RealEstate', 'NASA']

start = time.time()

if RUN:
    for trial in range(1, 7):
        print("Trial:", trial)
        if RUN:
            for dataset, dname in zip(DATASETS, DATASET_NAMES):
                print("Dataset: {}".format(dname))
                print("Trial: {}".format(trial))
                print("Time elasped (h):", (time.time()-start)/3600)
                rl = RelationalLoader(dataset[0], dataset[1], dataset[2])

                
                



                group_keys_all, column_dtypes, column_names, rg_size  = rl.extract_group_keys()
                alpha_queries, non_alpha_queries = rl.load_queries()
                utilities = rl.load_utilities()

                if dname == 'NASA':
                    k = 12
                elif dname == 'ElectricVehicles':
                    k = 5
                else:
                    k = 22
                results = {'Bloom': [], 'Hybrid Bloom': [], 'Top Utility Hybrid': []}
                
                results_ranges = {'Bloom': [], 'Hybrid Bloom': [], 'Top Utility Hybrid': []}
                crs = np.arange(0.1, 1.0, 0.1)
                fpr = FPR

                for cr in crs:
                    print("CR: {}".format(cr))
                
                    tuh = TopUtilityHybrid(column_names, column_dtypes, len(group_keys_all[0]),
                                    utilities, dataset[0], rg_size)
                    tuh.construct_indexes(group_keys_all, fpr, cr)
                    
                    hb = HybridBloom(column_names, column_dtypes, len(group_keys_all[0]),
                                    utilities, dataset[0], rg_size)
                    hb.construct_indexes(group_keys_all, fpr, cr)
                    
                    bs = BloomSkipping(column_names, column_dtypes, len(group_keys_all[0]),
                                    utilities, dataset[0], rg_size)
                    bs.construct_indexes(group_keys_all, fpr, cr)
                    
                    methods = {'Bloom': bs, 'Hybrid Bloom': hb, 'Top Utility Hybrid': tuh}
                    
                    cleanup()
                    
                    for i, q in enumerate(non_alpha_queries):
                        if i % 25 == 0:
                            print(f"Query {i}/{len(non_alpha_queries)}")
                        
                        for name, method in methods.items():
                                qresult = method.query(q.predicates, k)
                        
                    for name in results:
                            results[name].append(methods[name].average_stats())
                # write nested dictionary to json file
                with open('REVISION_RESULTS/HYBRID_{}_RGSIZE{}.json'.format(dname, trial), 'w') as fp:
                    json.dump(results, fp)
                
            

if DOUBLE_PLOT:
    ds_names = ['RealEstate', 'NASA']
    all_results = {'RealEstate': [], 'NASA': []}
    figql, axql = plt.subplots(1, 2, figsize=(14, 5))
    for trial in range(1, 7):
        
        for i, dname in enumerate(ds_names):
            # read nested dictionary from json file
            with open('REVISION_RESULTS/HYBRID_{}_RGSIZE{}.json'.format(dname, trial), 'r') as fp:
                results = json.load(fp)
        
            all_results[dname].append(results)
        
        
    for i, dname in enumerate(ds_names):
        results = {}
        avg_result = {name: [] for name in all_results[dname][0] if name != 'Bloom'}
        for result in all_results[dname]:
            for name in result:
                if name != 'Bloom':
                    avg_result[name].append(result[name])

        for name, result in avg_result.items():
            budget_list = [[] for _ in range(len(result[0]))]
            for one_run in result:
                for j, budget_run in enumerate(one_run):
                    budget_list[j].append(budget_run)
            
            budget_means = []
            for budget_run in budget_list:
                mean = {key: np.median([d[key] for d in budget_run]) for key in budget_run[0]}
                budget_means.append(mean)
            results[name] = budget_means
        
        with open('REVISION_RESULTS/NEW_HYBRID_{}_RGSIZE_AVG_DOUBLE.json'.format(dname), 'w') as fp:
            json.dump(results, fp)
            
        # axql[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Query latencies']*1000 for d in results['Bloom']], label='BB (us)', marker='*')
        axql[i].plot([d['Index size']/8e6 for d in results['Hybrid Bloom']], [d['Query latencies']*1000 for d in results['Hybrid Bloom']], label='HBB (us)', marker='*', color='navy')
        axql[i].plot([d['Index size']/8e6 for d in results['Top Utility Hybrid']], [d['Query latencies']*1000 for d in results['Top Utility Hybrid']], label='HTU', marker='o', color='green')
        axql[i].set_xlabel("Index size (MB)")
        
    axql[0].set_ylabel("Average query latency (ms)")
    
    dnames_human_friendly = ['Real Estate', 'NASA']
    
    handles, labels = axql[-1].get_legend_handles_labels()
    figql.legend(handles, labels, loc='upper center', ncols=3, borderpad=0.1)
    figql.savefig("REVISION_RESULTS/NEW_HYBRID_querylatency_vs_cr_all_1e-4_SUBPLOTS_RGSIZE_AVG_DOUBLE.png", bbox_inches='tight')