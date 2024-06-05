from skippingbaselines import *
from relationalquerygeneration import RelationalLoader
import matplotlib.pyplot as plt
import json
import pandas as pd
import time

plt.rcParams['figure.dpi'] = 300
plt.style.use("seaborn-v0_8-paper")
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'axes.labelweight': 'bold'}

plt.rcParams['lines.markersize'] = 7.5
plt.rcParams['lines.linewidth'] = 2

plt.rcParams.update(params)


RUN = True
PLOT = False
# CONFINTS = False
RANGES = True
MULTIPLE_RUNS = True

FPR = 0.0001



DATASETS = [['data/access_frequency_skipping_data/Real_Estate/parquet/Real_Estate_Sales_2001-2020.parquet',
                'data/access_frequency_skipping_data/Real_Estate/queries',
                'data/access_frequency_skipping_data/Real_Estate/utilities/utilities.npy'],
                ['data/access_frequency_skipping_data/Electric_Vehicles/parquet/Electric_Vehicle_Population_Data.parquet',
                'data/access_frequency_skipping_data/Electric_Vehicles/queries',
                'data/access_frequency_skipping_data/Electric_Vehicles/utilities/utilities.npy'],
                ['data/access_frequency_skipping_data/Nasa/parquet/NASA.parquet',
                'data/access_frequency_skipping_data/Nasa/queries',
                'data/access_frequency_skipping_data/Nasa/utilities/utilities.npy']
                ]
DATASET_NAMES = ['RealEstate', 'ElectricVehicles', 'NASA']

start = time.time()

if RUN:
    for trial in range(1, 11):
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
                results = {'Bloom': [], 'Range': [], 'Disk': [], 'Equal Truncation': [], 'Top Utility': [], 'ElasticBF': []}
                
                results_ranges = {'Bloom': [], 'Range': [], 'Disk': [], 'Equal Truncation': [], 'Top Utility': [], 'ElasticBF': []}
                crs = np.arange(0.1, 1.0, 0.1)
                fpr = FPR

                rs = RangeSkipping(column_names, column_dtypes, len(group_keys_all[0]),
                                    utilities, dataset[0], rg_size)
                rs.construct_indexes(group_keys_all)

                ds = DiskIndex(column_names, column_dtypes, len(group_keys_all[0]),
                                    utilities, dataset[0], rg_size)
                ds.construct_indexes(group_keys_all, fpr, 1.0)
                
                time.sleep(15)

                for q in non_alpha_queries:
                    qresult = ds.query(q.predicates, k)
                
                time.sleep(15)

                for q in alpha_queries:
                    qresult = rs.query(q.predicates, k)

                results['Disk'].extend([ds.average_stats() for _ in range(len(crs))])
                results['Range'].extend([rs.average_stats() for _ in range(len(crs))])
                
                if RANGES:
                    results_ranges['Disk'].extend([ds.range_stats() for _ in range(len(crs))])
                    results_ranges['Range'].extend([rs.range_stats() for _ in range(len(crs))])

                for cr in crs:
                    print("CR: {}".format(cr))
                    bs = BloomSkipping(column_names, column_dtypes, len(group_keys_all[0]),
                                    utilities, dataset[0], rg_size)

                    bs.construct_indexes(group_keys_all, fpr, cr)
                    
                    imet = InMemEqualTrunc(column_names, column_dtypes, len(group_keys_all[0]),
                                    utilities, dataset[0], rg_size)
                    imet.construct_indexes(group_keys_all, fpr, cr)
                    
                    tu = TopUtility(column_names, column_dtypes, len(group_keys_all[0]),
                                    utilities, dataset[0], rg_size)
                    tu.construct_indexes(group_keys_all, fpr, cr)
                    
                    ebf = ElasticBF(column_names, column_dtypes, len(group_keys_all[0]),
                                    utilities, dataset[0], rg_size)
                    ebf.construct_indexes(group_keys_all, fpr, cr)
                    
                    methods = {'Bloom': bs, 'Range': rs, 'Disk': ds, 'Equal Truncation': imet, 'Top Utility' : tu, 'ElasticBF': ebf}
                    
                    time.sleep(15)
                    for i, q in enumerate(non_alpha_queries):
                        if i % 25 == 0:
                            print(f"Query {i}/{len(non_alpha_queries)}")
                        
                        for name, method in methods.items():
                            if name != 'Range' and name != 'Disk':
                                qresult = method.query(q.predicates, k)
                        
                    for name in results:
                        if name != 'Range' and name != 'Disk':
                            results[name].append(methods[name].average_stats())
                            if RANGES:
                                results_ranges[name].append(methods[name].range_stats())

                with open('REVISION_RESULTS/{}_RGSIZE{}.json'.format(dname, trial), 'w') as fp:
                    json.dump(results, fp)
                
                if RANGES:
                    with open('REVISION_RESULTS/{}_ranges_RGSIZE{}.json'.format(dname, trial), 'w') as fp:
                        json.dump(results_ranges, fp)
            
        if PLOT:
            # subplot for each metric
            figsr, axsr = plt.subplots(1, 3, figsize=(15, 5))
            figwt, axwt = plt.subplots(1, 3, figsize=(15, 5))
            figql, axql = plt.subplots(1, 3, figsize=(15, 5))
            
            
            for i, dname in enumerate(DATASET_NAMES):  
                with open('REVISION_RESULTS/{}_RGSIZE{}.json'.format(dname, trial), 'r') as fp:
                    results = json.load(fp)
                
                axsr[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Skip rate'] for d in results['Bloom']], label='BB (us)', marker='*')
                axsr[i].scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Skip rate']], label='R', marker='x', color='red')
                axsr[i].scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Skip rate']], label='D', marker='x', color='purple')
                axsr[i].plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Skip rate'] for d in results['Equal Truncation']], label='PT', marker='^')
                axsr[i].plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Skip rate'] for d in results['Top Utility']], label='TU', marker='o')
                axsr[i].plot([d['Index size']/8e6 for d in results['ElasticBF']], [d['Skip rate'] for d in results['ElasticBF']], label='EBF', marker='s')
                axsr[i].set_xlabel("Index size (MB)")
                
                axwt[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time']*1000 for d in results['Bloom']], label='BB (us)', marker='*')
                axwt[i].scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Wasted time']*1000], label='R', marker='x', color='red')
                axwt[i].scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Wasted time']*1000], label='D', marker='x', color='purple')
                axwt[i].plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Wasted time']*1000 for d in results['Equal Truncation']], label='PT', marker='^')
                axwt[i].plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Wasted time']*1000 for d in results['Top Utility']], label='TU', marker='o')
                axwt[i].plot([d['Index size']/8e6 for d in results['ElasticBF']], [d['Wasted time']*1000 for d in results['ElasticBF']], label='EBF', marker='s')
                axwt[i].set_xlabel("Index size (MB)")
                
                axql[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Query latencies']*1000 for d in results['Bloom']], label='BB (us)', marker='*')
                axql[i].scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Query latencies']*1000], label='R', marker='x', color='red')
                axql[i].scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Query latencies']*1000], label='D', marker='x',  color='purple')
                axql[i].plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Query latencies']*1000 for d in results['Equal Truncation']], label='PT', marker='^')
                axql[i].plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Query latencies']*1000 for d in results['Top Utility']], label='TU', marker='o')
                axql[i].plot([d['Index size']/8e6 for d in results['ElasticBF']], [d['Query latencies']*1000 for d in results['ElasticBF']], label='EBF', marker='s')
                axql[i].set_xlabel("Index size (MB)")
                
            axsr[0].set_ylabel("Average skip rate")
            axwt[0].set_ylabel("Average wasted time (ms)")
            axql[0].set_ylabel("Average query latency (ms)")
            
            dnames_human_friendly = ['Real Estate', 'Electric Vehicles', 'NASA']
            handles, labels = axsr[-1].get_legend_handles_labels()
            figsr.legend(handles, labels, loc='upper center', ncols=6)
            
            handles, labels = axwt[-1].get_legend_handles_labels()
            figwt.legend(handles, labels, loc='upper center', ncols=6)
            
            handles, labels = axql[-1].get_legend_handles_labels()
            figql.legend(handles, labels, loc='upper center', ncols=6)
            
            # save each subplot
            figsr.savefig("REVISION_RESULTS/skiprate_vs_cr_all_1e-4_SUBPLOTS_RGSIZE{}.png".format(trial), bbox_inches='tight')
            figwt.savefig("REVISION_RESULTS/wastedtime_vs_cr_all_1e-4_SUBPLOTS_RGSIZE{}.png".format(trial), bbox_inches='tight')
            figql.savefig("REVISION_RESULTS/querylatency_vs_cr_all_1e-4_SUBPLOTS_RGSIZE{}.png".format(trial), bbox_inches='tight')
            
            
if MULTIPLE_RUNS:
    all_results = {'RealEstate': [], 'ElectricVehicles': [], 'NASA': []}
    figsr, axsr = plt.subplots(1, 3, figsize=(14, 4))
    figwt, axwt = plt.subplots(1, 3, figsize=(14, 4))
    figql, axql = plt.subplots(1, 3, figsize=(14, 4))
    for trial in range(1, 11):
        
        for i, dname in enumerate(DATASET_NAMES):
            with open('REVISION_RESULTS/{}_RGSIZE{}.json'.format(dname, trial), 'r') as fp:
                results = json.load(fp)
        
            all_results[dname].append(results)
        
        
    for i, dname in enumerate(DATASET_NAMES):
        results = {}
        avg_result = {name: [] for name in all_results[dname][0]}
        for result in all_results[dname]:
            for name in result:
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
        
        with open('REVISION_RESULTS/{}_RGSIZE_AVG.json'.format(dname), 'w') as fp:
            json.dump(results, fp)
            
        axsr[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Skip rate'] for d in results['Bloom']], label='BB (us)', marker='*')
        axsr[i].scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Skip rate']], label='R', marker='x', color='red')
        axsr[i].scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Skip rate']], label='D', marker='x', color='purple')
        axsr[i].plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Skip rate'] for d in results['Equal Truncation']], label='PT', marker='^')
        axsr[i].plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Skip rate'] for d in results['Top Utility']], label='TU', marker='o')
        axsr[i].plot([d['Index size']/8e6 for d in results['ElasticBF']], [d['Skip rate'] for d in results['ElasticBF']], label='EBF', marker='s', color='lightcoral')
        axsr[i].set_xlabel("Index size (MB)")
        
        axwt[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time']*1000 for d in results['Bloom']], label='BB (us)', marker='*')
        axwt[i].scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Wasted time']*1000], label='R', marker='x', color='red')
        axwt[i].scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Wasted time']*1000], label='D', marker='x', color='purple')
        axwt[i].plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Wasted time']*1000 for d in results['Equal Truncation']], label='PT', marker='^')
        axwt[i].plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Wasted time']*1000 for d in results['Top Utility']], label='TU', marker='o')
        axwt[i].plot([d['Index size']/8e6 for d in results['ElasticBF']], [d['Wasted time']*1000 for d in results['ElasticBF']], label='EBF', marker='s', color='lightcoral')
        axwt[i].set_xlabel("Index size (MB)")
        
        axql[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Query latencies']*1000 for d in results['Bloom']], label='BB (us)', marker='*')
        axql[i].scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Query latencies']*1000], label='R', marker='x', color='red')
        axql[i].scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Query latencies']*1000], label='D', marker='x',  color='purple')
        axql[i].plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Query latencies']*1000 for d in results['Equal Truncation']], label='PT', marker='^')
        axql[i].plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Query latencies']*1000 for d in results['Top Utility']], label='TU', marker='o')
        axql[i].plot([d['Index size']/8e6 for d in results['ElasticBF']], [d['Query latencies']*1000 for d in results['ElasticBF']], label='EBF', marker='s', color='lightcoral')
        axql[i].set_xlabel("Index size (MB)")
        
    axsr[0].set_ylabel("Average skip rate")
    axwt[0].set_ylabel("Average wasted time (ms)")
    axql[0].set_ylabel("Average query latency (ms)")
    
    dnames_human_friendly = ['Real Estate', 'Electric Vehicles', 'NASA']

    handles, labels = axsr[-1].get_legend_handles_labels()
    figsr.legend(handles, labels, loc='upper center', ncols=6, borderpad=0.1)
    
    handles, labels = axwt[-1].get_legend_handles_labels()
    figwt.legend(handles, labels, loc='upper center', ncols=6, borderpad=0.1)
    
    handles, labels = axql[-1].get_legend_handles_labels()
    figql.legend(handles, labels, loc='upper center', ncols=6, borderpad=0.1)
    
    figsr.savefig("REVISION_RESULTS/skiprate_vs_cr_all_1e-4_SUBPLOTS_RGSIZE_AVG.png", bbox_inches='tight')
    figwt.savefig("REVISION_RESULTS/wastedtime_vs_cr_all_1e-4_SUBPLOTS_RGSIZE_AVG.png", bbox_inches='tight')
    figql.savefig("REVISION_RESULTS/querylatency_vs_cr_all_1e-4_SUBPLOTS_RGSIZE_AVG.png", bbox_inches='tight')
    