from skippingbaselines import *
from relationalquerygeneration import RelationalLoader
import matplotlib.pyplot as plt
import json
import pandas as pd

plt.rcParams['figure.dpi'] = 300
plt.style.use("seaborn-v0_8-paper")
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

RUN = True
PLOT = False
# CONFINTS = False
RANGES = False
MULTIPLE_RUNS = True

FPR = 0.0001



# def confidence_interval(stats, z=1.96):
#         return {name: [np.mean(stat)-z*np.std(stat, ddof=1)/np.sqrt(len(stat)), np.mean(stat)+z*np.std(stat, ddof=1)/np.sqrt(len(stat))] for name, stat in stats.items() if name != 'Index size'}


DATASETS = [['data/skipping_data_processed/Real_Estate/parquet/Real_Estate_Sales_2001-2020.parquet',
                'data/skipping_data_processed/Real_Estate/queries',
                'data/skipping_data_processed/Real_Estate/utilities/utilities.npy'],
                ['data/skipping_data_processed/Electric_Vehicles/parquet/Electric_Vehicle_Population_Data.parquet',
                'data/skipping_data_processed/Electric_Vehicles/queries',
                'data/skipping_data_processed/Electric_Vehicles/utilities/utilities.npy'],
                ['data/skipping_data_processed/Nasa/parquet/Nasa.parquet',
                'data/skipping_data_processed/Nasa/queries',
                'data/skipping_data_processed/Nasa/utilities/utilities.npy']
                ]
DATASET_NAMES = ['RealEstate', 'ElectricVehicles', 'NASA']

if RUN:
    for trial in range(1, 6):
        print("Trial:", trial)
        if RUN:
            # rl = RelationalLoader("data/skipping_data_processed/Real_Estate/parquet/Real_Estate_Sales_2001-2020.parquet",
            #                       "data/skipping_data_processed/Real_Estate/queries",
            #                       "data/skipping_data_processed/Real_Estate/utilities/utilities.npy")
            for dataset, dname in zip(DATASETS, DATASET_NAMES):
                print("Dataset: {}".format(dname))
                rl = RelationalLoader(dataset[0], dataset[1], dataset[2])
                
                



                group_keys_all, column_dtypes, column_names, rg_size  = rl.extract_group_keys()
                alpha_queries, non_alpha_queries = rl.load_queries()
                utilities = rl.load_utilities()

                # k = 25
                # k = 10
                if dname == 'NASA':
                    # single column, so joint distirbution contains many more rows -> bigger k
                    k = 257
                    # k = 250
                elif dname == 'ElectricVehicles':
                    k = 19
                else:
                    # k = 25
                    k = 25
                results = {'Bloom': [], 'Range': [], 'Disk': [], 'Equal Truncation': [], 'Top Utility': []}
                
                results_ranges = {'Bloom': [], 'Range': [], 'Disk': [], 'Equal Truncation': [], 'Top Utility': []}
                # crs = np.arange(0.1, 1.0, 0.1)
                crs = np.arange(0.1, 1.0, 0.1)
                # fpr = 0.001
                fpr = FPR

                # run RangeSkipping and DiskIndex once
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

                # copy the results from the first run
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
                    

                    
                    
                    
                    # ds = DiskIndex(column_names, column_dtypes, len(group_keys_all[0]),
                    #                 utilities, "data/skipping_data_processed/Real_Estate/parquet/Real_Estate_Sales_2001-2020.parquet", len(group_keys_all[0][0]))
                    # ds.construct_indexes(group_keys_all, fpr, cr)
                    
                    # rs = RangeSkipping(column_names, column_dtypes, len(group_keys_all[0]),
                    #                 utilities, "data/skipping_data_processed/Real_Estate/parquet/Real_Estate_Sales_2001-2020.parquet", len(group_keys_all[0][0]))
                    # rs.construct_indexes(group_keys_all)
                    
                    imet = InMemEqualTrunc(column_names, column_dtypes, len(group_keys_all[0]),
                                    utilities, dataset[0], rg_size)
                    imet.construct_indexes(group_keys_all, fpr, cr)
                    
                    tu = TopUtility(column_names, column_dtypes, len(group_keys_all[0]),
                                    utilities, dataset[0], rg_size)
                    tu.construct_indexes(group_keys_all, fpr, cr)
                    
                    methods = {'Bloom': bs, 'Range': rs, 'Disk': ds, 'Equal Truncation': imet, 'Top Utility' : tu}
                    
                    time.sleep(15)
                    for i, q in enumerate(non_alpha_queries):
                        if i % 25 == 0:
                            print(f"Query {i}/{len(non_alpha_queries)}")
                        
                        for name, method in methods.items():
                            if name != 'Range' and name != 'Disk':
                                qresult = method.query(q.predicates, k)
                    
                    # for i, q in enumerate(alpha_queries):
                    #     if i % 25 == 0:
                    #         print(f"Query {i}/{len(alpha_queries)}")
                        
                    #     for name, method in methods.items():
                    #         if name == 'Range':
                    #             qresult = method.query(q.predicates, k)
                        
                    for name in results:
                        if name != 'Range' and name != 'Disk':
                            results[name].append(methods[name].average_stats())
                            if RANGES:
                                results_ranges[name].append(methods[name].range_stats())
                # write nested dictionary to json file
                with open('relationalresults/{}_RGSIZE{}.json'.format(dname, trial), 'w') as fp:
                    json.dump(results, fp)
                
                if RANGES:
                    with open('relationalresults/{}_ranges_RGSIZE{}.json'.format(dname, trial), 'w') as fp:
                        json.dump(results_ranges, fp)
            
        if PLOT:
            # subplot for each metric
            figsr, axsr = plt.subplots(1, 3, figsize=(14, 3.75))
            figwt, axwt = plt.subplots(1, 3, figsize=(14, 3.75))
            figql, axql = plt.subplots(1, 3, figsize=(14, 3.75))
            
            
            for i, dname in enumerate(DATASET_NAMES):  
                        # assert len(qresult) == 25, "Query result should have 25 rows. Got {} instead.".format(len(qresult))
                # read nested dictionary from json file
                with open('relationalresults/{}_RGSIZE{}.json'.format(dname, trial), 'r') as fp:
                    results = json.load(fp)
                
                # plot skip rate vs index size in MB for each method, currently is in bits
                axsr[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Skip rate'] for d in results['Bloom']], label='Bounded Bloom (us)', marker='*')
                # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Skip rate'] for d in results['Range']], label='Range', linestyle='--')
                # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Skip rate'] for d in results['Disk']], label='Disk', linestyle='--')
                axsr[i].scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Skip rate']], label='Range', marker='x', color='red')
                axsr[i].scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Skip rate']], label='Disk', marker='x', color='purple')
                axsr[i].plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Skip rate'] for d in results['Equal Truncation']], label='Equal Truncation', marker='^')
                axsr[i].plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Skip rate'] for d in results['Top Utility']], label='Top Utility', marker='o')
                axsr[i].set_xlabel("Index size (MB)")
                
                # OLD CODE
                # # plot skip rate vs index size in MB for each method, currently is in bits
                # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Skip rate'] for d in results['Bloom']], label='Bounded Bloom (us)', marker='*')
                # # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Skip rate'] for d in results['Range']], label='Range', linestyle='--')
                # # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Skip rate'] for d in results['Disk']], label='Disk', linestyle='--')
                # plt.scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Skip rate']], label='Range', marker='x', color='red')
                # plt.scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Skip rate']], label='Disk', marker='x', color='purple')
                # plt.plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Skip rate'] for d in results['Equal Truncation']], label='Equal Truncation', marker='^')
                # plt.plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Skip rate'] for d in results['Top Utility']], label='Top Utility', marker='o')
                # plt.xlabel("Index size (MB)")
                # plt.ylabel("Skip rate")
                # plt.legend()
                # # plt.title("Skip rate vs. index size for {}".format(dname))
                # # plt.savefig("relationalresults/skiprate_vs_cr_all.png")
                # # plt.savefig("relationalresults/skiprate_vs_cr_all_tail_correction.png")
                # plt.savefig("relationalresults/skiprate_vs_cr_all_{}_1e-4.png".format(dname))
                # plt.clf()
                
                # plot wasted time vs index size for each method
                axwt[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time']*1000 for d in results['Bloom']], label='Bounded Bloom (us)', marker='*')
                # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time'] for d in results['Range']], label='Range', linestyle='--')
                # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time'] for d in results['Disk']], label='Disk', linestyle='--')
                axwt[i].scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Wasted time']*1000], label='Range', marker='x', color='red')
                axwt[i].scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Wasted time']*1000], label='Disk', marker='x', color='purple')
                axwt[i].plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Wasted time']*1000 for d in results['Equal Truncation']], label='Equal Truncation', marker='^')
                axwt[i].plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Wasted time']*1000 for d in results['Top Utility']], label='Top Utility', marker='o')
                axwt[i].set_xlabel("Index size (MB)")
                
                
                # OLD CODE
                # # plot index size vs wasted time for each method
                # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time']*1000 for d in results['Bloom']], label='Bounded Bloom (us)', marker='*')
                
                # # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time'] for d in results['Range']], label='Range', linestyle='--')
                # # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time'] for d in results['Disk']], label='Disk', linestyle='--')
                # plt.scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Wasted time']*1000], label='Range', marker='x', color = 'red')
                # plt.scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Wasted time']*1000], label='Disk', marker='x', color='purple')
                # plt.plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Wasted time']*1000 for d in results['Equal Truncation']], label='Equal Truncation', marker='^')
                # plt.plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Wasted time']*1000 for d in results['Top Utility']], label='Top Utility', marker='o')
                # # add diagonal arrow that points away from the origin that says "lower is better"
                
                # plt.xlabel("Index size (MB)")
                # plt.ylabel("Wasted time (ms)")
                # plt.legend()
                # # plt.title("Wasted time vs. index size for {}".format(dname))
                # # plt.savefig("relationalresults/wastedtime_vs_cr_all.png")
                # # plt.savefig("relationalresults/wastedtime_vs_cr_all_tail_correction.png")
                # plt.savefig("relationalresults/wastedtime_vs_cr_all_{}_1e-4.png".format(dname))
                # plt.clf()
                
                # plot query latency vs index size for each method
                axql[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Query latencies']*1000 for d in results['Bloom']], label='Bounded Bloom (us)', marker='*')
                # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Query latencies'] for d in results['Range']], label='Range', linestyle='--')
                # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Query latencies'] for d in results['Disk']], label='Disk', linestyle='--')
                axql[i].scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Query latencies']*1000], label='Range', marker='x', color='red')
                axql[i].scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Query latencies']*1000], label='Disk', marker='x',  color='purple')
                axql[i].plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Query latencies']*1000 for d in results['Equal Truncation']], label='Equal Truncation', marker='^')
                axql[i].plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Query latencies']*1000 for d in results['Top Utility']], label='Top Utility', marker='o')
                axql[i].set_xlabel("Index size (MB)")
                

                # OLD CODE
                # # plot query latency vs index size for each method
                # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Query latencies']*1000 for d in results['Bloom']], label='Bounded Bloom (us)', marker='*')
                # # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Query latencies'] for d in results['Range']], label='Range', linestyle='--')
                # # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Query latencies'] for d in results['Disk']], label='Disk', linestyle='--')
                # plt.scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Query latencies']*1000], label='Range', marker='x', color='red')
                # plt.scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Query latencies']*1000], label='Disk', marker='x',  color='purple')
                # plt.plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Query latencies']*1000 for d in results['Equal Truncation']], label='Equal Truncation', marker='^')
                # plt.plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Query latencies']*1000 for d in results['Top Utility']], label='Top Utility', marker='o')
                # plt.xlabel("Index size (MB)")
                # plt.ylabel("Query latency (ms)")
                # plt.legend()
                # # plt.title("Query latency vs. index size for {}".format(dname))
                # # plt.savefig("relationalresults/querylatency_vs_cr_all.png")
                # # plt.savefig("relationalresults/querylatency_vs_cr_all_tail_correction.png")
                # plt.savefig("relationalresults/querylatency_vs_cr_all_{}_1e-4.png".format(dname))
                # plt.clf()
                
                # RANGES, UNCOMMENT IF NEEDED
                # with open('relationalresults/{}_ranges.json'.format(dname), 'r') as fp:
                #     results_ranges = json.load(fp)
                
                # # plot maximum wasted time vs index size for each method
                # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time'][1]*1000 for d in results_ranges['Bloom']], label='Bounded Bloom (us)', marker='*')
                # # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time'][1] for d in results_ranges['Range']], label='Range', linestyle='--')
                # # plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time'][1] for d in results_ranges['Disk']], label='Disk', linestyle='--')
                # plt.scatter([results['Range'][0]['Index size']/8e6], [results_ranges['Range'][0]['Wasted time'][1]*1000], label='Range', marker='x', color='red')
                # plt.scatter([results['Disk'][0]['Index size']/8e6], [results_ranges['Disk'][0]['Wasted time'][1]*1000], label='Disk', marker='x', color='purple')
                # plt.plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Wasted time'][1]*1000 for d in results_ranges['Equal Truncation']], label='Equal Truncation', marker='^')
                # plt.plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Wasted time'][1]*1000 for d in results_ranges['Top Utility']], label='Top Utility', marker='o')
                # plt.xlabel("Index size (MB)")
                # plt.ylabel("Maximum wasted time (ms)")
                # plt.legend()
                # plt.savefig("relationalresults/maxwastedtime_vs_cr_all_{}_1e-4.png".format(dname))
                # plt.clf()
                
                
                
                

                # plt.plot(crs, [r['Skip rate'] for r in results])
                # plt.xlabel("Compression Ratio")
                # plt.ylabel("Skip Rate")
                # plt.savefig("relationalresults/skiprate_vs_cr.png")
                # plt.clf()

                # plt.plot([r['Skip rate'] for r in results], [r['Wasted time'] for r in results])
                # plt.xlabel("Skip Rate")
                # plt.ylabel("Wasted Time")
                # plt.savefig("relationalresults/wastedtime_vs_skiprate.png")
                # plt.clf()

                # plt.plot(crs, [r['Query latencies'] for r in results])
                # plt.xlabel("Compression Ratio")
                # plt.ylabel("Query Latency")
                # plt.savefig("relationalresults/querylatency_vs_cr.png")
                # plt.clf()
            # shared y axis for each subplot
            axsr[0].set_ylabel("Average skip rate")
            axwt[0].set_ylabel("Average wasted time (ms)")
            axql[0].set_ylabel("Average query latency (ms)")
            
            dnames_human_friendly = ['Real Estate', 'Electric Vehicles', 'NASA']
            # set title as dataset name
            # for i, dname in enumerate(dnames_human_friendly):
            #     # add space between words
            #     dname = dname.replace("_", " ")
            #     axsr[i].set_title(dname)
            #     axwt[i].set_title(dname)
            #     axql[i].set_title(dname)
            
            # shared legend
            handles, labels = axsr[-1].get_legend_handles_labels()
            figsr.legend(handles, labels, loc='upper center', ncols=5)
            
            handles, labels = axwt[-1].get_legend_handles_labels()
            figwt.legend(handles, labels, loc='upper center', ncols=5)
            
            handles, labels = axql[-1].get_legend_handles_labels()
            figql.legend(handles, labels, loc='upper center', ncols=5)
            
            # save each subplot
            figsr.savefig("relationalresults/skiprate_vs_cr_all_1e-4_SUBPLOTS_RGSIZE{}.png".format(trial), bbox_inches='tight')
            figwt.savefig("relationalresults/wastedtime_vs_cr_all_1e-4_SUBPLOTS_RGSIZE{}.png".format(trial), bbox_inches='tight')
            figql.savefig("relationalresults/querylatency_vs_cr_all_1e-4_SUBPLOTS_RGSIZE{}.png".format(trial), bbox_inches='tight')
            
            
if MULTIPLE_RUNS:
    all_results = {'RealEstate': [], 'ElectricVehicles': [], 'NASA': []}
    figsr, axsr = plt.subplots(1, 3, figsize=(14, 3.75))
    figwt, axwt = plt.subplots(1, 3, figsize=(14, 3.75))
    figql, axql = plt.subplots(1, 3, figsize=(14, 3.75))
    for trial in range(1, 6):
        
        for i, dname in enumerate(DATASET_NAMES):
            # read nested dictionary from json file
            with open('relationalresults/{}_RGSIZE{}.json'.format(dname, trial), 'r') as fp:
                results = json.load(fp)
        
            all_results[dname].append(results)
        
        
    for i, dname in enumerate(DATASET_NAMES):
        results = {}
        avg_result = {name: [] for name in all_results[dname][0]}
        for result in all_results[dname]:
            for name in result:
                avg_result[name].append(result[name])
        # caclulate average of each metric
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
        
        # write nested dictionary to json file
        with open('relationalresults/{}_RGSIZE_AVG.json'.format(dname), 'w') as fp:
            json.dump(results, fp)
            
        # plot all three metrics
        axsr[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Skip rate'] for d in results['Bloom']], label='Bounded Bloom (us)', marker='*')
        axsr[i].scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Skip rate']], label='Range', marker='x', color='red')
        axsr[i].scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Skip rate']], label='Disk', marker='x', color='purple')
        axsr[i].plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Skip rate'] for d in results['Equal Truncation']], label='Equal Truncation', marker='^')
        axsr[i].plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Skip rate'] for d in results['Top Utility']], label='Top Utility', marker='o')
        axsr[i].set_xlabel("Index size (MB)")
        
        axwt[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time']*1000 for d in results['Bloom']], label='Bounded Bloom (us)', marker='*')
        axwt[i].scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Wasted time']*1000], label='Range', marker='x', color='red')
        axwt[i].scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Wasted time']*1000], label='Disk', marker='x', color='purple')
        axwt[i].plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Wasted time']*1000 for d in results['Equal Truncation']], label='Equal Truncation', marker='^')
        axwt[i].plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Wasted time']*1000 for d in results['Top Utility']], label='Top Utility', marker='o')
        axwt[i].set_xlabel("Index size (MB)")
        
        axql[i].plot([d['Index size']/8e6 for d in results['Bloom']], [d['Query latencies']*1000 for d in results['Bloom']], label='Bounded Bloom (us)', marker='*')
        axql[i].scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Query latencies']*1000], label='Range', marker='x', color='red')
        axql[i].scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Query latencies']*1000], label='Disk', marker='x',  color='purple')
        axql[i].plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Query latencies']*1000 for d in results['Equal Truncation']], label='Equal Truncation', marker='^')
        axql[i].plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Query latencies']*1000 for d in results['Top Utility']], label='Top Utility', marker='o')
        axql[i].set_xlabel("Index size (MB)")
        
    # shared y axis for each subplot
    axsr[0].set_ylabel("Average skip rate")
    axwt[0].set_ylabel("Average wasted time (ms)")
    axql[0].set_ylabel("Average query latency (ms)")
    
    dnames_human_friendly = ['Real Estate', 'Electric Vehicles', 'NASA']
    # set title as dataset name
    # for i, dname in enumerate(dnames_human_friendly):
    #     # add space between words
    #     dname = dname.replace("_", " ")
    #     axsr[i].set_title(dname)
    #     axwt[i].set_title(dname)
    #     axql[i].set_title(dname)
    
    # shared legend
    handles, labels = axsr[-1].get_legend_handles_labels()
    figsr.legend(handles, labels, loc='upper center', ncols=5)
    
    handles, labels = axwt[-1].get_legend_handles_labels()
    figwt.legend(handles, labels, loc='upper center', ncols=5)
    
    handles, labels = axql[-1].get_legend_handles_labels()
    figql.legend(handles, labels, loc='upper center', ncols=5)
    
    # save each subplot
    figsr.savefig("relationalresults/skiprate_vs_cr_all_1e-4_SUBPLOTS_RGSIZE_AVG.png", bbox_inches='tight')
    figwt.savefig("relationalresults/wastedtime_vs_cr_all_1e-4_SUBPLOTS_RGSIZE_AVG.png", bbox_inches='tight')
    figql.savefig("relationalresults/querylatency_vs_cr_all_1e-4_SUBPLOTS_RGSIZE_AVG.png", bbox_inches='tight')
    
        
        
    
        
        
            
    