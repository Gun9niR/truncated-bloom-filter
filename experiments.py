from generalbaselines import Scan, InvertedIndex, TopKInvertedIndex, TopMDoc, TopMDocSet
from wrinklebaselines import EqualTruncation, TopUtility, RandomUniform, RandomUtility
# from wrinklebaselinesfixedm import EqualTruncation, TopUtility, RandomUniform, RandomUtility
from boundedbloomfast import BoundedBlooms
from boundedbloomfixedm import BoundedBlooms as BoundedBloomsFixed
from dataloader import DataLoader
from utils import Metrics
from abc import abstractmethod
import copy
import math
import time
import pandas as pd
import numpy as np
from dataloader import AmazonLoader
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
# make fonts bigger and bold
plt.rcParams['font.size'] = 18

plt.style.use("seaborn-v0_8-paper")

TARGET_FPR = 0.0001
# TARGET_FPR = 0.001
# TARGET_FPR = 0.01

class Experiment:
    def __init__(self, dataset: DataLoader):
        self.utilities = dataset.read_utilities()
        self.corpus = dataset.read_corpus()
        self.queries = dataset.read_queries()
        self.scan = Scan(self.corpus, self.utilities)
        self.scan.new_docstore(self.corpus, 'scan.npy')
    
    @abstractmethod    
    def run(self):
        pass


class GeneralExperiment(Experiment):
    def __init__(self, dataset: DataLoader):
        super().__init__(dataset)
    
    def run(self, k, compression_ratios=np.arange(0.1, 1.0, 0.05), docs_tkii=10000, outpath = 'results/general_results_k='):
        # print(len(compression_ratios))
        # print("Max utility:", np.max(self.utilities))
        # print("Min utility:", np.min(self.utilities))
        metrics = Metrics()
        bloom_basic = BoundedBlooms(int(1e5), TARGET_FPR)
        bloom_basic.add_all(self.corpus, self.utilities)
        bloom_basic.new_docstore(self.corpus, 'bloom.npy')
        bounded_bloom = copy.deepcopy(bloom_basic)
        ii = InvertedIndex(self.corpus, self.utilities)
        ii.build(self.corpus)
        
        # defines the compression ratio that is shared between bounded baselines ()
        bb_size_og = bounded_bloom.index_size()
        max_size = min(bb_size_og, ii.index_size())
        # print("Max Size:", max_size)

        names = ['Inverted Index', 'Top M Doc', 'Top M Doc Set', 'Top K Inverted Index', 'Basic Bloom', 'Bounded Bloom Disk', 'Bounded Bloom']
        index_sizes = []
        times = []
        precision_at_k = []
        utility_metric = []

        # recall_at_k = [[] for _ in range(len(names))]

        for cr in compression_ratios:
            print("Compression Ratio:", cr)
            compressed_target = int(math.floor(cr*max_size))
            
            tmd = TopMDoc(self.corpus, self.utilities, compressed_target)
            tmds = TopMDocSet(self.corpus, self.utilities, compressed_target)
            tkii = TopKInvertedIndex(self.corpus, self.utilities, docs_tkii)
            baselines = [tmd, tmds, tkii]
            

            for baseline in baselines:
                baseline.build(self.corpus)
            
            baselines = [ii] + baselines

            bounded_bloom.update_budget(compressed_target)
            bounded_bloom.update_filter_lengths(optimizer_type='jensen', rounding_scheme='floor',
                                                cval='standard', equality_constraint=True)

            baselines.extend([bloom_basic, bounded_bloom])

            local_index_sizes = [0]

            for i, baseline in enumerate(baselines):
                local_index_sizes.append(baseline.index_size())
            
            local_index_sizes.append(bounded_bloom.index_size())

            # convert to MB
            local_index_sizes = [s/8/1000/1000 for s in local_index_sizes]
            index_sizes.append(local_index_sizes)

            tot_baselines = len(names)
            local_prec = [0 for _ in range(tot_baselines)]
            # local_recall = [0 for _ in range(tot_baselines)]
            local_times = [0 for _ in range(tot_baselines)]
            local_utility_metric = [0 for _ in range(tot_baselines)]

            scan_time = 0
            scan_prec = 0
            scan_utility = 0
            for qcnt, q in enumerate(self.queries):
                start_scan = time.time()
                ground_truth = self.scan.query(q, k)
                rt_scan = time.time() - start_scan
                scan_time += rt_scan
                prec, _ = metrics.prec_rec_at_k(ground_truth, ground_truth, k)
                scan_prec += prec
                scan_utility += metrics.mean_utility_at_k(ground_truth, self.utilities, k)

                for i in range(tot_baselines):
                    start = time.time()
                    if i == tot_baselines - 1:
                        result = baselines[i-1].query(q, k, disk=False)
                        # print("Our mean utility:", metrics.mean_utility_at_k(result, self.utilities))
                        # print("Ground truth mean utility:", np.mean([self.utilities[i] for i in ground_truth]))
                        # print("Mean intersection count:", np.mean([len(set(self.corpus[i]).intersection(set(q))) for i in result]))
                    else:
                        result = baselines[i].query(q, k)
                    rt = time.time() - start
                    local_times[i] += rt
                    prec, _ = metrics.prec_rec_at_k(result, ground_truth, k)

                    local_utility_metric[i] += metrics.mean_utility_at_k(result, self.utilities, k)
                    local_prec[i] += prec
                if (qcnt) % 10 == 0:
                    print("Finished query {}/{}".format(qcnt, len(self.queries)))
            times.append([scan_time/len(self.queries)]+[lt/len(self.queries) for lt in local_times])
            precision_at_k.append([scan_prec/len(self.queries)] + [lp/len(self.queries) for lp in local_prec])
            utility_metric.append([scan_utility/len(self.queries)] + [um/len(self.queries) for um in local_utility_metric])
            # recall_at_k.append([lr/len(self.queries) for lr in local_recall])
            bounded_bloom.reset()
            assert bounded_bloom.index_size() == bb_size_og, "Not same size after reset"
        # store results in pandas data frame
        results = pd.DataFrame()
        results['Compression Ratio'] = compression_ratios
        names = ['Scan'] + names
        for i, name in enumerate(names):
            results[name + ' Index Size'] = [index_sizes[j][i] for j in range(len(compression_ratios))]
            results[name + ' Precision'] = [precision_at_k[j][i] for j in range(len(compression_ratios))]
            results[name + ' Time'] = [times[j][i] for j in range(len(compression_ratios))]
            results[name + ' Utility'] = [utility_metric[j][i] for j in range(len(compression_ratios))]
        results.to_csv(''.join([outpath, '{}.csv'.format(k)]), index=False)
    

    def plot_general_results(self, results_path, k, outpath='results', identifier=''):
        # import matplotlib.pyplot as plt
        df = pd.read_csv(results_path)
        cr = ['Compression Ratio']
        index_size_cols = [col for col in df.columns if 'Index Size' in col]
        precision_cols = [col for col in df.columns if 'Precision' in col]
        time_cols = [col for col in df.columns if 'Time' in col]
        
        names = ['Scan', 'Inverted Index', 'Top M Doc', 'Top M Doc Set', 'Top K Inverted Index', 'Basic Bloom', 'Bounded Bloom Disk', 'Bounded Bloom']
        # make dictionary of colors for each baseline
        colors = {}
        for i, name in enumerate(names):
            if name == 'Scan':
                colors[name] = 'black'
            elif name == 'Inverted Index':
                colors[name] = 'green'
            elif name == 'Top M Doc':
                colors[name] = 'green'
            elif name == 'Top M Doc Set':
                colors[name] = 'red'
            elif name == 'Top K Inverted Index':
                colors[name] = 'green'
            elif name == 'Basic Bloom':
                colors[name] = 'dodgerblue'
            elif name == 'Bounded Bloom Disk':
                colors[name] = 'dodgerblue'
            elif name == 'Bounded Bloom':
                colors[name] = 'dodgerblue'
            
        # make dictionary where baselines that share a color are given different line styles
        linestyles = {}
        for i, name in enumerate(names):
            if name == 'Scan':
                linestyles[name] = None
            elif name == 'Inverted Index':
                linestyles[name] = None
            elif name == 'Top M Doc':
                linestyles[name] = '-'
            elif name == 'Top M Doc Set':
                linestyles[name] = '-'
            elif name == 'Top K Inverted Index':
                linestyles[name] = '--'
            elif name == 'Basic Bloom':
                linestyles[name] = '-'
            elif name == 'Bounded Bloom Disk':
                linestyles[name] = '--'
            elif name == 'Bounded Bloom':
                linestyles[name] = '-'
        
        
        markers = {}
        for i, name in enumerate(names):
            if name == 'Scan':
                markers[name] = 'x'
            elif name == 'Inverted Index':
                markers[name] = 'x'
            elif name == 'Top M Doc':
                markers[name] = '^'
            elif name == 'Top M Doc Set':
                markers[name] = 'o'
            elif name == 'Top K Inverted Index':
                markers[name] = 'D'
            elif name == 'Basic Bloom':
                markers[name] = 'x'
            elif name == 'Bounded Bloom Disk':
                markers[name] = '*'
            elif name == 'Bounded Bloom':
                markers[name] = '*'
       
        
        # plot precision against index size for each baseline
        for i, baseline in enumerate(index_size_cols):
            name = baseline.replace(' Index Size', '')
            if name in ['Scan', 'Inverted Index', 'Basic Bloom']:
                plt.scatter(df[baseline][0], df[precision_cols[i]][0], label=name, color=colors[name], marker=markers[name])
            elif name in ['Top K Inverted Index']:
                plt.scatter(df[baseline][0], df[precision_cols[i]][0], label=name, color=colors[name],  marker=markers[name])
            else:
                if name == 'Bounded Bloom Disk':
                    plt.plot(df[baseline], df[precision_cols[i]], label=name + ' (us)', color=colors[name], linestyle=linestyles[name], marker=markers[name])
                elif name == 'Bounded Bloom':
                    plt.plot(df[baseline], df[precision_cols[i]], label=name + ' (us)', color=colors[name], linestyle=linestyles[name], marker=markers[name])
                else:
                    plt.plot(df[baseline], df[precision_cols[i]], label=name, color=colors[name], linestyle=linestyles[name], marker=markers[name])
        plt.xlabel('Index size (MB)')
        plt.ylabel('Precision@{}'.format(k))
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [labels.index('Bounded Bloom Disk (us)'), labels.index('Bounded Bloom (us)'), labels.index('Basic Bloom'), labels.index('Top M Doc'), labels.index('Top K Inverted Index'), labels.index('Inverted Index'), labels.index('Top M Doc Set'), labels.index('Scan')]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        plt.savefig(os.path.join(outpath,identifier+'precision_vs_index_size.png'))
        plt.clf()

        # # plot index size against compression ratio
        # sub_df = df[cr + index_size_cols]
        # # remove index size string from each index size column
        # sub_df.columns = [col.replace(' Index Size', '') for col in sub_df.columns]
        # sub_df = sub_df.set_index(cr)
        # sub_df.plot()
        # plt.xlabel('Compression Ratio')
        # plt.ylabel('Index size (KB)')
        # # plt.title('Index Size vs Compression Ratio')
        # plt.savefig(os.path.join(outpath,identifier+'index_size_vs_cr.png'))
        # plt.clf()
        

        # # plot precision against compression ratio
        # sub_df = df[cr + precision_cols]
        # # remove precision string from each precision column
        # sub_df.columns = [col.replace(' Precision', '') for col in sub_df.columns]
        # sub_df = sub_df.set_index(cr)
        # sub_df.plot()
        # plt.xlabel('Compression Ratio')
        # plt.ylabel('Precision@{}'.format(k))

        # # plt.title('Precision@{} vs Compression Ratio'.format(k))
        # plt.savefig(os.path.join(outpath,identifier+'precision_vs_cr.png'))
        # plt.clf()
        
        # plot time against index size
        for i, baseline in enumerate(index_size_cols):
            name = baseline.replace(' Index Size', '')
            if name in ['Scan', 'Inverted Index', 'Basic Bloom']:
                plt.scatter(df[baseline][0], df[time_cols[i]][0], label=name, color=colors[name], marker=markers[name])
            elif name in ['Top K Inverted Index']:
                plt.scatter(df[baseline][0], df[time_cols[i]][0], label=name, color=colors[name], marker=markers[name])
            else:
                if name == 'Bounded Bloom Disk':
                    plt.plot(df[baseline], df[time_cols[i]], label=name + ' (us)', color=colors[name], linestyle=linestyles[name], marker=markers[name])
                elif name == 'Bounded Bloom':
                    plt.plot(df[baseline], df[time_cols[i]], label=name + ' (us)', color=colors[name], linestyle=linestyles[name], marker=markers[name])
                else:
                    plt.plot(df[baseline], df[time_cols[i]], label=name, color=colors[name], linestyle=linestyles[name], marker=markers[name])
        
        plt.xlabel('Index size (MB)')
        plt.ylabel('Query latency (s)')
        # put bounded blooms and bounded bloom disk first in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [labels.index('Bounded Bloom Disk (us)'), labels.index('Bounded Bloom (us)'), labels.index('Basic Bloom'), labels.index('Top M Doc'), labels.index('Top K Inverted Index'), labels.index('Inverted Index'), labels.index('Top M Doc Set'), labels.index('Scan')]
        
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        
        plt.savefig(os.path.join(outpath,identifier+'time_vs_index_size.png'))
        plt.clf()

        # # plot time against compression ratio
        
        
        # sub_df = df[cr + time_cols]
        # # remove time string from each time column
        # sub_df.columns = [col.replace(' Time', '') for col in sub_df.columns]
        # sub_df = sub_df.set_index(cr)
        # sub_df.plot()
        # plt.xlabel('Compression Ratio')
        # plt.ylabel('Query latency (s)')
        # # plt.title('Time vs Compression Ratio')
        # plt.savefig(os.path.join(outpath,identifier+'time_vs_cr.png'))
        # plt.clf()
        
        # plot utility against index size
        utility_cols = [col for col in df.columns if 'Utility' in col]
        for i, baseline in enumerate(index_size_cols):
            name = baseline.replace(' Index Size', '')
            if name in ['Scan', 'Inverted Index', 'Basic Bloom']:
                plt.scatter(df[baseline][0], df[utility_cols[i]][0], label=name, color=colors[name])
            elif name in ['Top K Inverted Index']:
                plt.scatter(df[baseline][0], df[utility_cols[i]][0], label=name, color=colors[name], marker=markers[name])
            else:
                if name == 'Bounded Bloom Disk':
                    plt.plot(df[baseline], df[utility_cols[i]], label=name + ' (us)', color=colors[name], linestyle=linestyles[name], marker=markers[name])
                elif name == 'Bounded Bloom':
                    plt.plot(df[baseline], df[utility_cols[i]], label=name + ' (us)', color=colors[name], linestyle=linestyles[name], marker=markers[name])
                else: 
                    plt.plot(df[baseline], df[utility_cols[i]], label=name, color=colors[name], linestyle=linestyles[name], marker=markers[name])
        plt.xlabel('Index size (KB)')
        plt.ylabel('Utility')
         # put bounded blooms and bounded bloom disk first in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [labels.index('Bounded Bloom Disk (us)'), labels.index('Bounded Bloom (us)'), labels.index('Basic Bloom'), labels.index('Top M Doc'), labels.index('Top K Inverted Index'), labels.index('Inverted Index'), labels.index('Top M Doc Set'), labels.index('Scan')]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        plt.savefig(os.path.join(outpath,identifier+'utility_vs_index_size.png'))
        plt.clf()
        
        # # plot utility against compression ratio
        # utility_cols = [col for col in df.columns if 'Utility' in col]
        # sub_df = df[cr + utility_cols]
        # # remove utility string from each utility column
        # sub_df.columns = [col.replace(' Utility', '') for col in sub_df.columns]
        # sub_df = sub_df.set_index(cr)
        # sub_df.plot()
        # plt.xlabel('Compression Ratio')
        # plt.ylabel('Utility@{}'.format(k))
        # # plt.title('Utility vs Compression Ratio')
        # plt.savefig(os.path.join(outpath,identifier+'utility_vs_cr.png'))
        # plt.clf()
    
    @staticmethod
    def plot_general_results_subplots(results_paths, k, outpath='results', identifier=''):
            params = {
                # set bold axis labels
            
            'legend.fontsize': 'small',
            'axes.labelsize': 'medium',
            'axes.titlesize':'medium',
            'xtick.labelsize':'medium',
            'ytick.labelsize':'medium',
            'axes.labelweight': 'bold',
            }
            plt.rcParams.update(params)         
            dfs = [pd.read_csv(results_path) for results_path in results_paths]
            cr = ['Compression Ratio']
            index_size_cols = [col for col in dfs[0].columns if 'Index Size' in col]
            precision_cols = [col for col in dfs[0].columns if 'Precision' in col]
            time_cols = [col for col in dfs[0].columns if 'Time' in col]
            names = ['Scan', 'Inverted Index', 'Top M Doc', 'Top M Doc Set', 'Top K Inverted Index', 'Basic Bloom', 'Bounded Bloom Disk', 'Bounded Bloom']
            # make dictionary of colors for each baseline
            colors = {}
            for i, name in enumerate(names):
                if name == 'Scan':
                    colors[name] = 'black'
                elif name == 'Inverted Index':
                    colors[name] = 'green'
                elif name == 'Top M Doc':
                    colors[name] = 'green'
                elif name == 'Top M Doc Set':
                    colors[name] = 'red'
                elif name == 'Top K Inverted Index':
                    colors[name] = 'green'
                elif name == 'Basic Bloom':
                    colors[name] = 'dodgerblue'
                elif name == 'Bounded Bloom Disk':
                    colors[name] = 'dodgerblue'
                elif name == 'Bounded Bloom':
                    colors[name] = 'dodgerblue'
                
            # make dictionary where baselines that share a color are given different line styles
            linestyles = {}
            for i, name in enumerate(names):
                if name == 'Scan':
                    linestyles[name] = None
                elif name == 'Inverted Index':
                    linestyles[name] = None
                elif name == 'Top M Doc':
                    linestyles[name] = '-'
                elif name == 'Top M Doc Set':
                    linestyles[name] = '-'
                elif name == 'Top K Inverted Index':
                    linestyles[name] = '--'
                elif name == 'Basic Bloom':
                    linestyles[name] = '-'
                elif name == 'Bounded Bloom Disk':
                    linestyles[name] = '--'
                elif name == 'Bounded Bloom':
                    linestyles[name] = '-'
            
            
            markers = {}
            for i, name in enumerate(names):
                if name == 'Scan':
                    markers[name] = 'x'
                elif name == 'Inverted Index':
                    markers[name] = 'x'
                elif name == 'Top M Doc':
                    markers[name] = '^'
                elif name == 'Top M Doc Set':
                    markers[name] = 'o'
                elif name == 'Top K Inverted Index':
                    markers[name] = 'D'
                elif name == 'Basic Bloom':
                    markers[name] = 'x'
                elif name == 'Bounded Bloom Disk':
                    markers[name] = '*'
                elif name == 'Bounded Bloom':
                    markers[name] = '*'
            
            nicknames = {}
            for i, name in enumerate(names):
                if name == 'Scan':
                    nicknames[name] = 'S'
                elif name == 'Inverted Index':
                    nicknames[name] = 'II'
                elif name == 'Top M Doc':
                    nicknames[name] = 'TMII'
                elif name == 'Top M Doc Set':
                    nicknames[name] = 'TMDS'
                elif name == 'Top K Inverted Index':
                    nicknames[name] = 'TKII'
                elif name == 'Basic Bloom':
                    nicknames[name] = 'BsB'
                elif name == 'Bounded Bloom Disk':
                    nicknames[name] = 'BBD'
                elif name == 'Bounded Bloom':
                    nicknames[name] = 'BB'
        
            prec_fig, prec_ax = plt.subplots(1, len(dfs), figsize=(12, 5))
            time_fig, time_ax = plt.subplots(1, len(dfs), figsize=(12, 5))
            
            for j, df in enumerate(dfs):
                for i, baseline in enumerate(index_size_cols):
                    name = baseline.replace(' Index Size', '')
                    if name in ['Scan', 'Inverted Index', 'Basic Bloom']:
                        prec_ax[j].scatter(df[baseline][0], df[precision_cols[i]][0], label=nicknames[name], color=colors[name], marker=markers[name])
                    elif name in ['Top K Inverted Index']:
                        prec_ax[j].scatter(df[baseline][0], df[precision_cols[i]][0], label=nicknames[name], color=colors[name],  marker=markers[name])
                    else:
                        if name == 'Bounded Bloom Disk':
                            prec_ax[j].plot(df[baseline], df[precision_cols[i]], label=nicknames[name] + ' (us)', color=colors[name], linestyle=linestyles[name], marker=markers[name])
                        elif name == 'Bounded Bloom':
                            prec_ax[j].plot(df[baseline], df[precision_cols[i]], label=nicknames[name] + ' (us)', color=colors[name], linestyle=linestyles[name], marker=markers[name])
                        else:
                            prec_ax[j].plot(df[baseline], df[precision_cols[i]], label=nicknames[name], color=colors[name], linestyle=linestyles[name], marker=markers[name])
                prec_ax[j].set_xlabel("Index size (MB)")
                if j == 0:
                    prec_ax[j].set_ylabel('Precision@{}'.format(k))
                else:
                    handles, labels = prec_ax[j].get_legend_handles_labels()
                    order = [labels.index('BBD (us)'), labels.index('BB (us)'), labels.index('BsB'), labels.index('TMII'), labels.index('TKII'), labels.index('II'), labels.index('TMDS'), labels.index('S')]
                    prec_fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper center', ncols=len(order))
                    prec_fig.savefig(os.path.join(outpath,identifier+'precision_vs_index_size.png'), bbox_inches='tight')
        
            # plot time against index size
            for j, df in enumerate(dfs):
                for i, baseline in enumerate(index_size_cols):
                    name = baseline.replace(' Index Size', '')
                    if name in ['Scan', 'Inverted Index', 'Basic Bloom']:
                        time_ax[j].scatter(df[baseline][0], df[time_cols[i]][0]*1000, label=nicknames[name], color=colors[name], marker=markers[name])
                    elif name in ['Top K Inverted Index']:
                        time_ax[j].scatter(df[baseline][0], df[time_cols[i]][0]*1000, label=nicknames[name], color=colors[name], marker=markers[name])
                    else:
                        if name == 'Bounded Bloom Disk':
                            time_ax[j].plot(df[baseline], df[time_cols[i]]*1000, label=nicknames[name] + ' (us)', color=colors[name], linestyle=linestyles[name], marker=markers[name])
                        elif name == 'Bounded Bloom':
                            time_ax[j].plot(df[baseline], df[time_cols[i]]*1000, label=nicknames[name] + ' (us)', color=colors[name], linestyle=linestyles[name], marker=markers[name])
                        else:
                            time_ax[j].plot(df[baseline], df[time_cols[i]]*1000, label=nicknames[name], color=colors[name], linestyle=linestyles[name], marker=markers[name])
                time_ax[j].set_xlabel("Index size (MB)")
                if j == 0:
                    time_ax[j].set_ylabel('Average query latency (ms)')
                else:
                    handles, labels = time_ax[j].get_legend_handles_labels()
                    order = [labels.index('BBD (us)'), labels.index('BB (us)'), labels.index('BsB'), labels.index('TMII'), labels.index('TKII'), labels.index('II'), labels.index('TMDS'), labels.index('S')]
                    time_fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper center', ncols=len(order))
                    time_fig.savefig(os.path.join(outpath,identifier+'time_vs_index_size.png'), bbox_inches='tight')
                        

class WrinkleExperiment(Experiment):
    def __init__(self, dataset: DataLoader):
        super().__init__(dataset)
    
    def run(self, k, compression_ratios=np.arange(0.1, 1.0, 0.05), outpath = 'results/wrinkle_results_k=', disk=False):
        names = ['Equal Truncation', 'Top Utility', 'Random Uniform', 'Random Utility', 'Bounded Bloom']
        times = []
        precision_at_k = []
        metrics = Metrics()


        baselines = [EqualTruncation(1e5, TARGET_FPR), TopUtility(1e5, TARGET_FPR), RandomUniform(1e5, TARGET_FPR), RandomUtility(1e5, TARGET_FPR), BoundedBlooms(1e5, TARGET_FPR)]

        for baseline in baselines:
            baseline.add_all(self.corpus, self.utilities)
        if disk:
            baselines[0].new_docstore(self.corpus, 'bounded_bloom.npy')
            for i in range(1, len(baselines)):
                baselines[i].docstore = baselines[0].docstore
            
        max_size = max([b.index_size() for b in baselines])

        times = []
        precision_at_k = []
        utility_metric = []
        intersection_metric = []
        index_sizes = []

        for cr in compression_ratios:
            print("Compression Ratio:", cr)
            compressed_target = int(math.floor(cr*max_size))
            index_sizes.append(compressed_target/8/1000/1000)

            for bcnt, baseline in enumerate(baselines):
                baseline.update_budget(compressed_target)
                if bcnt == len(baselines) - 1:
                    baseline.update_filter_lengths(optimizer_type='jensen', rounding_scheme='floor',
                                                cval='standard', equality_constraint=True)
                else:
                    baseline.update_filter_lengths()


            tot_baselines = len(names)
            local_prec = [0 for _ in range(tot_baselines)]
            local_times = [0 for _ in range(tot_baselines)]
            local_utility_metric = [0 for _ in range(tot_baselines)]
            local_intersection_metric = [0 for _ in range(tot_baselines)]
            for qcnt, q in enumerate(self.queries):
                ground_truth = self.scan.query(q, k)
                # print("Ground truth mean utility:", np.mean([self.utilities[i] for i in ground_truth]))
                for j, baseline in enumerate(baselines):
                    start = time.time()
                    # result = baseline.query(q, k, disk=False)
                    result = baseline.query(q, k, disk=disk)
                    rt = time.time() - start
                    local_times[j] += rt
                    prec, _ = metrics.prec_rec_at_k(result, ground_truth, k)
                    # print("{} mean utility:".format(names[j]), metrics.mean_utility_at_k(result, self.utilities))
                    # intersections = [len(set(self.corpus[i]).intersection(set(q))) for i in result]
                    local_intersection_metric[j] += metrics.mean_intersection_percent_at_k(set(q), [set(self.corpus[i]) for i in result], k)
                    local_utility_metric[j] += metrics.mean_utility_at_k(result, self.utilities, k)
                    local_prec[j] += prec
                if (qcnt) % 10 == 0:
                    print("Finished query {}/{}".format(qcnt, len(self.queries)))
            times.append([lt/len(self.queries) for lt in local_times])
            precision_at_k.append([lp/len(self.queries) for lp in local_prec])
            utility_metric.append([um/len(self.queries) for um in local_utility_metric])
            intersection_metric.append([im/len(self.queries) for im in local_intersection_metric])
            for baseline in baselines:
                baseline.reset()
                assert baseline.index_size() == max_size, "Not same size after reset"

        results = pd.DataFrame()
        results['Compression Ratio'] = compression_ratios
        for i, name in enumerate(names):
            results[name + ' Precision'] = [precision_at_k[j][i] for j in range(len(compression_ratios))]
            results[name + ' Time'] = [times[j][i] for j in range(len(compression_ratios))]
            results[name + ' Utility'] = [utility_metric[j][i] for j in range(len(compression_ratios))]
            results[name + ' Intersection'] = [intersection_metric[j][i] for j in range(len(compression_ratios))]
            results[name + ' Index Size'] = [index_sizes[j] for j in range(len(compression_ratios))]
        results.to_csv(''.join([outpath, '{}.csv'.format(k)]), index=False)

    def plot_wrinkle_results(self, results_path, k, outpath='results', identifier=''):
        # import matplotlib.pyplot as plt
        # plot precision against index size 
        df = pd.read_csv(results_path)
        precision_cols = [col for col in df.columns if 'Precision' in col]
        time_cols = [col for col in df.columns if 'Time' in col]
        index_size_cols = [col for col in df.columns if 'Index Size' in col]
        utility_cols = [col for col in df.columns if 'Utility' in col and 'Time' not in col and 'Precision' not in col and 'Intersection' not in col and 'Index Size' not in col]
    
        intersection_cols = [col for col in df.columns if 'Intersection' in col]
        names = ['Equal Truncation', 'Top Utility', 'Random Uniform', 'Random Utility', 'Bounded Bloom']
        
        markers = {}
        for i, name in enumerate(names):
            if name == 'Equal Truncation':
                markers[name] = '^'
            elif name == 'Top Utility':
                markers[name] = 'o'
            elif name == 'Random Uniform':
                markers[name] = 'D'
            elif name == 'Random Utility':
                markers[name] = 'D'
            elif name == 'Bounded Bloom':
                markers[name] = '*'
        
        colors  = {}
        for i, name in enumerate(names):
            if name == 'Equal Truncation':
                colors[name] = 'orange'
            elif name == 'Top Utility':
                colors[name] = 'green'
            elif name == 'Random Uniform':
                colors[name] = 'purple'
            elif name == 'Random Utility':
                colors[name] = 'red'
            elif name == 'Bounded Bloom':
                colors[name] = 'dodgerblue'

        
        # plot precision against index size for each baseline
        for i, baseline in enumerate(index_size_cols):
            name = baseline.replace(' Index Size', '')
            if name == 'Bounded Bloom':
                plt.plot(df[baseline], df[precision_cols[i]], label=name + ' (us)', color=colors[name], marker=markers[name])
            else:
                plt.plot(df[baseline], df[precision_cols[i]], label=name, color=colors[name], marker=markers[name])
        
        plt.xlabel('Index size (MB)')
        plt.ylabel('Precision@{}'.format(k))
        # put bounded blooms first in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [labels.index('Bounded Bloom (us)'), labels.index('Equal Truncation'), labels.index('Top Utility'), labels.index('Random Uniform'), labels.index('Random Utility')]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        plt.savefig(os.path.join(outpath,identifier+'wrinkle_precision_vs_index_size.png'), bbox_inches='tight')
        plt.clf()
        
        # plot time against index size
        for i, baseline in enumerate(index_size_cols):
            name = baseline.replace(' Index Size', '')
            if name == 'Bounded Bloom':
                plt.plot(df[baseline], df[time_cols[i]]*1000, label=name + ' (us)', color=colors[name], marker=markers[name])
            else:
                # make sure that blue or dodger blue is not used for any other baseline
                plt.plot(df[baseline], df[time_cols[i]]*1000, label = name, color=colors[name], marker=markers[name])
        
        plt.xlabel('Index size (MB)')
        plt.ylabel('Query latency (ms)')
        # put bounded blooms first in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [labels.index('Bounded Bloom (us)'), labels.index('Equal Truncation'), labels.index('Top Utility'), labels.index('Random Uniform'), labels.index('Random Utility')]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        plt.savefig(os.path.join(outpath,identifier+'wrinkle_time_vs_index_size.png'), bbox_inches='tight')
        plt.clf()
        
        # plot utility against index size
        for i, baseline in enumerate(index_size_cols):
            name = baseline.replace(' Index Size', '')
            if name == 'Bounded Bloom':
                plt.plot(df[baseline], df[utility_cols[i]], label=name + ' (us)', color = colors[name], marker=markers[name])
            else:
                plt.plot(df[baseline], df[utility_cols[i]], label=name, color = colors[name], marker=markers[name])
        
        plt.xlabel('Index size (MB)')
        plt.ylabel('Average utility')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [labels.index('Bounded Bloom (us)'), labels.index('Equal Truncation'), labels.index('Top Utility'), labels.index('Random Uniform'), labels.index('Random Utility')]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        plt.savefig(os.path.join(outpath,identifier+'wrinkle_utility_vs_index_size.png'), bbox_inches='tight')
        plt.clf()
        
        # plot intersection against index size
        for i, baseline in enumerate(index_size_cols):
            name = baseline.replace(' Index Size', '')
            if name == 'Bounded Bloom':
                plt.plot(df[baseline], df[intersection_cols[i]], label=name + ' (us)', color=colors[name], marker=markers[name])    
            else:
                plt.plot(df[baseline], df[intersection_cols[i]], label=name, color=colors[name], marker=markers[name])
        
        plt.xlabel('Index size (MB)')
        plt.ylabel('Average overlap coefficient')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [labels.index('Bounded Bloom (us)'), labels.index('Equal Truncation'), labels.index('Top Utility'), labels.index('Random Uniform'), labels.index('Random Utility')]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        plt.savefig(os.path.join(outpath,identifier+'wrinkle_intersection_vs_index_size.png'), bbox_inches='tight')
        plt.clf()
    
    def plot_wrinkle_results_subplots(self, results_path, k, outpath='results', identifier=''):
        df = pd.read_csv(results_path)
        precision_cols = [col for col in df.columns if 'Precision' in col]
        time_cols = [col for col in df.columns if 'Time' in col]
        index_size_cols = [col for col in df.columns if 'Index Size' in col]
        utility_cols = [col for col in df.columns if 'Utility' in col and 'Time' not in col and 'Precision' not in col and 'Intersection' not in col and 'Index Size' not in col]
    
        intersection_cols = [col for col in df.columns if 'Intersection' in col]
        names = ['Equal Truncation', 'Top Utility', 'Random Uniform', 'Random Utility', 'Bounded Bloom']
        
        markers = {}
        for i, name in enumerate(names):
            if name == 'Equal Truncation':
                markers[name] = '^'
            elif name == 'Top Utility':
                markers[name] = 'o'
            elif name == 'Random Uniform':
                markers[name] = 'D'
            elif name == 'Random Utility':
                markers[name] = 'D'
            elif name == 'Bounded Bloom':
                markers[name] = '*'
        
        colors  = {}
        for i, name in enumerate(names):
            if name == 'Equal Truncation':
                colors[name] = 'orange'
            elif name == 'Top Utility':
                colors[name] = 'green'
            elif name == 'Random Uniform':
                colors[name] = 'purple'
            elif name == 'Random Utility':
                colors[name] = 'red'
            elif name == 'Bounded Bloom':
                colors[name] = 'dodgerblue'
        
        combo_fig, combo_ax = plt.subplots(1, 4, figsize=(18, 2.5))
        for i, baseline in enumerate(index_size_cols):
            name = baseline.replace(' Index Size', '')
            if name == 'Bounded Bloom':
                combo_ax[0].plot(df[baseline], df[precision_cols[i]], label=name + ' (us)', color=colors[name], marker=markers[name])
            else:
                combo_ax[0].plot(df[baseline], df[precision_cols[i]], label=name, color=colors[name], marker=markers[name])
        
        combo_ax[0].set_xlabel('Index size (MB)')
        combo_ax[0].set_ylabel('Precision@{}'.format(k))
        
        # plot intersection against index size
        for i, baseline in enumerate(index_size_cols):
            name = baseline.replace(' Index Size', '')
            if name == 'Bounded Bloom':
                combo_ax[1].plot(df[baseline], df[intersection_cols[i]], label=name + ' (us)', color=colors[name], marker=markers[name])    
            else:
                combo_ax[1].plot(df[baseline], df[intersection_cols[i]], label=name, color=colors[name], marker=markers[name])
        
        combo_ax[1].set_xlabel('Index size (MB)')
        combo_ax[1].set_ylabel('Average overlap coefficient')
        
        # plot utility against index size
        for i, baseline in enumerate(index_size_cols):
            name = baseline.replace(' Index Size', '')
            if name == 'Bounded Bloom':
                combo_ax[2].plot(df[baseline], df[utility_cols[i]], label=name + ' (us)', color = colors[name], marker=markers[name])
            else:
                combo_ax[2].plot(df[baseline], df[utility_cols[i]], label=name, color = colors[name], marker=markers[name])
        
        combo_ax[2].set_xlabel('Index size (MB)')
        combo_ax[2].set_ylabel('Average utility')
        
        # plot time against index size
        for i, baseline in enumerate(index_size_cols):
            name = baseline.replace(' Index Size', '')
            if name == 'Bounded Bloom':
                combo_ax[3].plot(df[baseline], df[time_cols[i]]*1000, label=name + ' (us)', color=colors[name], marker=markers[name])
            else:
                # make sure that blue or dodger blue is not used for any other baseline
                combo_ax[3].plot(df[baseline], df[time_cols[i]]*1000, label = name, color=colors[name], marker=markers[name])
        
        combo_ax[3].set_xlabel('Index size (MB)')
        combo_ax[3].set_ylabel('Average query latency (ms)')
        
        handles, labels = combo_ax[0].get_legend_handles_labels()
        order = [labels.index('Bounded Bloom (us)'), labels.index('Equal Truncation'), labels.index('Top Utility'), labels.index('Random Uniform'), labels.index('Random Utility')]
        combo_fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(order))
        combo_fig.savefig(os.path.join(outpath,identifier+'wrinkle_combo.png'), bbox_inches='tight')
        plt.clf()
    
        # df = pd.read_csv(results_path)
        # cr = ['Compression Ratio']

        # precision_cols = [col for col in df.columns if 'Precision' in col]
        # time_cols = [col for col in df.columns if 'Time' in col]

        # # plot precision against compression ratio
        # sub_df = df[cr + precision_cols]
        # sub_df = sub_df.set_index(cr)
        # sub_df.plot()
        # plt.xlabel('Compression Ratio')
        # plt.ylabel('Precision@{}'.format(k))

        # # plt.title('Precision@{} vs Compression Ratio'.format(k))
        # plt.savefig(os.path.join(outpath,identifier+'wrinkle_precision_vs_cr.png'))
        # plt.clf()

        # # plot time against compression ratio
        # cr = ['Compression Ratio']
        # sub_df = df[cr + time_cols]
        # sub_df = sub_df.set_index(cr)
        # sub_df.plot()
        # plt.xlabel('Compression Ratio')
        # plt.ylabel('Query latency (s)')
        # # plt.title('Time vs Compression Ratio')
        # plt.savefig(os.path.join(outpath,identifier+'wrinkle_time_vs_cr.png'))
        # plt.clf()

        # # plot utility against compression ratio
        # utility_cols = [col for col in df.columns if 'Utility' in col and 'Time' not in col and 'Precision' not in col and 'Intersection' not in col]
        # sub_df = df[cr + utility_cols]
        # sub_df = sub_df.set_index(cr)
        # sub_df.plot()
        # plt.xlabel('Compression Ratio')
        # plt.ylabel('Utility@{}'.format(k))
        # # plt.title('Utility vs Compression Ratio')
        # plt.savefig(os.path.join(outpath,identifier+'wrinkle_utility_vs_cr.png'))
        # plt.clf()

        # # plot intersection against compression ratio
        # intersection_cols = [col for col in df.columns if 'Intersection' in col]
        # sub_df = df[cr + intersection_cols]
        # sub_df = sub_df.set_index(cr)
        # sub_df.plot()
        # plt.xlabel('Compression Ratio')
        # plt.ylabel('Overlap coefficient@{}'.format(k))
        # # plt.title('Overlap coefficient vs Compression Ratio')
        # plt.savefig(os.path.join(outpath,identifier+'wrinkle_intersection_vs_cr.png'))
        # plt.clf()

class BloomyComparison(Experiment):
    def __init__(self, dataset: DataLoader):
        super().__init__(dataset)
        
    def run(self, k, compression_ratios=np.arange(0.1, 1.0, 0.05), outpath = 'results/bloomy_comparison_k=', disk=False):
        names = ["Bounded Bloom", "Bounded Bloom Fixed M"]
        times = []
        precision_at_k = []
        metrics = Metrics()
    


        baselines = [BoundedBlooms(1e5, TARGET_FPR), BoundedBloomsFixed(1e5, m=np.max([len(doc) for doc in self.corpus])*10)]

        for baseline in baselines:
            baseline.add_all(self.corpus, self.utilities)
        if disk:
            baselines[0].new_docstore(self.corpus, 'bounded_bloom.npy')
            for i in range(1, len(baselines)):
                baselines[i].docstore = baselines[0].docstore
        index_sizes = [b.index_size() for b in baselines]

        times = []
        precision_at_k = []
        utility_metric = []
        intersection_metric = []
        index_sizes_global = []

        for cr in compression_ratios:
            print("Compression Ratio:", cr)
            

            for bcnt, baseline in enumerate(baselines):
                compressed_target = int(math.floor(cr*index_sizes[bcnt]))
                baseline.update_budget(compressed_target)
                baseline.update_filter_lengths(optimizer_type='jensen', rounding_scheme='floor',
                                            cval='standard', equality_constraint=True)

            index_sizes_global.append([b.index_size()/8000 for b in baselines])
            tot_baselines = len(names)
            local_prec = [0 for _ in range(tot_baselines)]
            local_times = [0 for _ in range(tot_baselines)]
            local_utility_metric = [0 for _ in range(tot_baselines)]
            local_intersection_metric = [0 for _ in range(tot_baselines)]
            for qcnt, q in enumerate(self.queries):
                ground_truth = self.scan.query(q, k)
                # print("Ground truth mean utility:", np.mean([self.utilities[i] for i in ground_truth]))
                for j, baseline in enumerate(baselines):
                    start = time.time()
                    # result = baseline.query(q, k, disk=False)
                    result = baseline.query(q, k, disk=disk)
                    rt = time.time() - start
                    local_times[j] += rt
                    prec, _ = metrics.prec_rec_at_k(result, ground_truth, k)
                    # print("{} mean utility:".format(names[j]), metrics.mean_utility_at_k(result, self.utilities))
                    # intersections = [len(set(self.corpus[i]).intersection(set(q))) for i in result]
                    local_intersection_metric[j] += metrics.mean_intersection_percent_at_k(set(q), [set(self.corpus[i]) for i in result], k)
                    local_utility_metric[j] += metrics.mean_utility_at_k(result, self.utilities, k)
                    local_prec[j] += prec
                if (qcnt) % 10 == 0:
                    print("Finished query {}/{}".format(qcnt, len(self.queries)))
            times.append([lt/len(self.queries) for lt in local_times])
            precision_at_k.append([lp/len(self.queries) for lp in local_prec])
            utility_metric.append([um/len(self.queries) for um in local_utility_metric])
            intersection_metric.append([im/len(self.queries) for im in local_intersection_metric])
            for bcnt, baseline in enumerate(baselines):
                baseline.reset()
                assert baseline.index_size() == index_sizes[bcnt], "Not same size after reset"

        results = pd.DataFrame()
        results['Compression Ratio'] = compression_ratios
        for i, name in enumerate(names):
            results[name + ' Index Size'] = [index_sizes_global[j][i] for j in range(len(compression_ratios))]
            results[name + ' Precision'] = [precision_at_k[j][i] for j in range(len(compression_ratios))]
            results[name + ' Time'] = [times[j][i] for j in range(len(compression_ratios))]
            results[name + ' Utility'] = [utility_metric[j][i] for j in range(len(compression_ratios))]
            results[name + ' Intersection'] = [intersection_metric[j][i] for j in range(len(compression_ratios))]
        results.to_csv(''.join([outpath, '{}.csv'.format(k)]), index=False)
        
    def plot_bb_compare_result(self, results_path, k, outpath='results', identifier='bb_compare'):
        # import matplotlib.pyplot as plt
        df = pd.read_csv(results_path)
        cr = ['Compression Ratio']
        index_size_cols = [col for col in df.columns if 'Index Size' in col]
        precision_cols = [col for col in df.columns if 'Precision' in col]
        time_cols = [col for col in df.columns if 'Time' in col]

        # plot index size against compression ratio
        sub_df = df[cr + index_size_cols]
        sub_df = sub_df.set_index(cr)
        sub_df.plot()
        plt.xlabel('Compression Ratio')
        plt.ylabel('Index Size (KB)')
        # plt.title('Index Size vs Compression Ratio')
        plt.savefig(os.path.join(outpath,identifier+'index_size_vs_cr.png'))
        plt.clf()

        # plot precision against compression ratio
        sub_df = df[cr + precision_cols]
        sub_df = sub_df.set_index(cr)
        sub_df.plot()
        plt.xlabel('Compression Ratio')
        plt.ylabel('Precision@{}'.format(k))

        # plt.title('Precision@{} vs Compression Ratio'.format(k))
        plt.savefig(os.path.join(outpath,identifier+'precision_vs_cr.png'))
        plt.clf()

        # plot time against compression ratio
        sub_df = df[cr + time_cols]
        sub_df = sub_df.set_index(cr)
        sub_df.plot()
        plt.xlabel('Compression Ratio')
        plt.ylabel('Query latency (s)')
        # plt.title('Time vs Compression Ratio')
        plt.savefig(os.path.join(outpath,identifier+'time_vs_cr.png'))
        plt.clf()

        # plot utility against compression ratio
        utility_cols = [col for col in df.columns if 'Utility' in col]
        sub_df = df[cr + utility_cols]
        sub_df = sub_df.set_index(cr)
        sub_df.plot()
        plt.xlabel('Compression Ratio')
        plt.ylabel('Utility@{}'.format(k))
        # plt.title('Utility vs Compression Ratio')
        plt.savefig(os.path.join(outpath,identifier+'utility_vs_cr.png'))
        plt.clf()       


class VaryingFPR(Experiment):
    def __init__(self, dataset: DataLoader, fprs=[0.01, 0.001, 0.0001]):
        super().__init__(dataset)
        self.fprs = fprs
    
    def run(self, k, compression_ratios=np.arange(0.1, 1.0, 0.05), outpath = 'results/varying_fpr_k='):
        names = ["Top Utility", "Equal Truncation", "Bounded Bloom"]
        metrics = Metrics()
        
        precision_diff_fprs = []
    
        for fpr in self.fprs:
            print("FPR:", fpr)
            precision_at_k = []
            baselines = [TopUtility(1e5, fpr), EqualTruncation(1e5, fpr), BoundedBlooms(1e5, fpr)]
            
            for baseline in baselines:
                baseline.add_all(self.corpus, self.utilities)
                
            max_size = max([b.index_size() for b in baselines])
            
            for cr in compression_ratios:
                print("Compression Ratio:", cr)
                compressed_target = int(math.floor(cr*max_size))

                for bcnt, baseline in enumerate(baselines):
                    baseline.update_budget(compressed_target)
                    if bcnt == len(baselines) - 1:
                        baseline.update_filter_lengths(optimizer_type='jensen', rounding_scheme='floor',
                                                    cval='standard', equality_constraint=True)
                    else:
                        baseline.update_filter_lengths()


                tot_baselines = len(names)
                local_prec = [0 for _ in range(tot_baselines)]
                for qcnt, q in enumerate(self.queries):
                    ground_truth = self.scan.query(q, k)
                    for j, baseline in enumerate(baselines):
                        result = baseline.query(q, k, disk=False)
                        prec, _ = metrics.prec_rec_at_k(result, ground_truth, k)
                        local_prec[j] += prec
                    if (qcnt) % 10 == 0:
                        print("Finished query {}/{}".format(qcnt, len(self.queries)))
                precision_at_k.append([lp/len(self.queries) for lp in local_prec])
                for baseline in baselines:
                    baseline.reset()
                    assert baseline.index_size() == max_size, "Not same size after reset"
            precision_diff_fprs.append(precision_at_k)
        results = pd.DataFrame()
        results['Compression Ratio'] = compression_ratios
        for i, name in enumerate(names):
            for j, fpr in enumerate(self.fprs):
                results[name + ' FPR={}'.format(fpr)] = [precision_diff_fprs[j][k][i] for k in range(len(compression_ratios))]
                
        results.to_csv(''.join([outpath, '{}.csv'.format(k)]), index=False)
        
    def plot_varying_fpr(self, results_path, k, outpath='results', identifier='varying_fpr', linestyles = ['-', '--', '-.'], markers = ['o', 'v', 's']):
        # import matplotlib.pyplot as plt
        df = pd.read_csv(results_path)
        cr = ['Compression Ratio']
        precision_cols = [col for col in df.columns if 'Compression Ratio' not in col]
        
        red_columns = [col for col in precision_cols if 'Bounded Bloom' in col]
        blue_columns = [col for col in precision_cols if 'Top Utility' in col]
        green_columns = [col for col in precision_cols if 'Equal Truncation' in col]
        
        # different linestyles for different fprs
        
        plt.figure()
        
        for ls, fpr, mrk in zip(linestyles, self.fprs, markers):
            red_lines = [col for col in red_columns if 'FPR={}'.format(fpr) in col]
            blue_lines = [col for col in blue_columns if 'FPR={}'.format(fpr) in col]
            green_lines = [col for col in green_columns if 'FPR={}'.format(fpr) in col]
            plt.plot(df[cr], df[red_lines], linestyle=ls, marker=mrk, color='red', label='Bounded Bloom FPR={}'.format(fpr))
            plt.plot(df[cr], df[blue_lines], linestyle=ls, marker=mrk, color='blue', label='Top Utility FPR={}'.format(fpr))
            plt.plot(df[cr], df[green_lines], linestyle=ls, marker=mrk, color='green', label='Equal Truncation FPR={}'.format(fpr))
            
        plt.xlabel('Compression Ratio')
        plt.ylabel('Precision@{}'.format(k))
        plt.title('Precision@{} vs Compression Ratio'.format(k))
        plt.legend()
        plt.savefig(os.path.join(outpath,identifier+'precision_vs_cr.png'))
        plt.clf()
            