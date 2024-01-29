from dataloader import AmazonLoader
from boundedbloomfast import BoundedBlooms
from utils import Metrics
import numpy as np
import math
from generalbaselines import Scan
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

plt.style.use("seaborn-v0_8-paper")

TARGET_FPR = 0.0001

k = 1
dataset = AmazonLoader('data/processed', 'amazon_industrial_scientific_utilskew')

distributions = ['norm', 'uni', 'normmix_1.1', 'normmix_2.1', 'normmix_3.1']
# distributions = ['normmix_1.25', 'normmix_2.25', 'normmix_3.25']
plot_friendly_names = ['Normal(1, 0.1)', 'Uniform(0.45, 1.55)', 'Skewed (ρ=1.1)', 'Skewed (ρ=2.1)', 'Skewed (ρ=3.1)']
# plot_friendly_names = ['Skewed (1.25)', 'Skewed (2.25)', 'Skewed (3.25)']

dataset.load_raw('data/raw/Industrial_and_Scientific_5.json.gz')
dataset.clean(5, 100)

corpus = dataset.read_corpus()

metrics  = Metrics()


NQUE = 50

results_prec = [[] for _ in range(len(distributions))]
index_sizes = [[] for _ in range(len(distributions))]
for i, d in enumerate(distributions):
    print("Running {}".format(d))
    if "normmix" in d:
        spread_factor = float(d.split('_')[1])  
        dataset.write_utilities(distribution_type="normmix", spread_factor=spread_factor)
    else:
        dataset.write_utilities(distribution_type=d)
    

    dataset.write_ngram_queries(5, NQUE, k, 3)
    utilities = dataset.read_utilities()
    bb = BoundedBlooms(int(1e5), TARGET_FPR)
    bb.add_all(corpus, utilities)
    bb_size_og = bb.index_size()
    
    queries = dataset.read_queries()
    
    scan = Scan(corpus, utilities)
    scan.new_docstore(corpus, 'scan.npy')
    
    for cr in np.arange(0.1, 0.95, 0.05):
        print(cr)
        compressed_target = int(math.floor(cr*bb_size_og))
        bb.update_budget(compressed_target)
        
        try:
            bb.update_filter_lengths(optimizer_type='jensen', rounding_scheme='floor', cval='standard', equality_constraint=True)
        except:
            print("Numerical instability".format(cr))
            continue     
            
                
            
        
        local_prec = []
        for qcnt, q in enumerate(queries):
            if qcnt % 10 == 0:
                print("Query {}/{}".format(qcnt, len(queries)))
            ground_truth = scan.query(q, k)
            result = bb.query(q, k, disk=False)
            prec, _ = metrics.prec_rec_at_k(result, ground_truth, k)
            local_prec.append(prec)
        index_sizes[i].append(bb.index_size()/8/1000/1000)
        print('Index size:', bb.index_size()/8/1000/1000)
        avg_prec = np.mean(local_prec)
        results_prec[i].append(avg_prec)
        bb.reset()
        assert bb.index_size() == bb_size_og, "Index size changed after reset"
markers = ['o', 'v', 's', 's', 's']
# plot results
for i, _ in enumerate(distributions):
    plt.plot(index_sizes[i], results_prec[i], label=plot_friendly_names[i], marker=markers[i])
plt.xlabel('Index Size (MB)')
plt.ylabel('Precision@1')
plt.legend()
plt.savefig('results/robustness_utilskew.png', bbox_inches='tight')
