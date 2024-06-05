from bloomfilter import BloomFilter, TruncatedBloomFilter
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
plt.style.use("seaborn-v0_8-paper")
plt.rcParams['figure.dpi'] = 300

params = {'legend.fontsize': 'large',
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large',
         'axes.labelweight': 'bold'}

plt.rcParams.update(params)



# generate random strings
N = 1000000
# N = 20000
n_chars = 20
chars = list('abcdefghijklmnopqrstuvwxyz0123456789')

string_keys = {}
for _ in range(N):
    string_keys[''.join(np.random.choice(chars, n_chars))] = True

not_in_keys = []
for _ in range(N):
    key = ''.join(np.random.choice(chars, n_chars))
    if key not in string_keys:
        not_in_keys.append(key)
        
string_keys = list(string_keys.keys())
all_keys = string_keys + not_in_keys

FPR = 0.0001
bf = BloomFilter(n=N, target_fpr=FPR)
tbf = TruncatedBloomFilter(target_fpr=FPR, n=N, precompute_binomial_coefs=True)

for key in string_keys:
    tbf.add(key)
    bf.add(key)

tbf_filter_copy = tbf.filter.copy()

time.sleep(2.5)
local = []
for key in not_in_keys:
    start = time.time()
    bf.query(key)
    end = time.time()
    local.append(end-start)
bf_avg_latency_neg = np.mean(local)
bf_latency_ci_width_neg = 1.96*np.std(local, ddof=1)/np.sqrt(len(local))

local = []
for key in string_keys:
    start = time.time()
    bf.query(key)
    end = time.time()
    local.append(end-start)
bf_avg_latency_pos = np.mean(local)
# bf_avg_latency_pos = np.median(local)
bf_latency_ci_width_pos = 1.96*np.std(local, ddof=1)/np.sqrt(len(local))


ratios = np.arange(0.05, 1, 0.05)
exp_fpr = []
lb_exp_fpr = []
proposed_fpr = []
emp_fpr = []
tbf_query_latencies = []
tbf_query_latencies_neg = []

for ratio in ratios:
    m_t = int(np.floor(ratio*tbf.m))
    
    tbf.filter = tbf_filter_copy
    tbf.truncate(m_t)

    fps = 0.0
    for key in not_in_keys:
        if tbf.query(key):
            fps += 1
    
    time.sleep(2.5)
    local = []
    for key in string_keys:
        start = time.time()
        tbf.query(key)
        end = time.time()
        local.append(end-start)
    avg = np.mean(local)
    width = 1.96*np.std(local, ddof=1)/np.sqrt(len(local))
    tbf_query_latencies.append((avg, avg-width, avg+width))
    
    time.sleep(2.5)
    local = []
    for key in not_in_keys:
        start = time.time()
        tbf.query(key)
        end = time.time()
        local.append(end-start)
    avg = np.mean(local)
    width = 1.96*np.std(local, ddof=1)/np.sqrt(len(local))
    tbf_query_latencies_neg.append((avg, avg-width, avg+width))

    emp_fpr.append(fps/len(not_in_keys))
    exp_fpr.append(tbf.truncated_false_positive_rate(m_t))
    lb_exp_fpr.append(tbf.truncated_lower_bound_false_positive_rate(m_t))
    proposed_fpr.append(tbf.proposed_conditional(m_t))
    
fprs = [0.1, 0.01, 0.001, 0.0001, 1e-6]
markers = ['o', 's', 'd', '*', '^']



for i, fpr in enumerate(fprs):
    tbf = TruncatedBloomFilter(target_fpr=fpr, n=N, precompute_binomial_coefs=True)
    for key in string_keys:
        tbf.add(key)
    tbf_filter_copy = tbf.filter.copy()
    tbf_fprs = []
    lbs_other = []
    for ratio in ratios:
        m_t = int(np.floor(ratio*tbf.m))
        
        tbf.filter = tbf_filter_copy
        tbf.truncate(m_t)
        lbs_other.append(tbf.truncated_lower_bound_false_positive_rate(m_t))
    
        fps = 0
        for key in not_in_keys:
            if tbf.query(key):
                fps += 1
        tbf_fprs.append(fps/len(not_in_keys))
    plt.plot(ratios, tbf_fprs, label='Construction FPR={}'.format(fpr), marker=markers[i])
    # get color of line
    color = plt.gca().lines[-1].get_color()
    plt.plot(ratios, lbs_other, label='Construction FPR={} LB'.format(fpr), linestyle='--', color=color)
    # shade between the two lines
    plt.fill_between(ratios, tbf_fprs, lbs_other, alpha=0.15, color=color)
plt.xlabel('Truncation ratio (p)')
plt.ylabel('False positive rate')
plt.legend()
plt.savefig('REVISION_RESULTS/truncatedfpr_multiconstructionfprs.png', bbox_inches='tight')
plt.clf()

# params = {
#             'legend.fontsize': 'medium',
#             'axes.labelsize': 'medium',
#             'axes.titlesize':'small',
#             'xtick.labelsize':'small',
#             'ytick.labelsize':'small'}
# plt.rcParams.update(params) 

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(ratios, emp_fpr, color='blue', marker='o', label='Empirical')
ax[0].plot(ratios, exp_fpr, color='red', marker='d', label='Expected')
ax[0].plot(ratios, lb_exp_fpr, color='green', marker='s', label='Lower bound')
# proposed (conditional) fpr seems wrong
# ax[0].plot(ratios, proposed_fpr, color='orange', marker='^', label='Proposed')
ax[0].set_xlabel('Truncation ratio (p)')
ax[0].set_ylabel('False positive rate')
ax[0].legend()
# plt.savefig('resultsmicro/truncatedfpr.png', bbox_inches='tight')
# plt.clf()

# convert latencies to microseconds
bf_avg_latency_pos *= 1e6
bf_latency_ci_width_pos *= 1e6
tbf_query_latencies = [(x*1e6, y*1e6, z*1e6) for (x, y, z) in tbf_query_latencies]

bf_avg_latency_neg *= 1e6
bf_latency_ci_width_neg *= 1e6
tbf_query_latencies_neg = [(x*1e6, y*1e6, z*1e6) for (x, y, z) in tbf_query_latencies_neg]


ax[1].plot(ratios, [x[0] for x in tbf_query_latencies], color='orange', marker='*', label='Truncated (+)')
ax[1].plot(ratios, [x[0] for x in tbf_query_latencies_neg], color='purple', marker='*', label='Truncated (-)')
# fill between using confience intervals
ax[1].fill_between(ratios, [x[1] for x in tbf_query_latencies], [x[2] for x in tbf_query_latencies], alpha=0.2, color='orange')
ax[1].fill_between(ratios, [x[1] for x in tbf_query_latencies_neg], [x[2] for x in tbf_query_latencies_neg], alpha=0.2, color='purple')
ax[1].scatter([1.0], [bf_avg_latency_pos], color='orange', marker='o', label='Standard (+)')
ax[1].errorbar([1.0], [bf_avg_latency_pos], yerr=bf_latency_ci_width_pos, color='orange', alpha=0.4)
ax[1].scatter([1.0], [bf_avg_latency_neg], color='purple', marker='o', label='Standard (-)')
ax[1].errorbar([1.0], [bf_avg_latency_neg], yerr=bf_latency_ci_width_neg, color='purple', alpha=0.4)
ax[1].set_xlabel('Truncation ratio (p)')
ax[1].set_ylabel('Query latency (μs)')
ax[1].legend()
fig.savefig('REVISION_RESULTS/truncatedquerylatency_fpr_new.png', bbox_inches='tight')

