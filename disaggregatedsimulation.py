import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.style.use("seaborn-v0_8-paper")
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
plt.rcParams.update(params)
import json

with open('relationalresults/{}.json'.format('NASA'), 'r') as fp:
    results = json.load(fp)

DISAGG_CONSTS = [5, 10]

# plot with broken axis

translucency_levels = [0.5, 1.0]

for i, dac in enumerate(DISAGG_CONSTS):
    
    if i == 0:
        plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time']*dac*1000 for d in results['Bloom']], marker='*', alpha=translucency_levels[i])
        bloom_color = plt.gca().lines[-1].get_color()

        # plt.scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Wasted time']*dac], marker='x', color='red', alpha=translucency_levels[i])
        # range_color = plt.gca().lines[-1].get_color()
        # plt.scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Wasted time']*dac], marker='x',  color='purple', alpha=translucency_levels[i])
        disk_color = plt.gca().lines[-1].get_color()
        plt.plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Wasted time']*dac*1000 for d in results['Equal Truncation']], marker='^', alpha=translucency_levels[i])
        equal_color = plt.gca().lines[-1].get_color()
        plt.plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Wasted time']*dac*1000 for d in results['Top Utility']], marker='o', alpha=translucency_levels[i])
        top_color = plt.gca().lines[-1].get_color()
        
    elif i < len(DISAGG_CONSTS) - 1:
        plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time']*dac*1000 for d in results['Bloom']], marker='*', color=bloom_color, alpha=translucency_levels[i])
        plt.plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Wasted time']*dac*1000 for d in results['Equal Truncation']], marker='^', color=equal_color, alpha=translucency_levels[i])
        plt.plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Wasted time']*dac*1000 for d in results['Top Utility']], marker='o', color=top_color, alpha=translucency_levels[i])
        # plt.scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Wasted time']*dac], marker='x', color="red", alpha=translucency_levels[i])
        # plt.scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Wasted time']*dac], marker='x', color="purple", alpha=translucency_levels[i])
    else:
        plt.plot([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time']*dac*1000 for d in results['Bloom']], marker='*', color=bloom_color, alpha=translucency_levels[i], label='Bounded Bloom (us)')
        plt.plot([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Wasted time']*dac*1000 for d in results['Equal Truncation']], marker='^', color=equal_color, alpha=translucency_levels[i], label='Equal Truncation')
        plt.plot([d['Index size']/8e6 for d in results['Top Utility']], [d['Wasted time']*dac*1000 for d in results['Top Utility']], marker='o', color=top_color, alpha=translucency_levels[i], label='Top Utility')
        # plt.scatter([results['Range'][0]['Index size']/8e6], [results['Range'][0]['Wasted time']*dac], marker='x', color="red", alpha=translucency_levels[i], label='Range')
        # plt.scatter([results['Disk'][0]['Index size']/8e6], [results['Disk'][0]['Wasted time']*dac], marker='x', color="purple", alpha=translucency_levels[i], label='Disk')

# fill between
plt.fill_between([d['Index size']/8e6 for d in results['Bloom']], [d['Wasted time']*DISAGG_CONSTS[0]*1000 for d in results['Bloom']], [d['Wasted time']*DISAGG_CONSTS[1]*1000 for d in results['Bloom']], color=bloom_color, alpha=0.1)
plt.fill_between([d['Index size']/8e6 for d in results['Equal Truncation']], [d['Wasted time']*DISAGG_CONSTS[0]*1000 for d in results['Equal Truncation']], [d['Wasted time']*DISAGG_CONSTS[1]*1000 for d in results['Equal Truncation']], color=equal_color, alpha=0.1)
plt.fill_between([d['Index size']/8e6 for d in results['Top Utility']], [d['Wasted time']*DISAGG_CONSTS[0]*1000 for d in results['Top Utility']], [d['Wasted time']*DISAGG_CONSTS[1]*1000 for d in results['Top Utility']], color=top_color, alpha=0.1)

    
plt.xlabel("Index size (MB)")
plt.ylabel("Wasted time (ms)")
plt.legend()
    
plt.savefig('relationalresults/{}_disaggregation.png'.format('NASA'), bbox_inches='tight')
    
