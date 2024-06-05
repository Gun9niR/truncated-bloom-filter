from bloomfilteropt import *
import matplotlib.pyplot as plt
import time
plt.rcParams['figure.dpi'] = 300
plt.style.use("seaborn-v0_8-paper")

params = {'legend.fontsize': 'medium',
            'axes.labelsize': 'medium',
            'axes.titlesize':'medium',
            'xtick.labelsize':'medium',
            'ytick.labelsize':'medium',
            'axes.labelweight': 'bold'}
plt.rcParams.update(params)

optimizer = ConvexJensen()

REPEAT = 10

Ns = list(range(1000, 1000000, 50000))

means_std, means_linearized, means_asym = [], [], []

for N in Ns:
    print(N)
    m = np.round(np.random.normal(10000, 1000, N))
    n = np.round(np.random.uniform(1000, 2000, N))
    k = np.round((m/n)*np.log(2))
    p = np.random.normal(100, 15, N)
    
    B = int(0.5*np.sum(m))
    local = []
    local_linear = []
    local_asymptotic = []
    for _ in range(REPEAT):
        start = time.time()
        m_opt = optimizer.optimize(B, p, m, n, k, equality_constraint=True, cval='standard')
        local.append(time.time()-start)

        
    means_std.append(np.median(local))

    
plt.plot(Ns, means_std, color='blue', marker = 'o')
plt.xlabel('Number of bloom filters (N)')
plt.ylabel('Optimization time (s)')
plt.savefig('REVISION_RESULTS/optimizationtime.png', bbox_inches='tight')
plt.clf()

with open('REVISION_RESULTS/optimizationtime.txt', 'w') as f:
    for i in range(len(Ns)):
        f.write(str(Ns[i]) + ' ' + str(means_std[i]) + '\n')
    