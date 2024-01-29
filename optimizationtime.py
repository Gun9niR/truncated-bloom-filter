from bloomfilteropt import *
import matplotlib.pyplot as plt
import time
plt.rcParams['figure.dpi'] = 300
plt.style.use("seaborn-v0_8-paper")

optimizer = ConvexJensen()

REPEAT = 5

# number of bloom filters
Ns = list(range(1000, 1000000, 50000))
# Ns = list(range(2000, 3000, 100))

means_std, means_linearized, means_asym = [], [], []
# lower_upper = []

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
        # time.sleep(0.5)
        start = time.time()
        m_opt = optimizer.optimize(B, p, m, n, k, equality_constraint=True, cval='standard')
        local.append(time.time()-start)

        # time.sleep(0.5)
        # start = time.time()
        # m_opt = optimizer.optimize(B, p, m, n, k, equality_constraint=True, cval='linearized')
        # local_linear.append(time.time()-start)

        # time.sleep(0.5)
        # start = time.time()
        # m_opt = optimizer.optimize(B, p, m, n, k, equality_constraint=True, cval='asymptotic')
        # local_asymptotic.append(time.time()-start)
        
    means_std.append(np.mean(local))
    # means_linearized.append(np.mean(local_linear))
    # means_asym.append(np.mean(local_asymptotic))
    # width = CONFID_VAL*np.std(local, ddof=1)/np.sqrt(REPEAT)
    # lower_upper.append((np.mean(local)-width, np.mean(local)+width))
    
plt.plot(Ns, means_std, color='blue', marker = 'o')
# plt.plot(Ns, means_linearized, color='red', marker = 'd', label='Linearized')
# plt.plot(Ns, means_asym, color='green', marker = 's', label='Asymptotic')
# plt.fill_between(Ns, [x[0] for x in lower_upper], [x[1] for x in lower_upper], color='blue', alpha=0.2)
plt.xlabel('Number of bloom filters (N)')
plt.ylabel('Optimization time (s)')
# plt.legend()
plt.savefig('resultsmicro/optimizationtime.png')
plt.clf()
    