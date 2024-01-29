import mmh3
import bitarray
import numpy as np
from scipy.special import comb

class BloomFilter:
    def __init__(self, m=None, k=None, target_fpr=0.01, n=None, defer_seeds=False):
        if m is None or k is None:
            if n is None:
                raise ValueError('n must be set if either m or k are not')
            self.n = n
            self.compute_optimal_params(target_fpr)
            
        else:
            self.m = m
            self.k = k
            self.n = 0
        if not defer_seeds:
            self.generate_seeds()
        self.filter = bitarray.bitarray(self.m)
        self.filter.setall(0)
    
    def generate_seeds(self, k=None):
        if k is None:
            self.seeds = np.random.choice(range(2**16), self.k, replace=False)
        else:
            self.seeds = np.random.choice(range(2**16), k, replace=False)
    
    def set_seeds(self, seeds):
        self.seeds = seeds[:self.k]

    def compute_optimal_params(self, target_fpr):
        self.m = int(np.ceil(-(self.n*np.log(target_fpr))/(np.log(2)**2)))
        self.k = int(np.ceil((self.m/self.n)*np.log(2)))

    def add(self, key):
        for seed in self.seeds:
            index = mmh3.hash(key, seed) % self.m
            self.filter[index] = 1
    
    def query(self, key):
        for seed in self.seeds:
            index = mmh3.hash(key, seed) % self.m
            if self.filter[index] == 0:
                return False
        return True
   

    def false_positive_rate(self):
        return (1- (1 - 1 / self.m) ** (self.k * self.n)) ** self.k

class TruncatedBloomFilter(BloomFilter):
    def __init__(self, m=None, k=None, target_fpr=0.01, n=None, precompute_binomial_coefs=False, defer_seeds=False):
        super().__init__(m, k, target_fpr, n, defer_seeds)
        if precompute_binomial_coefs:
            self.precompute_binomial_coefs()

    def precompute_binomial_coefs(self):
        binom_coefs = np.zeros((self.m+1))
        for x in range(self.k+1):
            binom_coefs[x] = comb(self.k, x, exact=True)
        self.binom_coefs = binom_coefs
    
   
    def add(self, key):
        for seed in self.seeds:
            index = mmh3.hash(key, seed) % self.m
            if index < len(self.filter):
                self.filter[index] = 1
    
    def query(self, key):
        for seed in self.seeds:
            index = mmh3.hash(key, seed) % self.m
            if index < len(self.filter) and self.filter[index] == 0:
                return False
        return True

     # this is buggy
    def query_fast_multiple(self, positions: np.array):
        positions = np.mod(positions[:, :self.k], self.m)
        n = positions.shape[0]
        positions[np.where((positions >= len(self.filter)))] = -1

        for i in range(n):
            for j in range(self.k):
                f_index = positions[i, j]

                if f_index == -1 or self.filter[f_index] == 1:
                    positions[i, j] = 1
                else:
                    positions[i, j] = 0
        return np.all(positions, axis=1)
            


    def truncate(self, m_t):
        if m_t > self.m:
            raise ValueError('m_t must be smaller than m. It is currently {} which is larger than {}'.format(m_t, self.m))
        self.filter = self.filter[:m_t]
    
    def truncated_false_positive_rate(self, m_t):
        if not hasattr(self, 'binom_coefs'):
            self.precompute_binomial_coefs()
        expected_value = 0.0
        for x in range(self.k+1):
            expected_value += self.binom_coefs[x]*((m_t/self.m)**x)*((1-(m_t/self.m))**(self.k-x))*((1-(1-(1/self.m))**(self.k*self.n))**x)
        return expected_value
    
    def truncated_lower_bound_false_positive_rate(self, m_t):
        return (1-(1-1/self.m)**(self.k*self.n))**(self.k*m_t/self.m)

    def proposed_conditional(self, m_t):
        # c = self.m - m_t
        return (self.m-m_t)*(1-(1-1/self.m)**(self.k*self.n))*((m_t/self.m)**self.k)