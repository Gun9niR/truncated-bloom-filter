from boundedbloomfixedm import BoundedBlooms
import numpy as np

class EqualTruncation(BoundedBlooms):
    def __init__(self, bit_budget, m=250, maxk=250):
        super().__init__(bit_budget, m, maxk)
    
    def update_filter_lengths(self):
        original_size = sum([len(bf) for bf in self.bloom_filters])
        for i in range(len(self.bloom_filters)):
            m_i = int(np.floor(self.bit_budget/original_size*len(self.bloom_filters[i])))
            self.bloom_filters[i] = self.bloom_filters[i][:m_i]
    

class TopUtility(BoundedBlooms):
    def __init__(self, bit_budget, m=250, maxk=250):
        super().__init__(bit_budget, m, maxk)
    
    def update_filter_lengths(self):
        sorted_indexes = np.argsort(self.utilities)[::-1]
        running_budget = 0
        budget_reached = False
        for i in sorted_indexes:
            bf = self.bloom_filters[i]
            if running_budget + len(bf) > self.bit_budget:
                budget_reached = True
            
            if budget_reached:
                self.bloom_filters[i] = self.bloom_filters[i][:0]
            else:
                running_budget += len(bf)


class RandomUniform(BoundedBlooms):
    def __init__(self, bit_budget, m=250, maxk=250):
        super().__init__(bit_budget, m, maxk)
    
    def update_filter_lengths(self):
        # sample from discrete uniform distribution 
        # until budget is reached
        running_budget = 0
        sample = []
        N = len(self.bloom_filters)
        while True:
            i = np.random.randint(0, N)

            if i in sample:
                continue
            bf = self.bloom_filters[i]
            if running_budget + len(bf) > self.bit_budget:
                break
            sample.append(i)
            running_budget += len(bf)
        
        for i in list(set(range(N)) - set(sample)):
            self.bloom_filters[i] = self.bloom_filters[i][:0]

class RandomUtility(BoundedBlooms):
    def __init__(self, bit_budget, m=250, maxk=250):
        super().__init__(bit_budget, m, maxk)
    
    def update_filter_lengths(self):
        running_budget = 0
        sample = []
        N = len(self.bloom_filters)
        utility_prob = np.array(self.utilities)/np.sum(self.utilities)
        indexes = range(N)
        while True:
            i = np.random.choice(indexes, p=utility_prob)
            if i in sample:
                continue
            bf = self.bloom_filters[i]
            if running_budget + len(bf) > self.bit_budget:
                break
            sample.append(i)
            running_budget += len(bf)
        
        for i in list(set(range(N)) - set(sample)):
            self.bloom_filters[i] = self.bloom_filters[i][:0]


class UtilityProportional(BoundedBlooms):
    def __init__(self, bit_budget, m=250, maxk=250):
        super().__init__(bit_budget, m, maxk)
    
    def update_filter_lengths(self):
        raise NotImplementedError("UtilityProportional is not implemented yet")
