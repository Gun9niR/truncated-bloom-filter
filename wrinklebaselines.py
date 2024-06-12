from boundedbloomfast import BoundedBlooms
import numpy as np
import time
import math

class EqualTruncation(BoundedBlooms):
    def __init__(self, bit_budget, target_fpr=0.01, maxk=250):
        super().__init__(bit_budget, target_fpr, maxk)
    
    def update_filter_lengths(self):
        original_size = sum([len(bf) for bf in self.bloom_filters])
        for i in range(len(self.bloom_filters)):
            m_i = int(np.floor(self.bit_budget/original_size*len(self.bloom_filters[i])))
            self.bloom_filters[i] = self.bloom_filters[i][:m_i]
    

class TopUtility(BoundedBlooms):
    def __init__(self, bit_budget, target_fpr=0.01, maxk=250):
        super().__init__(bit_budget, target_fpr, maxk)
    
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
    def __init__(self, bit_budget, target_fpr=0.01, maxk=250):
        super().__init__(bit_budget, target_fpr, maxk)
    
    def update_filter_lengths(self):
        start = time.time()
        # sample from discrete uniform distribution 
        # until budget is reached
        running_budget = 0
        sample = []
        N = len(self.bloom_filters)
        in_sample = {i: False for i in range(N)}
        
        # initial_guess = int(self.bit_budget/np.percentile([len(bf) for bf in self.bloom_filters], 60))
        ratio = self.bit_budget/sum([len(bf) for bf in self.bloom_filters])
        initial_guess = math.floor(ratio*N)
        initial_sample = np.random.randint(0, N, size=initial_guess)
        

        correction_needed = True
        
        for i in initial_sample:
            bf = self.bloom_filters[i]
            if running_budget + len(bf) > self.bit_budget:
                correction_needed = False
                break
            sample.append(i)
            in_sample[i] = True
            running_budget += len(bf)
        if correction_needed:
            while True:
                i = np.random.randint(0, N)

                if in_sample[i]:
                    continue
                bf = self.bloom_filters[i]
                if running_budget + len(bf) > self.bit_budget:
                    break
                sample.append(i)
                in_sample[i] = True
                running_budget += len(bf)
                
        # print("Extra operations needed:", len(sample) - initial_guess)
        
        for i in list(set(range(N)) - set(sample)):
            self.bloom_filters[i] = self.bloom_filters[i][:0]
        
        # print("RandomUniform took {} seconds".format(time.time() - start))
        assert sum([len(bf) for bf in self.bloom_filters]) <= self.bit_budget

class RandomUtility(BoundedBlooms):
    def __init__(self, bit_budget, target_fpr=0.01, maxk=250):
        super().__init__(bit_budget, target_fpr, maxk)
    
    def update_filter_lengths(self):
        start = time.time()
        running_budget = 0
        sample = []
        N = len(self.bloom_filters)
        utility_prob = np.array(self.utilities)/np.sum(self.utilities)
        indexes = range(N)
        in_sample = {i: False for i in range(N)}
        
        ratio = self.bit_budget/sum([len(bf) for bf in self.bloom_filters])
        initial_guess = math.floor(ratio*N)
        
        # initial_guess = int(self.bit_budget/np.percentile([len(bf) for bf in self.bloom_filters], 60))
            
            
        
        initial_sample = np.random.choice(indexes, size=initial_guess, replace=False, p=utility_prob)
        correction_needed = True
        for i in initial_sample:
            bf = self.bloom_filters[i]
            if running_budget + len(bf) > self.bit_budget:
                correction_needed = False
                break
            sample.append(i)
            in_sample[i] = True
            running_budget += len(bf)
        
        if correction_needed:
            while True:
                i = np.random.choice(indexes, p=utility_prob)
                if in_sample[i]:
                    continue
                bf = self.bloom_filters[i]
                if running_budget + len(bf) > self.bit_budget:
                    break
                sample.append(i)
                in_sample[i] = True
                running_budget += len(bf)
        
        for i in list(set(range(N)) - set(sample)):
            self.bloom_filters[i] = self.bloom_filters[i][:0]
        
        assert sum([len(bf) for bf in self.bloom_filters]) <= self.bit_budget
        
        # print("Extra operations needed:", len(sample) - initial_guess)
        # print("RandomUtility took {} seconds".format(time.time() - start))
        


class UtilityProportional(BoundedBlooms):
    def __init__(self, bit_budget, target_fpr=0.01, maxk=250):
        super().__init__(bit_budget, target_fpr, maxk)
    
    def update_filter_lengths(self):
        raise NotImplementedError("UtilityProportional is not implemented yet")
