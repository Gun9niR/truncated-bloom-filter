from bloomfilteropt import ConvexJensen, FloorRounding, GreedyPairing
import copy
import numpy as np
from bitarray import bitarray
import mmh3
from utils import Preprocessor


class BoundedBlooms:
    def __init__(self, bit_budget, target_fpr=0.01, maxk=250):
        self.bit_budget = bit_budget
        self.bloom_filters = []
        self.bloom_filter_backups = []
        self.utilities = []
        self.ms = []
        self.ns = []
        self.ks = []
        self.target_fpr = target_fpr
        self.maxk = maxk
        self.generate_seeds()

    def compute_optimal_params(self, n):
        m = int(np.ceil(-(n*np.log(self.target_fpr))/(np.log(2)**2)))
        self.ms.append(m)
        self.ks.append(int(np.ceil((m/n)*np.log(2))))
        if self.ks[-1] > self.maxk:
            raise ValueError('k is too large. Increase maxk.')
    
    def generate_seeds(self):
        self.shared_seeds = np.random.choice(range(2**16), self.maxk, replace=False)

    def process_and_add(self, corpus, utilities, fpath):
        word_lists, self.docstore = Preprocessor().preprocess_and_store(corpus, fpath)
        self.add_all(word_lists, utilities)
    
    def new_docstore(self, processed_corpus, fpath):
        self.docstore = Preprocessor().create_new_docstore(fpath, processed_corpus)
    
    def set_utilities(self, utilities):
        self.utilities = utilities

    def add_all(self, word_lists, utilities):
        for l, u in zip(word_lists, utilities):
            n = len(l)
            self.compute_optimal_params(n)
            m = self.ms[-1]
            k = self.ks[-1]
            self.ns.append(n)
            self.utilities.append(u)
            b = bitarray(m)
            b.setall(0)
            for word in l:
                for i in range(k):
                    b[mmh3.hash(word, self.shared_seeds[i]) % m] = True
            self.bloom_filters.append(b)
            self.bloom_filter_backups.append(copy.copy(b))
        self.max_m = max(self.ms)
        self.max_k = max(self.ks)
        self.shared_seeds = self.shared_seeds[:self.max_k]

    
    def query(self, word_list, topk=5, disk=True):
        n = len(word_list)
        base_hashes = [[] for _ in range(n)]
        for i, word in enumerate(word_list):
            for seed in self.shared_seeds:
                base_hashes[i].append(mmh3.hash(word, seed))
        query_filter = bitarray(self.max_m)
        query_filter.setall(False)
        result = []
        traversal_order = np.argsort(self.utilities)[::-1]
        # dacess = 0
        # retention = 0.0
        # number of disk accesses
        members = bitarray(n)
        match = bitarray(n)
        match.setall(True)
        for ix, (bf, m, k, _) in enumerate(sorted(zip(self.bloom_filters, self.ms, self.ks, self.utilities), key=lambda x: x[3], reverse=True)):

            members.setall(False)
            mtrunc = len(bf)
            # degenerate case 1: 0 length filter -> none are members
            if mtrunc > 0:
                for i in range(n):
                    query_filter.setall(False)
                    hashes = [base_hashes[i][j] % m for j in range(k)]
                    valid_hashes = [h for h in hashes if h < mtrunc]
                    # degenerate case 2: 0 valid hashes -> member
                    if len(valid_hashes) == 0:
                        members[i] = True
                        continue
                    for h in valid_hashes:
                        query_filter[h] = True
                    if (query_filter[:mtrunc] & bf).count(True) == len(set(valid_hashes)):
                        members[i] = True
                    else:
                        break
                    # query_filter.setall(False)
            # if members.count(True) == n:
            if members & match == match:
                j = traversal_order[ix]
                # retention += len(bf)/m
                if disk:
                    # dacess += 1
                    if len(set(self.docstore.get(j).split(',')).intersection(set(word_list))) == n:
                        result.append(j)
                else:
                    result.append(j)
            if len(result) == topk:
                # print("Went through:{} percent".format(ix/len(self.bloom_filters)))
                # print("Percent disk accesses:", dacess/(ix+1))
                # print("Average retention:", retention/len(result))
                break
        # return result, dacess/(ix+1)
        return result

    def reset(self):
        for i in range(len(self.bloom_filters)):
            self.bloom_filters[i] = copy.copy(self.bloom_filter_backups[i])
        
    def update_budget(self, new_budget):
        self.bit_budget = new_budget

    def update_filter_lengths(self, optimizer_type, rounding_scheme, equality_constraint=True, cval='standard'):
        N = len(self.bloom_filters)
        params = np.zeros((N, 3))

        params[:, 0] = self.ms
        params[:, 1] = self.ns
        params[:, 2] = self.ks

        utilities = np.array(self.utilities)
        
        if optimizer_type == 'jensen':
            opt = ConvexJensen().optimize(self.bit_budget, utilities, params[:, 0],
                                          params[:, 1], params[:, 2], equality_constraint, cval)
        if opt is None:
            raise ValueError('Optimization failed. No feasible solution exists.')
        
        if rounding_scheme == 'floor':
            opt = FloorRounding().round(opt)
        elif rounding_scheme == 'greedy':
            opt = GreedyPairing().round(opt, utilities, params[:, 0])
        else:
            raise ValueError('rounding_scheme must be "floor" or "greedy"')
        for i in range(N):
            self.bloom_filters[i] = self.bloom_filters[i][:opt[i]]
        
        n_cached_docs = sum([1 for bf in self.bloom_filters if len(bf) > 0])
        self.cache_doc_pct = n_cached_docs/len(self.bloom_filters)
    
    def generate_split(self, optimizer_type, rounding_scheme, equality_constraint=True, cval='standard'):
        N = len(self.bloom_filters)
        self.second_halves = []
        params = np.zeros((N, 3))

        params[:, 0] = self.ms
        params[:, 1] = self.ns
        params[:, 2] = self.ks

        utilities = np.array(self.utilities)
        
        if optimizer_type == 'jensen':
            opt = ConvexJensen().optimize(self.bit_budget, utilities, params[:, 0],
                                          params[:, 1], params[:, 2], equality_constraint, cval)
        if opt is None:
            raise ValueError('Optimization failed. No feasible solution exists.')
        
        if rounding_scheme == 'floor':
            opt = FloorRounding().round(opt)
        elif rounding_scheme == 'greedy':
            opt = GreedyPairing().round(opt, utilities, params[:, 0])
        else:
            raise ValueError('rounding_scheme must be "floor" or "greedy"')

        self.ms = [len(bf) for bf in self.bloom_filters]
        
        temp = copy.copy(self.bloom_filters)
        
        
        for i in range(N):
            # print('og:', temp[i])
            # print('L:', self.bloom_filters[i][:opt[i]])
            # print('R:', self.bloom_filters[i][opt[i]:])
            self.second_halves.append(self.bloom_filters[i][opt[i]:])
            self.bloom_filters[i] = self.bloom_filters[i][:opt[i]]
        
        self.length_ratios = [len(bf)/m for bf, m in zip(self.second_halves, self.ms)]
        
        n_cached_docs = sum([1 for bf in self.bloom_filters if len(bf) > 0])
        self.cache_doc_pct = n_cached_docs/len(self.bloom_filters)
    
    def index_size(self):
        return sum([len(bf) for bf in self.bloom_filters])
    
    def cached_doc_pct(self):
        return self.cached_doc_pct
