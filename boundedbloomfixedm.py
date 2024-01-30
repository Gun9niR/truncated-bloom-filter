from bloomfilteropt import ConvexJensen, FloorRounding, GreedyPairing
import copy
import numpy as np
from bitarray import bitarray
import mmh3
from utils import Preprocessor


class BoundedBlooms:
    def __init__(self, bit_budget, m=1000, maxk=250):
        self.bit_budget = bit_budget
        self.bloom_filters = []
        self.bloom_filter_backups = []
        self.utilities = []
        self.m = int(m)
        self.maxk = maxk
        self.generate_seeds()

    def compute_optimal_params(self, ns):
        self.ns = ns
        self.ms = [self.m for _ in range(len(ns))]
        self.ks = [int(np.ceil((self.m/ns[i])*np.log(2))) for i in range(len(ns))]

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
        self.compute_optimal_params([len(l) for l in word_lists])
        m = self.ms[0]
        for i, (l, u) in enumerate(zip(word_lists, utilities)):
            self.utilities.append(u)
            b = bitarray(m)
            b.setall(0)
            for word in l:
                for j in range(self.ks[i]):
                    b[mmh3.hash(word, self.shared_seeds[j]) % m] = True
            self.bloom_filters.append(b)
            self.bloom_filter_backups.append(copy.copy(b))
        self.max_m = m
        self.max_k = max(self.ks)
        self.shared_seeds = self.shared_seeds

    
    def query(self, word_list, topk=5, disk=True):
        n = len(word_list)
        query_filters = {}
        for k in list(set(self.ks)):
            query_filter = bitarray(self.max_m)
            query_filter.setall(False)

            for word in word_list:
                # for i in range(self.max_k):
                for i in range(k):
                    query_filter[mmh3.hash(word, self.shared_seeds[i]) % self.max_m] = True
            query_filters[k] = query_filter
        
        result = []
        traversal_order = np.argsort(self.utilities)[::-1]

        for ix, (bf, _, k) in enumerate(sorted(zip(self.bloom_filters, self.utilities, self.ks), key=lambda x: x[1], reverse=True)):
            mtrunc = len(bf)
            # degenerate case 1: 0 length filter -> none are members
            if mtrunc > 0:
                qft = query_filters[k][:mtrunc]
                if qft & bf == qft:
                    j = traversal_order[ix]
                    if disk:
                        if len(set(self.docstore.get(j).split(',')).intersection(set(word_list))) == n:
                            result.append(j)
                    else:
                        result.append(j)
            if len(result) == topk:
                break
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
    
    def index_size(self):
        return sum([len(bf) for bf in self.bloom_filters])
