from boundedbloomfast import BoundedBlooms
import mmh3
import math
import json
import numpy as np
from abc import abstractmethod
import bitarray
import pyarrow.parquet as pq
import time
from abc import abstractmethod
from typing import List
import os


FLOAT_SIZE = 32
BITS_PER_BYTE = 8
        
class ColumnBloom(BoundedBlooms):
    def __init__(self, bit_budget, target_fpr=0.01, maxk=250, column_name="sample", row_group_size=1000):
        super().__init__(bit_budget, target_fpr, maxk)
        self.column_name = column_name
        self.row_group_size = row_group_size
    
    def query(self, key, row_group_id):
        bf = self.bloom_filters[row_group_id]
        mprime = len(bf)
        if mprime == 0:
            return True
        for i in range(self.ks[row_group_id]):
            pos = mmh3.hash(key, self.shared_seeds[i]) % self.ms[row_group_id]
            if pos < mprime and bf[pos] == False:
                return False
        return True

class DocBloom(BoundedBlooms):
    def __init__(self, bit_budget, target_fpr=0.01, maxk=250, column_name="sample", row_group_size=1000):
        super().__init__(bit_budget, target_fpr, maxk)
        self.column_name = column_name
        self.row_group_size = row_group_size
    
    def query(self, keywords, row_group_id):
        start = row_group_id*self.row_group_size
        end = min(start + self.row_group_size, len(self.bloom_filters))
        
        n = len(keywords)
        base_hashes = [[] for _ in range(n)]
        for i, word in enumerate(keywords):
            for seed in self.shared_seeds:
                base_hashes[i].append(mmh3.hash(word, seed))
        query_filter = bitarray(self.max_m)
        query_filter.setall(False)
        members = bitarray(n)
        match = bitarray(n)
        match.setall(True)
        for bf, m, k in zip(self.bloom_filters[start:end], self.ms[start:end], self.ks[start:end]):
            members.setall(False)
            mtrunc = len(bf)
            # degenerate case 1: 0 length filter -> automatically search
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
            else:
                return True
            if members & match == match:
                return True
        return False
    
class ColumnPredicate:
    def __init__(self, column_name, column_dtype, value):
        self.column_name = column_name
        self.column_dtype = column_dtype
        self.value = value
    
    @abstractmethod
    def __call__(self, filter):
        raise NotImplementedError
    
class RangePredicate(ColumnPredicate):
    def __init__(self, column_name, column_dtype, value):
        super().__init__(column_name, column_dtype, value)
        if type(value) == float or type(value) == int or type(value) == str:
            self.scalar = True
        else:
            self.scalar = False
        
    
    def __call__(self, rnge_filter):
        if self.scalar:
            return self.value >= rnge_filter[0] and self.value <= rnge_filter[1]
        return self.value[0] >= rnge_filter[0] and self.value[1] <= rnge_filter[1]
  
class BloomPredicate(ColumnPredicate):
    def __init__(self, column_name, column_dtype, value):
        super().__init__(column_name, column_dtype, value)
    
    def __call__(self, column_bloom, row_group_id):
        return column_bloom.query(self.value, row_group_id)
    
class DocPredicate(ColumnPredicate):
    def __init__(self, column_name, column_dtype, value):
        super().__init__(column_name, column_dtype, value)
    
    def __call__(self, doc_bloom, row_group_id):
        return doc_bloom.query(self.value, row_group_id)

class Query:
    def __init__(self, predicates: List[ColumnPredicate]):
        self.predicates = predicates

class ColumnIndexes:
    def __init__(self, column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size=1000):
        self.column_names = column_names
        self.column_dtypes = column_dtypes
        self.ngroups = ngroups
        self.row_utilities = row_utilities
        self.parquet_file = pq.ParquetFile(parquet_file)
        self.row_group_size = row_group_size
        self.column_indexes = {}
        self.exp_stats = {'Skip rate': [], 'Query latencies': [], 'Wasted time': []}
    
    @abstractmethod
    def construct_indexes(self):
        pass
    
    def average_stats(self):
        if len(self.exp_stats['Skip rate']) == 0:
            raise ValueError("No queries have been executed yet")
        return {name: sum(stat)/len(stat) for name, stat in self.exp_stats.items()}
    
    def range_stats(self):
        if len(self.exp_stats['Skip rate']) == 0:
            raise ValueError("No queries have been executed yet")
        return {name: [min(stat), max(stat)] for name, stat in self.exp_stats.items()}
    
    def calculate_row_group_utilities(self):
        # print('row utils:', np.isnan(self.row_utilities).any())
        self.row_group_utilities = np.zeros(self.ngroups)
        for i in range(self.ngroups):
            # self.row_group_utilities[i] = np.mean(self.row_utilities[i*self.ngroups:min((i+1)*self.ngroups, len(self.row_utilities))])
            self.row_group_utilities[i] = np.mean(self.row_utilities[i*self.row_group_size:(i+1)*self.row_group_size])
    
    def build_bloom_index(self, column_id, group_keys, fpr, compression_ratio, optim=True, trunc=True, topm=False):
        if len(group_keys) != self.ngroups:
            raise ValueError('Number of groups does not match number of keys')
        if self.column_dtypes[column_id] != str:
            raise ValueError('Column must be of type str. Cast to str before building index')
        bb = ColumnBloom(1e8, fpr, column_name=self.column_names[column_id], row_group_size=self.row_group_size)
        
        # calculate row group utilities
        self.calculate_row_group_utilities()
        bb.add_all(group_keys, self.row_group_utilities)
        
        if trunc:
            if optim:
                size = bb.index_size()
                bb.update_budget(math.floor(size*compression_ratio))
                bb.update_filter_lengths(optimizer_type='jensen', rounding_scheme='floor',
                                    cval='standard', equality_constraint=True)
            elif not optim and topm:
                size = bb.index_size()
                budget = math.floor(size*compression_ratio)
                running_budget = 0
                for j in range(len(bb.bloom_filters)):
                    cost = len(bb.bloom_filters[j])
                    if cost + running_budget > budget:
                        for l in range(j, len(bb.bloom_filters)):
                            bb.bloom_filters[l] = bb.bloom_filters[j][:0]
                        break
                    running_budget += cost
            else:
                for j in range(len(bb.bloom_filters)):
                    size = math.floor(compression_ratio*len(bb.bloom_filters[j]))   
                    bb.bloom_filters[j] = bb.bloom_filters[j][:size]
        self.column_indexes[self.column_names[column_id]] = bb
    
    # def read_times_range_indexes(self):
    #     self.read_times = {}
    #     for index, name in zip(self.column_indexes, self.column_names):
    #         if type(index) != dict:
    #             continue
            
    #         with open("test.json", 'w') as f:
    #             json.dump(index, f)

    #         start = time.time()
    #         with open("test.json", 'r') as f:
    #             json.load(f)
    #         end = time.time()
    #         # estimate read time per row group
    #         index_read_time = (end-start)/self.ngroups
    #         self.read_times[name] = index_read_time
    
    def write_to_disk(self):
        for name, index in self.column_indexes.items():
            if type(index) == dict:
                if type(index[0][0]) != str:
                    index = {k: [int(i) for i in v] for k, v in index.items()}
                with open("{}.json".format(name), 'w') as f:
                    json.dump(index, f)
            elif type(index) == ColumnBloom:
                for rg, bf in enumerate(index.bloom_filters):
                    with open ("{}_{}.bin".format(name, rg), 'wb') as f:
                        bf.tofile(f)
    
    def read_from_disk(self):
        indexes = []
        for name, index in self.column_indexes.items():
            if type(index) == dict:
                with open("{}.json".format(name), 'r') as f:
                    index = json.load(f)
                    indexes.append(index)
            elif type(index) == ColumnBloom:
                bfs = []
                for rg, bf in enumerate(index.bloom_filters):
                    a = bitarray.bitarray()
                    with open ("{}_{}.bin".format(name, rg), 'rb') as f:
                        a.fromfile(f)
                        bfs.append(a)
                indexes.append(bfs)
        
    def cleanup_disk(self):
        for name, index in self.column_indexes.items():
            if type(index) == dict:
                os.remove("{}.json".format(name))
            elif type(index) == ColumnBloom:
                for rg in range(len(index.bloom_filters)):
                    os.remove("{}_{}.bin".format(name, rg))
                

    def build_alpha_range_index(self, column_id, group_keys):
        if len(group_keys) != self.ngroups:
            raise ValueError('Number of groups does not match number of keys')
        if self.column_dtypes[column_id] != str:
            raise ValueError('Column must be of type str. Cast to str before building index')
        self.column_indexes[self.column_names[column_id]] = {}
        
        for i, keys in enumerate(group_keys):
            keys.sort()
            rnge = (keys[0], keys[-1])
            self.column_indexes[self.column_names[column_id]][i] = rnge
    
    def build_range_index(self, column_id, group_keys):
        if len(group_keys) != self.ngroups:
            raise ValueError('Number of groups does not match number of keys')
        if self.column_dtypes[column_id] != float and self.column_dtypes[column_id] != int:
            raise ValueError('Column must be of type float or int')
        
        self.column_indexes[self.column_names[column_id]] = {}
        
        for i, keys in enumerate(group_keys):
            rnge = (np.min(keys), np.max(keys))
            self.column_indexes[self.column_names[column_id]][i] = rnge
        
        
    def build_doc_index(self, column_id, doc_word_lists, fpr, compression_ratio):
        if self.column_dtypes[column_id] != str:
            raise ValueError('Column must be of type str.')

        bb = BoundedBlooms(1e10, fpr)
        bb.add_all(doc_word_lists, self.row_utilities)
        bb.new_docstore(doc_word_lists, "{}.npy".format(self.column_names[column_id]))
        size = bb.index_size()
        bb.update_budget(math.floor(size*compression_ratio))
        bb.update_filter_lengths(optimizer_type='jensen', rounding_scheme='floor',
                                 cval='standard', equality_constraint=True)
        self.column_indexes[self.column_names[column_id]] = bb

    def index_size(self):
        size = 0
        for _, index in self.column_indexes.items():
            if type(index) == ColumnBloom or type(index) == DocBloom:
                size += index.index_size()
            elif type(index) == dict and type(index[0][0]) == str:
                for _, rnge in index.items():
                    low, high = rnge
                    size += (len(low)+len(high))*BITS_PER_BYTE
            else:  
                size += 2*FLOAT_SIZE
        return size
    # currently restricted to conjunctive queries            
    def query_rowgroup(self, predicates, row_group_id):
        eval_result = []
        for p in predicates:
            index = self.column_indexes[p.column_name]
            if type(index) == ColumnBloom:
                eval_result.append(p(index, row_group_id))
            elif type(index) == DocBloom:
                eval_result.append(p(index, row_group_id))
            elif type(index) == dict:
                eval_result.append(p(index[row_group_id]))
            else:
                raise ValueError('Column index type not recognized')
        if eval_result[0] and all(eval_result):
            return True
        return False
    
    def disk_read(self, row_group_id):
        return self.parquet_file.read_row_group(row_group_id).to_pandas()

    def build_pandas_query(self, predicates):
        nondoc_predicates = [p for p in predicates if type(p) != DocPredicate]
        query_string = ''
        for i, p in enumerate(nondoc_predicates):
            if type(p) == RangePredicate:
                if p.scalar:
                    if type(p.value) == str:
                        query_string += ' {} == "{}"'.format(p.column_name, p.value)
                    else:
                        query_string += ' {} == {}'.format(p.column_name, p.value)
                else:
                    query_string += ' {} >= {} & {} <= {}'.format(p.column_name, p.value[0], p.column_name, p.value[1])
            else:
                query_string += ' {} == "{}"'.format(p.column_name, p.value)
            if i < len(nondoc_predicates) - 1:
                query_string += ' {}'.format(' &')           
        return query_string
    
    def query_docstore(self, doc_predicates, df, k, result):
        docs_matching_predicates = {id: [] for id in df.index}
        for p in doc_predicates:
            docstore = self.column_indexes[p.column_name].docstore

            for doc_id in df.index:
                s = set(p.value)
                if set(docstore.get(doc_id)).intersection(s) == s:
                    docs_matching_predicates[id].append(True)
                else:
                    docs_matching_predicates[id].append(False)
            
            for id, matches in docs_matching_predicates.items():
                if matches[0] and all(matches):
                    result.append(id)
                
                if len(result) == k:
                    return result
        

    def query_with_docstore(self, predicates, k):
        # build pandas query string from predicates and junctions
        
        query_string = self.build_pandas_query(predicates)
        
        # get doc predicates and junctions
        doc_predicates = [p for p in predicates if type(p) == DocPredicate]
            
        result = []            
        for rg in range(self.ngroups):
            if self.query_rowgroup(predicates, rg):
                original_ids = list(range(rg*self.row_group_size, min((rg+1)*self.row_group_size, len(self.row_utilities))))
                df = self.disk_read(rg)
                
                df.set_index(np.array(original_ids), inplace=True)

                df = df.query(query_string)
                
                if len(doc_predicates) == 0:
                    for id in df.index:
                        result.append(id)
                        if len(result) == k:
                            return result
                if df.shape[0] > 0:
                    result = self.query_docstore(doc_predicates, df, k, result)
                    if len(result) == k:
                        return result     
        return result

    def query(self, predicates, k):
        skips = 0
        wasted_time = 0.0
        start = time.time()
        query_string = self.build_pandas_query(predicates)
        result = []            
        for rg in range(self.ngroups):
            # waste_start = time.time()
            if self.query_rowgroup(predicates, rg):
                waste_start = time.time()
                original_ids = list(range(rg*self.row_group_size, min((rg+1)*self.row_group_size, len(self.row_utilities))))
                df = self.disk_read(rg)
                
                df.set_index(np.array(original_ids), inplace=True)

                df = df.query(query_string)
                
                if df.shape[0] == 0:
                    wasted_time += time.time()-waste_start
                    
                for id in df.index:
                    result.append(id)
                    if len(result) == k:
                        self.exp_stats['Skip rate'].append(skips/(rg+1))
                        self.exp_stats['Query latencies'].append(time.time()-start)
                        self.exp_stats['Wasted time'].append(wasted_time)
                        return result 
            else:
                skips += 1
        self.exp_stats['Skip rate'].append(skips/(rg+1))
        self.exp_stats['Query latencies'].append(time.time()-start)
        self.exp_stats['Wasted time'].append(wasted_time)
        return result

                            