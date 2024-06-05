from columnindexes import *
import os
import numpy as np
    
class BloomSkipping(ColumnIndexes):
    def __init__(self, column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size=1000):
        super().__init__(column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size)
    
    def construct_indexes(self, group_keys_all, fpr, cr):
        # print(len(group_keys_all))
        # print(len(group_keys_all[0]))
        # print(len(group_keys_all[0][0]))
        for i in range(len(self.column_names)):
            if self.column_dtypes[i] == str:
                self.build_bloom_index(i, group_keys_all[i], fpr, cr, optim=True)
            else:
                self.build_range_index(i, group_keys_all[i])
        self.exp_stats['Index size'] = [self.index_size()]
                

class RangeSkipping(ColumnIndexes):
    def __init__(self, column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size=1000):
        super().__init__(column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size)
    
    def construct_indexes(self, group_keys_all):
        for i in range(len(self.column_names)):
            if self.column_dtypes[i] == str:
                self.build_alpha_range_index(i, group_keys_all[i])
            else:
                self.build_range_index(i, group_keys_all[i])
        self.exp_stats['Index size'] = [self.index_size()]

class DiskIndex(ColumnIndexes):
    def __init__(self, column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size=1000):
        super().__init__(column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size)
        
    def construct_indexes(self, group_keys_all, fpr, cr):
        for i in range(len(self.column_names)):
            if self.column_dtypes[i] == str:
                self.build_bloom_index(i, group_keys_all[i], fpr, cr, optim=True, trunc=False)
            else:
                self.build_range_index(i, group_keys_all[i])
        self.write_to_disk()
        read_time_start = time.time()
        self.read_from_disk()
        self.read_time_per_rg = (time.time()-read_time_start)/len(group_keys_all[0])
        self.cleanup_disk()
        self.exp_stats['Index size'] = [0]
    
    def query(self, predicates, k):
        skips = 0
        wasted_time = 0.0
        start = time.time()
        query_string = self.build_pandas_query(predicates)
        result = []            
        for rg in range(self.ngroups):
            # simulate loading index
            time.sleep(self.read_time_per_rg)
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
    
class InMemEqualTrunc(ColumnIndexes):
    def __init__(self, column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size=1000):
        super().__init__(column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size)
    
    def construct_indexes(self, group_keys_all, fpr, cr):
        for i in range(len(self.column_names)):
            if self.column_dtypes[i] == str:
                self.build_bloom_index(i, group_keys_all[i], fpr, cr, optim=False)
            else:
                self.build_range_index(i, group_keys_all[i])
        self.exp_stats['Index size'] = [self.index_size()]
        
class TopUtility(ColumnIndexes):
    def __init__(self, column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size=1000):
        super().__init__(column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size)
    
    def construct_indexes(self, group_keys_all, fpr, cr):
        for i in range(len(self.column_names)):
            if self.column_dtypes[i] == str:
                self.build_bloom_index(i, group_keys_all[i], fpr, cr, optim=False, trunc=True, topm=True)
            else:
                self.build_range_index(i, group_keys_all[i])
        self.exp_stats['Index size'] = [self.index_size()]

class HybridBloom(ColumnIndexes):
    def __init__(self, column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size=1000):
        super().__init__(column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size)
    
    def compute_per_rowgroup_sleep_times(self, read_time_per_rg):
        row_group_sleep_times = np.zeros(self.ngroups)
        n_cols = 0
        for _, index in self.column_indexes.items():
            if type(index) == dict:
                continue
            n_cols += 1
            row_group_sleep_times += np.array(index.length_ratios)
        row_group_sleep_times /= n_cols
        return row_group_sleep_times*read_time_per_rg
        

    def construct_indexes(self, group_keys_all, fpr, cr):
        for i in range(len(self.column_names)):
            if self.column_dtypes[i] == str:
                self.build_hybrid_bloom_index(i, group_keys_all[i], fpr, cr, variable_resolution=True)
            else:
                self.build_range_index(i, group_keys_all[i])
        
        self.write_to_disk()
        read_time_start = time.time()
        self.read_from_disk()
        self.read_time_per_rg = (time.time()-read_time_start)/len(group_keys_all[0])
        self.cleanup_disk()
        
        self.sleep_times = self.compute_per_rowgroup_sleep_times(self.read_time_per_rg)

        self.exp_stats['Index size'] = [self.index_size()]
    
    def query(self, predicates, k):
        skips = 0
        wasted_time = 0.0
        start = time.time()
        query_string = self.build_pandas_query(predicates)
        result = []            
        for rg in range(self.ngroups):
            if self.query_rowgroup(predicates, rg, self.sleep_times):
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
    
class TopUtilityHybrid(ColumnIndexes):
    def __init__(self, column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size=1000):
        super().__init__(column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size)
        self.breakpoints = {}
        self.index_size_sum = 0
    
    def index_size(self):
        return self.index_size_sum
    
    def compute_mean_breakpoint(self):
        breakpoint = 0
        ncols = 0
        for _, bp in self.breakpoints.items():
            breakpoint += bp
            ncols += 1
        # print('Breakpoint:', breakpoint/ncols)
        return math.floor(breakpoint/ncols)
        
    def construct_indexes(self, group_keys_all, fpr, cr):
        for i in range(len(self.column_names)):
            if self.column_dtypes[i] == str:
                self.build_hybrid_bloom_index(i, group_keys_all[i], fpr, cr, variable_resolution=False)
            else:
                self.build_range_index(i, group_keys_all[i])
        
        self.write_to_disk()
        read_time_start = time.time()
        self.read_from_disk()
        self.read_time_per_rg = (time.time()-read_time_start)/len(group_keys_all[0])
        self.cleanup_disk()
        
        # print('Read time per rowgroup:', self.read_time_per_rg*1e6)
        
        self.breakpoint = self.compute_mean_breakpoint()
        # print('Breakpoint:', self.breakpoint)
        
        self.exp_stats['Index size'] = [self.index_size()]
    
    def query(self, predicates, k):
        skips = 0
        wasted_time = 0.0
        start = time.time()
        query_string = self.build_pandas_query(predicates)
        result = []            
        for rg in range(self.ngroups):
            if rg >= self.breakpoint:
                time.sleep(self.read_time_per_rg)
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
    
    
class ElasticBF(ColumnIndexes):
    def __init__(self, column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size=1000):
        super().__init__(column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size)
    
    def construct_indexes(self, group_keys_all, fpr, cr):
        for i in range(len(self.column_names)):
            if self.column_dtypes[i] == str:
                self.build_elastic_bf_index(i, group_keys_all[i], fpr, cr)
            else:
                self.build_range_index(i, group_keys_all[i])
        self.exp_stats['Index size'] = [self.index_size()]
        
# class Monkey(ColumnIndexes):
#     def __init__(self, column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size=1000):
#         N = row_utilities.shape[0]
#         super().__init__(column_names, column_dtypes, ngroups, np.ones(N), parquet_file, row_group_size)
    
#     def construct_indexes(self, group_keys_all, fpr, cr):
#         for i in range(len(self.column_names)):
#             if self.column_dtypes[i] == str:
#                 self.build_bloom_index(i, group_keys_all[i], fpr, cr, optim=True)
#             else:
#                 self.build_range_index(i, group_keys_all[i])
#         self.exp_stats['Index size'] = [self.index_size()]