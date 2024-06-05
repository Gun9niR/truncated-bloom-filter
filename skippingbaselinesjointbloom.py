from columnindexesjoint import *
import os
    
class BloomSkipping(ColumnIndexes):
    def __init__(self, column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size=1000):
        super().__init__(column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size)
    
    def construct_indexes(self, group_keys_all, fpr, cr):
        combined_group_keys = []
        for rg in range(len(group_keys_all[0])):
            local = []
            for rw in range(len(group_keys_all[0][rg])):
                joint = ''.join([group_keys_all[c][rg][rw] for c in range(len(self.column_names))])
                local.append(joint)
            combined_group_keys.append(local)
        self.build_combined_bloom_index(combined_group_keys, fpr, cr, optim=True)
        print(self.row_group_utilities)
        print(len(self.row_group_utilities) == self.ngroups)
    
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
        combined_group_keys = []
        for rg in range(len(group_keys_all[0])):
            local = []
            for rw in range(len(group_keys_all[0][rg])):
                joint = ''.join([group_keys_all[c][rg][rw] for c in range(len(self.column_names))])
                local.append(joint)
            combined_group_keys.append(local)
        self.build_combined_bloom_index(combined_group_keys, fpr, cr, optim=True, trunc=False)

        self.write_to_disk()
        read_time_start = time.time()
        self.read_from_disk()
        self.read_time_per_rg = (time.time()-read_time_start)/len(group_keys_all[0])
        self.cleanup_disk()
        self.exp_stats['Index size'] = [0]
    
    def query_joint(self, predicates, k):
        skips = 0
        wasted_time = 0.0
        start = time.time()
        query_string = self.build_pandas_query(predicates)
        result = []            
        for rg in range(self.ngroups):
            # simulate loading index
            time.sleep(self.read_time_per_rg)
            # waste_start = time.time()
            if self.query_rowgroup_joint(predicates, rg):
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
        combined_group_keys = []
        for rg in range(len(group_keys_all[0])):
            local = []
            for rw in range(len(group_keys_all[0][rg])):
                joint = ''.join([group_keys_all[c][rg][rw] for c in range(len(self.column_names))])
                local.append(joint)
            combined_group_keys.append(local)
        self.build_combined_bloom_index(combined_group_keys, fpr, cr, optim=False)
        self.exp_stats['Index size'] = [self.index_size()]
        
class TopUtility(ColumnIndexes):
    def __init__(self, column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size=1000):
        super().__init__(column_names, column_dtypes, ngroups, row_utilities, parquet_file, row_group_size)
    
    def construct_indexes(self, group_keys_all, fpr, cr):
        combined_group_keys = []
        for rg in range(len(group_keys_all[0])):
            local = []
            for rw in range(len(group_keys_all[0][rg])):
                joint = ''.join([group_keys_all[c][rg][rw] for c in range(len(self.column_names))])
                local.append(joint)
            combined_group_keys.append(local)
        self.build_combined_bloom_index(combined_group_keys, fpr, cr, optim=False, trunc=True, topm=True)
        self.exp_stats['Index size'] = [self.index_size()]
    
