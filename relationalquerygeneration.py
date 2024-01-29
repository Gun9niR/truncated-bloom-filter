from columnindexes import BloomPredicate, RangePredicate, Query
import pickle
import os
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import random
import math
import matplotlib.pyplot as plt


class RelationalGenerator:
    def __init__(self, df):
        self.df = df
        self.df = self.df.sort_values(by=['utility'], ascending=False)
    
    def generate_queries(self, cat_cols, k, n_queries, outpath_prefix, tail_discard_prob=0.0, cutoff_pct=0.25, start=0):
        joint_dist = self.df.groupby(cat_cols).size().reset_index(name='count').sort_values(by=['count'], ascending=False)
        joint_dist = joint_dist.iloc[start:]
        num_cols = [col for col in self.df.columns if col not in cat_cols and col != 'utility']
        cutoff = math.floor(cutoff_pct*self.df.shape[0])
        #iterate through each row in the joint distribution and generate a query
        i = 0
        positions = []
        for _, row in joint_dist.iterrows():
            col_vals = row[cat_cols].values
            
            # convert the values to a list of strings
            col_vals = [str(val) for val in col_vals]
            
            # build pandas query string over categorical cols
            query_str = ' & '.join([f'{col} == "{val}"' for col, val in zip(cat_cols, col_vals)])
            
            #generate the query
            predicates_nonalpha = []
            predicates_alpha = []

            for j, col_name in enumerate(cat_cols):
                val = col_vals[j]
                #generate the predicates
                predicates_nonalpha.append(BloomPredicate(col_name, str, val))
                predicates_alpha.append(RangePredicate(col_name, str, val))
            

            
            # execute pandas query over the catgeorical columns and return top k rows
            top_k_rows = self.df.query(query_str).head(k)
            
            last_idx = np.max(list(top_k_rows.index.values))
            
            
            if last_idx >= cutoff and tail_discard_prob > 0.0:
                rnd = random.uniform(0, 1)
                if rnd <= tail_discard_prob:
                    continue
            
            positions.append(last_idx/self.df.shape[0])

            if len(top_k_rows) < k:
                raise ValueError('Not enough rows for desired number of queries')
            
            # extract min and max values for each numeric column
            min_vals = top_k_rows[num_cols].min().to_dict()
            max_vals = top_k_rows[num_cols].max().to_dict()
            
            # generate the predicates for the numeric columns
            predicates_num = []
            for col_name in num_cols:
                min_val = min_vals[col_name]
                max_val = max_vals[col_name]
                predicates_num.append(RangePredicate(col_name, float, (min_val, max_val)))
                
            qalpha = Query(predicates_alpha + predicates_num)
            qnonalpha = Query(predicates_nonalpha + predicates_num)
            
            # pickle dump to outpath using i as the query id and overwrite if necessary

            with open(f'{outpath_prefix}/query_{i}.pkl', 'wb') as f:
                pickle.dump((qalpha, qnonalpha), f)
            
            i += 1
                
            if i == n_queries:
                print(f'Generated {i} queries')
                print(f'Average position: {np.mean(positions)}')
                print(f'Max position: {np.max(positions)}')
                # plot histogram of positions
                plt.hist(positions, bins=25)
                plt.xlabel('Position (%)')
                plt.ylabel('Frequency')
                plt.show()
                break
    
class RelationalLoader:
    def __init__(self, parquet_file, query_directory, utility_file):
        self.parquet_file = parquet_file
        self.query_directory = query_directory
        self.utility_file = utility_file
    
    def load_queries(self):
        alpha_queries = []
        non_alpha_queries = []
        for qfile in os.listdir(self.query_directory):
            if not qfile.endswith('.pkl'):
                continue
            with open(os.path.join(self.query_directory, qfile), 'rb') as f:
                qalpha, qnonalpha = pickle.load(f)
                alpha_queries.append(qalpha)
                non_alpha_queries.append(qnonalpha)
        return alpha_queries, non_alpha_queries
    
    def extract_group_keys(self):
        parquet_file = pq.ParquetFile(self.parquet_file)
        group_keys_all = []
        col_dtypes = []
        
        table_pd = parquet_file.read().to_pandas()
        rg_size = parquet_file.read_row_group(0).num_rows
        
        for i in range(len(table_pd.columns)):
            group_keys = []
            for j in range(parquet_file.num_row_groups):
                s = table_pd[table_pd.columns[i]][j*rg_size:(j+1)*rg_size].tolist()
                if j == 0:
                    col_dtypes.append(type(s[0]))
                group_keys.append(s)
            group_keys_all.append(group_keys)
        return group_keys_all, col_dtypes, table_pd.columns
    
    def load_utilities(self):
        return np.load(self.utility_file)