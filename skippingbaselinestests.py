import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from skippingbaselines import *
from columnindexes import *

table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
                  'animal': ["Flamingo", "Parrot", "Dog", "Horse",
                             "Brittle stars", "Centipede"]})

rg_size = 3
pq.write_table(table, 'example.parquet', row_group_size=rg_size)
parquet_file = pq.ParquetFile('example.parquet')

table_pd = table.to_pandas()



# split each column into row groups and create one big list
group_keys_all = []
for i in range(len(table_pd.columns)):
    group_keys = []
    for j in range(parquet_file.num_row_groups):
        s = table_pd[table_pd.columns[i]][j*rg_size:(j+1)*rg_size].tolist()
        group_keys.append(s)
        print(type(s[0]))
    group_keys_all.append(group_keys)
    



column_names = table_pd.columns
column_dtypes = [int, str]

# test bloom skipping
print("Bloom skipping tests")
bs = BloomSkipping(column_names, column_dtypes, 2, np.random.normal(10, 1, table_pd.shape[0]), 'example.parquet', 3)
bs.construct_indexes(group_keys_all, 0.001, 0.5)

bp = BloomPredicate('animal', str, "Centipede")
rp = RangePredicate('n_legs', int, 100)

predicates = [bp, rp]

print(bs.query(predicates, 1))

bs.average_stats()

print(bs.exp_stats)

print("Testing DiskIndex")
di = DiskIndex(column_names, column_dtypes, 2, np.random.normal(10, 1, table_pd.shape[0]), 'example.parquet', 3)
di.construct_indexes(group_keys_all, 0.001, 0.5)
print(di.query(predicates, 1))
print(di.exp_stats)

print("Testing InMemEqualTrunc")
imet = InMemEqualTrunc(column_names, column_dtypes, 2, np.random.normal(10, 1, table_pd.shape[0]), 'example.parquet', 3)
imet.construct_indexes(group_keys_all, 0.001, 0.5)
print(imet.query(predicates, 1))
print(imet.exp_stats)

print("Testing top utility")
tu = TopUtility(column_names, column_dtypes, 2, np.random.normal(10, 1, table_pd.shape[0]), 'example.parquet', 3)
tu.construct_indexes(group_keys_all, 0.001, 0.5)
print(tu.query(predicates, 1))
print(tu.exp_stats)

print('Testing HybridBloom (split)')

hb = HybridBloom(column_names, column_dtypes, 2, np.random.normal(10, 1, table_pd.shape[0]), 'example.parquet', 3)
hb.construct_indexes(group_keys_all, 0.001, 0.5)
print(hb.query(predicates, 1))
print(hb.exp_stats)

print('Testing Hybrid top utility')
tuh = TopUtilityHybrid(column_names, column_dtypes, 2, np.random.normal(10, 1, table_pd.shape[0]), 'example.parquet', 3)
tuh.construct_indexes(group_keys_all, 0.001, 0.5)
print(tuh.query(predicates, 1))
print(tuh.exp_stats)

print('Testing Elastic BF analog')
table_large = pa.table({'n_legs': [2, 2, 4, 4, 5, 100, 2, 2, 4, 4, 5, 100],
                  'animal': ["Flamingo", "Parrot", "Dog", "Horse",
                             "Brittle stars", "Centipede", "Flamingo", "Parrot", "Dog", "Horse",
                             "Brittle stars", "Centipede"]})

rg_size = 3
pq.write_table(table_large, 'example2.parquet', row_group_size=rg_size)
parquet_file_large = pq.ParquetFile('example2.parquet')

table_pd_lg = table_large.to_pandas()

column_names = table_pd_lg.columns
column_dtypes = [int, str]

group_keys_all_2 = []
for i in range(len(table_pd_lg.columns)):
    group_keys = []
    for j in range(parquet_file_large.num_row_groups):
        s = table_pd_lg[table_pd_lg.columns[i]][j*rg_size:(j+1)*rg_size].tolist()
        group_keys.append(s)
        print(type(s[0]))
    group_keys_all_2.append(group_keys)
print(group_keys_all_2)
eb = ElasticBF(column_names, column_dtypes, len(group_keys_all_2[0]), np.random.normal(10, 1, table_pd_lg.shape[0]), 'example2.parquet', 3)
eb.construct_indexes(group_keys_all_2, 0.001, 0.5)
print(eb.query(predicates, 2))
print(eb.exp_stats)

print("Testing RangeSkipping")
rps = RangePredicate('animal', str, "Centipede")
predicates = [rps, rp]
rs = RangeSkipping(column_names, column_dtypes, 2, np.random.normal(10, 1, table_pd.shape[0]), 'example.parquet', 3)
rs.construct_indexes(group_keys_all)

print(rs.query(predicates, 1))
print(rs.exp_stats)
