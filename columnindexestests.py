from columnindexes import *
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
                  'animal': ["Flamingo", "Parrot", "Dog", "Horse",
                             "Brittle stars", "Centipede"]})

pq.write_table(table, 'example.parquet', row_group_size=3)
parquet_file = pq.ParquetFile('example.parquet')

table_pd = table.to_pandas()

# test column bloom
print("Column bloom tests")
cb = ColumnBloom(1e5, 0.001, row_group_size=3)
animal_col = table_pd['animal']
animal_col = list(animal_col)
animal_col1 = animal_col[:3]
animal_col2 = animal_col[3:]
test_input = [animal_col1, animal_col2]
utils = np.array([100, 90])

cb.add_all(test_input, utils)

index_size = cb.index_size()
print(index_size)

budget=int(0.8*index_size)

cb.update_budget(budget)
cb.update_filter_lengths(optimizer_type='jensen', rounding_scheme='floor',
                         cval='standard', equality_constraint=True)

assert abs(cb.index_size() - budget) < 5

# true positives rowgroup 0
start = time.time()
print(cb.query('Flamingo', 0))
print('Positive old:', (time.time()-start) * 1000*1000)
print(cb.query('Parrot', 0))
print(cb.query('Dog', 0))
start = time.time()
print(cb.query('Spiders', 0))
print('Negative old:', (time.time()-start) * 1000*1000)

# true positives rowgroup 1
print(cb.query('Horse', 1))
print(cb.query('Brittle stars', 1))
print(cb.query('Centipede', 1))

fp_count = 0

for q in ['Horse', 'Brittle stars', 'Centipede']:
    if cb.query(q, 0):
        fp_count += 1
# false positives rowgroup 0
print(fp_count/3)

# false positives rowgroup 1
fp_count = 0
for q in ['Flamingo', 'Parrot', 'Dog']:
    if cb.query(q, 1):
        fp_count += 1
print(fp_count/3)

# test column indexes
print("Column indexes tests")
ci = ColumnIndexes(table_pd.columns, [int, str], 2, np.random.normal(10, 1, table_pd.shape[0]), 'example.parquet', 3)

ci.build_bloom_index(1, test_input, 0.001, 0.5)



legs_column = table_pd['n_legs']
legs_column = list(legs_column)
legs_column1 = legs_column[:3]
legs_column2 = legs_column[3:]
test_input_2 = [legs_column1, legs_column2]

ci.build_range_index(0, test_input_2)

print(ci.column_indexes)

# sample predicates
bp = BloomPredicate('animal', str, 'Flamingo')
rp = RangePredicate('n_legs', int, (2,3))

predicates = [bp, rp]
print(ci.query(predicates, 1))

ci.write_to_disk()
ci.read_from_disk()

print("Alphabetical range tests")
ci.build_alpha_range_index(1, test_input)
print(ci.column_indexes)
print(ci.index_size())

predicates = [RangePredicate('animal', str, 'Flamingo'), rp]

print(ci.query(predicates, 1))

ci.write_to_disk()
start = time.time()
ci.read_from_disk()
print('Read time:', (time.time()-start) * 1000*1000)

# test hybrid bloom
print("Hybrid bloom tests")
cbs = ColumnBloomSplit(1e5, 0.001, row_group_size=3)
cbs.update_budget(budget)

cbs.add_all(test_input, utils)
cbs.generate_split(optimizer_type='jensen', rounding_scheme='floor',
                                cval='standard', equality_constraint=True)


# true positives rowgroup 0

print(cbs.query('Flamingo', 0, 0.1))
print(cbs.query('Parrot', 0, 0.1))
print(cbs.query('Dog', 0, 0.1))

# true positives rowgroup 1
start = time.time()
print(cbs.query('Horse', 1, 0.0000001))
print('Positive:', (time.time()-start) * 1000*1000)
print(cbs.query('Brittle stars', 1, 0.001))
print(cbs.query('Centipede', 1, 0.01))

start = time.time()
print(cbs.query('Spiders', 1, 0.0000001))
print('Negative:', (time.time()-start) * 1000*1000)


