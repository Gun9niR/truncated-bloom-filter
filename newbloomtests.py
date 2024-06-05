from boundedbloomfixedm import BoundedBlooms as bbf
from random import sample
import time
import numpy as np
from generalbaselines import Scan, InvertedIndex
import matplotlib.pyplot as plt



l = ['pear', 'apple', 'strawberry', 'blueberry', \
     'mango', 'grape', 'orange', 'banana', 'kiwi', 'pineapple', 'passionfruit',
     'watermelon', 'cantaloupe', 'honeydew', 'raspberry', 'blackberry', 'cherry', \
        'pomegranate', 'lychee', 'dragonfruit', 'guava', 'papaya', 'plum', 'apricot', \
            'peach', 'nectarine', 'coconut', 'lime', 'lemon', 'tangerine', 'grapefruit', \
                'kumquat', 'persimmon', 'fig', 'date', 'durian', 'jackfruit', 'breadfruit', \
                    'starfruit', 'quince', 'cranberry', 'gooseberry', 'boysenberry', 'loganberry', \
                        'elderberry', 'mulberry', 'currant', 'goji berry', 'salmonberry', 'cloudberry', \
                            'lingonberry', 'bearberry', 'barberry', 'cherimoya', 'custard apple', 'soursop',
                            'avocado', 'tomato', 'eggplant', 'bell pepper', 'chili pepper', 'jalapeno', \
                                'habanero', 'ghost pepper', 'cayenne pepper', 'poblano pepper', 'serrano pepper', \
                                    'anaheim pepper', 'pimento', 'cucumber', 'squash', 'zucchini', 'pumpkin', \
                                        'butternut squash', 'acorn squash', 'spaghetti squash', 'radish', 'turnip', \
                                            'rutabaga', 'carrot', 'beet', 'potato', 'sweet potato', 'yam', 'jicama', \
                                                'onion', 'shallot', 'garlic', 'ginger', 'wasabi', 'horseradish', 'celery', \
                                                    'fennel', 'asparagus', 'artichoke', 'broccoli', 'cauliflower', 'cabbage', \
                                                        'brussels sprouts', 'kale', 'collard greens', 'spinach', 'lettuce', 'arugula', \
                                                            'endive', 'chard', 'bok choy', 'watercress', 'chicory', 'radicchio', 'mushroom', \
                                                                'truffle', 'cassava', 'taro', 'yucca', 'bamboo shoot', 'bean sprout', 'water chestnut', \
                                                                    'lotus root', 'okra', 'chayote', 'rhubarb', 'beetroot', 'parsnip', 'parsley', 'cilantro', \
                                                                        'rosemary', 'thyme', 'oregano', 'sage', 'basil', 'mint', 'dill', 'chive', 'leek', 'scallion', \
                                                                            'kohlrabi', 'daikon', 'turnip', 'water spinach', 'watercress', 'chicory', 'radicchio', 'mushroom', \
                                                                                'truffle', 'cassava', 'taro', 'yucca', 'bamboo shoot', 'bean sprout', 'water chestnut', 'lotus root', \
                                                                                    'okra', 'chayote', 'rhubarb', 'beetroot', 'parsnip', 'parsley', 'cilantro', 'rosemary', 'thyme', 'oregano']






corpus_size = 10000
nitems = (len(l)-50)
corpus = [sample(l, nitems) for _ in range(corpus_size//3)] + [sample(l, nitems-15) for _ in range(corpus_size//3)] + [sample(l, nitems-50) for _ in range(corpus_size-(2*corpus_size//3))]
utils = np.concatenate([np.random.normal(100, 10, size=corpus_size//2), np.random.normal(1000, 10, size=corpus_size-corpus_size//2)]) 
q = ['apple', 'pear', 'raspberry', 'garlic']
k=10
n = len(q)
# ground truth
# print("ground truth")
# candidates = [i for i, cand in enumerate(corpus) if len(set(q).intersection(set(cand))) == n]

# srted = np.argsort(np.take(utils, candidates))[::-1][:k]
# print([candidates[j] for j in srted])
# print()
item_budget = 750
compressed_size = int(np.floor(0.5*corpus_size*item_budget))

scan = Scan(corpus, utils)
scan.new_docstore(corpus, 'test.npy')
start = time.time()
result2 = scan.query(q, topk=k)
rt = time.time() - start
print("Time for scan is: {}".format(rt))


bbfast = bbf(compressed_size, m=item_budget)

bbfast.add_all(corpus, utils)
original_size = bbfast.index_size()

print("Compression ratio is: {}".format(bbfast.index_size()/original_size))

print("Index size is {} KB".format(bbfast.index_size()/8000))



old_filter_lengths = [len(bf) for bf in bbfast.bloom_filters]
bbfast.new_docstore(corpus, 'test.npy')
bbfast.update_filter_lengths(optimizer_type='jensen', rounding_scheme='floor',
                                   cval='standard', equality_constraint=True)

new_filter_lengths = [len(bf) for bf in bbfast.bloom_filters]
print("Compression ratio is: {}".format(bbfast.index_size()/original_size))

plt.hist(np.array(new_filter_lengths)/np.array(old_filter_lengths), bins=100)
plt.savefig('filter_ratios_debug.png')
plt.clf()


disk = False
start = time.time()
result1 = bbfast.query(q, topk=k, disk=disk)
rt = time.time() - start


#debug

# for i in result2:
#     print('IDX', i)
#     print('n', bbfast.ns[i])
#     print('util', bbfast.utilities[i])
#     print(bbfast.bloom_filters[i])
#     print()

pcts = []
for i in result1:
    pcts.append(len(set(corpus[i]).intersection(set(q)))/len(set(q)))

print("Min intersection pct is: {}".format(min(pcts)))

print('BB new:', result1)
print('Scan:', result2)
print("Time for bbfast is: {}".format(rt))
print("Intersection of results is: {}".format(len(set(result1).intersection(set(result2)))/k))


ii = InvertedIndex(corpus, utils)
ii.build(corpus)

start = time.time()
result3 = ii.query(q, topk=k)
rt = time.time() - start
print("Time II for corpus size {} is: {}".format(corpus_size, rt))
print("Intersection of results is: {}".format(len(set(result3).intersection(set(result2)))/k))

