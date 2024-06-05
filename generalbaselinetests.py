from boundedbloomfast import BoundedBlooms as bbf
from random import sample
import time
import numpy as np
from generalbaselines import Scan, InvertedIndex, TopKInvertedIndex, TopMDoc, TopMDocSet



l = list(set(['pear', 'apple', 'strawberry', 'blueberry', \
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
                                                        # 'brussels sprouts', 'kale', 'spinach', 'lettuce', 'arugula', \
                                                            'endive', 'chard', 'bok choy', 'watercress', 'chicory', 'radicchio', 'mushroom', \
                                                                'truffle', 'cassava', 'taro', 'yucca', 'bamboo shoot', 'bean sprout', 'water chestnut', \
                                                                    'lotus root', 'okra', 'chayote', 'rhubarb', 'beetroot', 'parsnip', 'parsley', 'cilantro', \
                                                                        'rosemary', 'thyme', 'oregano', 'sage', 'basil', 'mint', 'dill', 'chive', 'leek', 'scallion', \
                                                                            'kohlrabi', 'daikon', 'turnip', 'water spinach', 'watercress', 'chicory', 'radicchio', 'mushroom', \
                                                                                'truffle', 'cassava', 'taro', 'yucca', 'bamboo shoot', 'bean sprout', 'water chestnut', 'lotus root', \
                                                                                    'okra', 'chayote', 'rhubarb', 'beetroot', 'parsnip', 'parsley', 'cilantro', 'rosemary', 'thyme', 'oregano']))

corpus_size = 10000
# corpus_size = 10
nitems = (len(l)-60)
# corpus = [sample(l, nitems) for _ in range(corpus_size)]
corpus = [sample(l, nitems-25) for _ in range(corpus_size//2)] + [sample(l, nitems) for _ in range(corpus_size-corpus_size//2)]
utils = np.random.normal(200, 40, size=corpus_size)
# utils = np.concatenate([np.random.normal(100, 10, size=corpus_size//2), np.random.normal(200, 10, size=corpus_size-corpus_size//2)]) 
q = ['apple', 'pear', 'raspberry', 'orange', 'potato', 'kale', 'okra', 'rhubarb', 'beetroot', 'parsnip', 'parsley']
k=5
n = len(q)

# ground truth

# candidates = [i for i, cand in enumerate(corpus) if len(set(q).intersection(set(cand))) == n]

# srted = np.argsort(np.take(utils, candidates))[::-1][:k]
# print("Ground truth top k", [candidates[j] for j in srted])



# item_budget = 6.5
#         # for item_budget in [5]:
# print("Item budget:", item_budget)
# compressed_size = int(corpus_size*nitems*item_budget)
# bbfast = bbf(compressed_size)
bbfast = bbf(1000)

bbfast.add_all(corpus, utils)
original_size = sum([len(bf) for bf in bbfast.bloom_filters])
print("Max filter length og: {}".format(max([len(bf) for bf in bbfast.bloom_filters])))
print("Min filter length og: {}".format(min([len(bf) for bf in bbfast.bloom_filters])))

compressed_size = int(0.9*original_size)
bbfast.update_budget(compressed_size)

print("Original size is: {}".format(original_size))
print("Compressed size is: {}".format(compressed_size))
print("Compressed ratio is: {}".format(compressed_size/original_size))
bbfast.new_docstore(corpus, 'test.npy')

for i, doc in enumerate(corpus):
    assert bbfast.docstore.get(i).split(',') == doc, "Docstore is messed up for bbfast:  {}".format(i)
bbfast.update_filter_lengths(optimizer_type='jensen', rounding_scheme='floor',
                            cval='standard', equality_constraint=True)
print("Max filter length compressed is: {}".format(max([len(bf) for bf in bbfast.bloom_filters])))
print("Min filter length compressed is: {}".format(min([len(bf) for bf in bbfast.bloom_filters])))

start = time.time()
# result1, daccess = bbfast.query(q, topk=k, disk=False)
result1 = bbfast.query(q, topk=k, disk=False)

rt = time.time() - start
print("BB results")
print("Time BB for corpus size {} is: {}".format(corpus_size, rt))
print("BB size is: {}".format(bbfast.index_size()))

scan = Scan(corpus, utils)
scan.new_docstore(corpus, 'test1.npy')
start = time.time()
result2 = scan.query(q, topk=k)
rt = time.time() - start
print("Time Scan for corpus size {} is: {}".format(corpus_size, rt))
# print("Amortized time for corpus size {} is: {}".format(corpus_size, rt/corpus_size))
olap_pct = len(set(result1).intersection(set(result2)))/len(result1)
print("Overlap pct BB:", olap_pct)

intersections = []
for i in result1:
    intersections.append(len(set(corpus[i]).intersection(set(q)))/len(q))

print("Worst case intersection percent:", min(intersections))

print("II results")
ii = InvertedIndex(corpus, utils)
ii.build(corpus)

start = time.time()
result3 = ii.query(q, topk=k)
rt = time.time() - start
print("Time II for corpus size {} is: {}".format(corpus_size, rt))
# print("Amortized time for corpus size {} is: {}".format(corpus_size, rt/corpus_size))
olap_pct = len(set(result2).intersection(set(result3)))/len(result2)
print("II size is: {}".format(ii.index_size()))
print("Overlap pct II:", olap_pct)
print("Top M Doc results")
# tmd = TopMDoc(corpus, utils, corpus_size*nitems*item_budget)
tmd = TopMDoc(corpus, utils, compressed_size)
tmd.build(corpus)
start = time.time()
result4 = tmd.query(q, topk=k)
rt = time.time() - start
print("Time TMD for corpus size {} is: {}".format(corpus_size, rt))
print("Number of docs:", tmd.M)

olap_pct = len(set(result2).intersection(set(result4)))/len(result2)
print("TMD size is: {}".format(tmd.index_size()))
print("Overlap pct TMD:", olap_pct)
print("Top M Doc Set results")

# tmds = TopMDocSet(corpus, utils, corpus_size*nitems*item_budget)
tmds = TopMDocSet(corpus, utils, compressed_size)
tmds.build(corpus)
start = time.time()
result5 = tmds.query(q, topk=k)
rt = time.time() - start
print("Time TMDS for corpus size {} is: {}".format(corpus_size, rt))
print("TMDS size is: {}".format(tmds.index_size()))

olap_pct = len(set(result2).intersection(set(result5)))/len(result2)
print("Overlap pct TMDS:", olap_pct)

print("Top K Inverted Index results")
tkii = TopKInvertedIndex(corpus, utils, 2500)

tkii.build(corpus)
start = time.time()
result6 = tkii.query(q, topk=k)
rt = time.time() - start
print("Time TKII for corpus size {} is: {}".format(corpus_size, rt))
print("TKII size is: {}".format(tkii.index_size()))

olap_pct = len(set(result2).intersection(set(result6)))/len(result2)

print("Overlap pct TKII:", olap_pct)
