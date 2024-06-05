from wrinklebaselines import *
from generalbaselines import Scan
import time
from random import sample
import numpy as np

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
nitems = (len(l)-50)
# corpus = [sample(l, nitems) for _ in range(corpus_size)]
corpus = [sample(l, nitems) for _ in range(corpus_size//2)] + [sample(l, nitems-30) for _ in range(corpus_size-corpus_size//2)]
utils = np.random.normal(200, 25, size=corpus_size)
# utils = np.concatenate([np.random.normal(100, 10, size=corpus_size//2), np.random.normal(200, 10, size=corpus_size-corpus_size//2)]) 
q = ['apple', 'pear', 'raspberry', 'orange', 'potato', 'kale', 'okra']
k=25
n = len(q)

# ground truth

# candidates = [i for i, cand in enumerate(corpus) if len(set(q).intersection(set(cand))) == n]

# srted = np.argsort(np.take(utils, candidates))[::-1][:k]
# print("Ground truth top k", [candidates[j] for j in srted])



item_budget = 3
        # for item_budget in [5]:
print("Item budget:", item_budget)
compressed_size = corpus_size*nitems*item_budget

scan = Scan(corpus, utils)
scan.new_docstore(corpus, 'test1.npy')
start = time.time()
gtruth = scan.query(q, topk=k)
rt = time.time() - start
print("Time Scan for corpus size {} is: {}".format(corpus_size, rt))

et = EqualTruncation(compressed_size)
et.add_all(corpus, utils)
et.new_docstore(corpus, 'test.npy')
et.update_filter_lengths()
print("Equal Truncation")
start = time.time()
result = et.query(q, topk=k, disk=True)

rt = time.time() - start
print("Time for corpus size {} is: {}".format(corpus_size, rt))

print("Overlap Equal Truncation", len(set(gtruth).intersection(set(result)))/len(gtruth))

tu = TopUtility(compressed_size)
tu.add_all(corpus, utils)
tu.new_docstore(corpus, 'test.npy')
tu.update_filter_lengths()
print("Top Utility")
start = time.time()
result = tu.query(q, topk=k, disk=True)
rt = time.time() - start
print("Time for corpus size {} is: {}".format(corpus_size, rt))
print("Overlap Top Utility", len(set(gtruth).intersection(set(result)))/len(gtruth))

ru = RandomUniform(compressed_size)
ru.add_all(corpus, utils)
ru.new_docstore(corpus, 'test.npy')
ru.update_filter_lengths()
print("Random Uniform")
start = time.time()

result = ru.query(q, topk=k, disk=True)
rt = time.time() - start
print("Time for corpus size {} is: {}".format(corpus_size, rt))
print("Overlap Random Uniform", len(set(gtruth).intersection(set(result)))/len(gtruth))

ru = RandomUtility(compressed_size)
ru.add_all(corpus, utils)
ru.new_docstore(corpus, 'test.npy')
ru.update_filter_lengths()
print("Random Utility")
start = time.time()

result = ru.query(q, topk=k, disk=True)
rt = time.time() - start
print("Time for corpus size {} is: {}".format(corpus_size, rt))
print("Overlap Random Utility", len(set(gtruth).intersection(set(result)))/len(gtruth))








