from boundedbloomfast import BoundedBlooms as bbf
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

# for corpus_size in [5000, 10000, 25000, 50000, 100000]:
#     corpus = [sample(l, len(l)-50) for _ in range(corpus_size)]
#     bbfast = bbf(1000000)
#     utils = np.random.normal(100, 10, size=corpus_size)
#     bbfast.add_all(corpus, utils)
#     bbfast.new_docstore(corpus, 'test.npy')

#     start = time.time()
#     q = ['apple', 'pear', 'raspberry', 'orange', 'truffle', 'sweet potato']
#     result = bbfast.query(q, topk=3, disk=True)
#     rt = time.time() - start
#     print("Time for corpus size {} is: {}".format(corpus_size, rt))
#     print("Amortized time for corpus size {} is: {}".format(corpus_size, rt/corpus_size))

#     for i in result:
#        print(utils[i])
#        print(len(q) - len(set(corpus[i]).intersection(set(q))))
# i = 0
# print("Query:", q)
# print("Member result:", result[i])
# print("Corpus:", corpus[i])

corpus_size = 10000
nitems = (len(l)-25)
corpus = [sample(l, nitems) for _ in range(corpus_size)]
utils = np.concatenate([np.random.normal(100, 10, size=corpus_size//2), np.random.normal(200, 10, size=corpus_size-corpus_size//2)]) 
q = ['apple', 'pear', 'raspberry', 'orange', 'potato', 'kale', 'okra']
k=100
n = len(q)
# ground truth
# print("ground truth")
# candidates = [i for i, cand in enumerate(corpus) if len(set(q).intersection(set(cand))) == n]

# srted = np.argsort(np.take(utils, candidates))[::-1][:k]
# print([candidates[j] for j in srted])
# print()

crs = []
olaps = [[], []]
disk_accesses = []
qtimes = []

for disk in [True, False]:
    for item_budget in [3, 4, 5, 6,7, 8]:
        # for item_budget in [5]:
        print("Item budget:", item_budget)
        compressed_size = corpus_size*nitems*item_budget
        bbfast = bbf(compressed_size)

        bbfast.add_all(corpus, utils)
        original_size = sum([len(bf) for bf in bbfast.bloom_filters])
        print("Original size is: {}".format(original_size))
        print("Compressed size is: {}".format(compressed_size))
        print("Compressed ratio is: {}".format(compressed_size/original_size))
        bbfast.new_docstore(corpus, 'test.npy')

        for i, doc in enumerate(corpus):
            assert bbfast.docstore.get(i).split(',') == doc, "Docstore is messed up for bbfast:  {}".format(i)
        bbfast.update_filter_lengths(optimizer_type='jensen', rounding_scheme='floor',
                                    cval='standard', equality_constraint=True)

        start = time.time()

        result1, daccess = bbfast.query(q, topk=k, disk=disk)
        rt = time.time() - start
        if disk:
            disk_accesses.append(daccess)
        print("Time BB for corpus size {} is: {}".format(corpus_size, rt))
        print("BB results")
        avg_gap = 0 
        for i in result1:
            avg_gap += len(q) - len(set(corpus[i]).intersection(set(q)))
        avg_gap /= len(result1)
        print("Avg gap:", avg_gap)
        # test scan
        scan = Scan(corpus, utils)
        scan.new_docstore(corpus, 'test1.npy')
        start = time.time()
        result2 = scan.query(q, topk=k)
        # for i in result2:
        #     print(i)
        #     print(set(corpus[i]).intersection(set(q)))
        #     print(utils[i])
        rt = time.time() - start
        print("Time Scan for corpus size {} is: {}".format(corpus_size, rt))
        # print("Amortized time for corpus size {} is: {}".format(corpus_size, rt/corpus_size))
        olap_pct = len(set(result1).intersection(set(result2)))/len(result1)

        if disk:
            olaps[1].append(olap_pct)
            crs.append(compressed_size/original_size)
        else:
            olaps[0].append(olap_pct)
        print("overlap percent:", olap_pct)

plt.plot(crs, olaps[0], label="No disk")
plt.plot(crs, olaps[1], label="Disk")
plt.xlabel('Compression Ratio')
plt.ylabel('Overlap coefficient with scan results')
plt.title('CR vs. overlap coefficient')
plt.legend()
plt.savefig("cr_ovlap_scan.png")
plt.clf()

plt.plot(crs, disk_accesses)
plt.xlabel('Compression Ratio')
plt.ylabel('Percent disk accesses')
plt.title('CR vs. disk accesses')
plt.legend()
plt.savefig("cr_daccess.png")

