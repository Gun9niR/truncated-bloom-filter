from dataloader import AmazonLoader
from boundedbloomfast import BoundedBlooms
import math
import matplotlib.pyplot as plt
import numpy as np

adl = AmazonLoader('data/processed', 'amazon_industrial_scientific')

corpus = adl.read_corpus()
# corpus = [doc for doc in corpus if len(doc) < 2000 and len(doc) > 75]
corpus = [doc for doc in corpus if len(doc) > 2 and len(doc) < 100]
print("Number of docs:", len(corpus))
utils = adl.read_utilities()[:len(corpus)]
# queries = adl.read_queries()

bbfast = BoundedBlooms(1000000)

bbfast.add_all(corpus, utils)
assert sum([len(bf) for bf in bbfast.bloom_filters]) == bbfast.index_size(), "Index size is messed up"

for frac in np.arange(0.1, 0.95, 0.05):
    compressed_size = int(frac*bbfast.index_size())
    original_size = sum([len(bf) for bf in bbfast.bloom_filters])
    print("Original size is: {}".format(original_size))
    print("Compressed size is: {}".format(compressed_size))
    print("Compressed ratio is: {}".format(compressed_size/original_size))
    bbfast.update_budget(compressed_size)

    assert bbfast.bit_budget == compressed_size, "Bit budget is messed up"

    # ms = [len(bf) for bf in bbfast.bloom_filters]
    # plt.hist(ms, bins=100)
    # plt.savefig('m_debug.png')
    # plt.clf()
    # ks = bbfast.ks
    # plt.hist(ks, bins=100)
    # plt.savefig('k_debug.png')
    # plt.clf()
    # ns = bbfast.ns
    # plt.hist(ns, bins=100)
    # plt.savefig('n_debug.png')
    # plt.clf()


    before = [len(bf) for bf in bbfast.bloom_filters]

    # bbfast.new_docstore(corpus, 'bloom.npy')
    bbfast.update_filter_lengths(optimizer_type='jensen', rounding_scheme='floor',
                                cval='standard', equality_constraint=True)
    

    after = [len(bf) for bf in bbfast.bloom_filters]
    view = 10
    print("ms:", bbfast.ms[:view])
    print("Differences:", [before[i] - after[i] for i in range(len(before))][:view])
    print("Utilities:", utils[:view])
    bbfast.reset()