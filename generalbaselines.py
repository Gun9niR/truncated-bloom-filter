from abc import abstractmethod
from utils import Preprocessor
import numpy as np
import math

INT_SIZE = 4
BITS_PER_BYTE = 8

class GeneralBaseline:
    def __init__(self, corpus, utilities):
        self.corpus = corpus
        self.utilities = utilities
    @abstractmethod
    def query(self, q, topk):
        pass
    @abstractmethod
    def index_size(self):
        pass
    
    def process_and_store_all(self, corpus, filter_parameter, fpath):
        word_lists, self.docstore = Preprocessor().preprocess_and_store(corpus, fpath)
        self.N = len(word_lists)
        return word_lists
    
    def new_docstore(self, processed_corpus, fpath):
        self.docstore = Preprocessor().create_new_docstore(fpath, processed_corpus)
        self.N = len(processed_corpus)
        
    def cached_doc_pct(self):
        return self.cached_doc_pct


class Scan(GeneralBaseline):
    def __init__(self, corpus, utilities):
        super().__init__(corpus, utilities)

    def query(self, q, topk):
        candidates = []
        for i in range(self.N):
            if len(set(self.docstore.get(i).split(',')).intersection(set(q))) == len(q):
                candidates.append(i)

        toputil = np.argsort(np.take(self.utilities, candidates))[::-1][:topk]
        if len(toputil) == 0:   
            raise ValueError("O length result scan!!!")
        return [candidates[j] for j in toputil]
        
    def index_size(self):
        return 0

class InvertedIndex(GeneralBaseline):
    def __init__(self, corpus, utilities):
        super().__init__(corpus, utilities)
    
    def build(self, processed_corpus):
        self.inverted_index = {}
        self.int_size_bits = math.ceil(math.log2(len(processed_corpus)))
        for i, word_list in enumerate(processed_corpus):
            for word in word_list:
                if word not in self.inverted_index:
                    self.inverted_index[word] = []
                self.inverted_index[word].append(i)

    def query(self, q, topk):
        candidates = {}
        for word in q:
            if word in self.inverted_index:
                for i in self.inverted_index[word]:
                    if i not in candidates:
                        candidates[i] = 0
                    candidates[i] += 1
        candidates = [cand for cand, matches in candidates.items() if matches == len(q)]
        toputil = np.argsort(np.take(self.utilities, candidates))[::-1][:topk]
        return [candidates[i] for i in toputil]
            
    def index_size(self):
        size = 0
        for word, doclist in self.inverted_index.items():
            # size += (len(word)*BITS_PER_BYTE + len(doclist)*INT_SIZE*BITS_PER_BYTE)
            size += (len(word)*BITS_PER_BYTE + len(doclist)*self.int_size_bits)
        return size
    
class TopKInvertedIndex(InvertedIndex):
    def __init__(self, corpus, utilities, k_max):
        super().__init__(corpus, utilities)
        self.k_max = k_max
    
    def build(self, processed_corpus):
        super().build(processed_corpus)
        for word, doclist in self.inverted_index.items():
            sorted = np.argsort(np.take(self.utilities, doclist))[::-1][:self.k_max]
            self.inverted_index[word] = [doclist[i] for i in sorted]


class TopMDoc(InvertedIndex):
    def __init__(self, corpus, utilities, budget):
        super().__init__(corpus, utilities)
        self.budget = budget
        self.size = 0

    def build(self, processed_corpus):
        self.int_size_bits = math.ceil(math.log2(len(processed_corpus)))
        self.M = 0
        top_util_idxs = np.argsort(self.utilities)[::-1]
        self.inverted_index = {}
        for i in top_util_idxs:
            word_list = processed_corpus[i]
            doc_cost = 0
            for word in word_list:
                if word not in self.inverted_index:
                    # doc_cost += (len(word)*BITS_PER_BYTE + INT_SIZE*BITS_PER_BYTE)
                    doc_cost += (len(word)*BITS_PER_BYTE + self.int_size_bits)
                else:
                    # doc_cost += INT_SIZE*BITS_PER_BYTE
                    doc_cost += self.int_size_bits
            if (self.index_size() + doc_cost) > self.budget:
                break
            for word in word_list:
                if word not in self.inverted_index:
                    self.inverted_index[word] = []
                self.inverted_index[word].append(i)
            self.size += doc_cost
            self.M += 1
            
        self.cached_doc_pct = self.M/len(self.corpus)
        
        for word, doclist in self.inverted_index.items():
            sorted = np.argsort(np.take(self.utilities, doclist))[::-1]
            self.inverted_index[word] = [doclist[i] for i in sorted]

    def index_size(self):
        return self.size
    

class TopMDocSet(GeneralBaseline):
    def __init__(self, corpus, utilities, budget):
        super().__init__(corpus, utilities)
        self.budget = budget
        self.size = 0
    
    def build(self, processed_corpus):
        self.M = 0
        top_util_idxs = np.argsort(self.utilities)[::-1]
        index_mapper = {}
        self.index = []

        for j, i in enumerate(top_util_idxs):
            word_list = processed_corpus[i]
            doc_cost = sum([len(word) for word in word_list])*BITS_PER_BYTE
            if (self.index_size() + doc_cost) > self.budget:
                break
            self.size += doc_cost
            self.index.append(word_list)
            self.M +=1
            index_mapper[j] = i
        self.index_mapper = index_mapper
        self.cached_doc_pct = self.M/len(self.corpus)
    
    def index_size(self):
        return self.size

    def query(self, q, topk):
        candidates = []
        for i, doc in enumerate(self.index):
            if len(set(doc).intersection(set(q))) == len(q):
                candidates.append(self.index_mapper[i])
        toputil = np.argsort(np.take(self.utilities, candidates))[::-1][:topk]
        return [candidates[i] for i in toputil]
        
# if __name__ == "__main__":
#     # test all of the indexes

#     test = ["January is cool", "February is cool", "March is cool", "April is cool", "May is cool", "June is cool", "July is cool", "August is cool", "September is cool", "October is cool", "November is cool", "December is cool"]
#     utilities = np.random.normal(1000, 100, len(test))
#     q = set(["february", "is", "cool"])
#     topk = 5
#     match_threshold_percent = 0.5
#     print("Testing Scan")
#     scan = Scan(test, utilities)
#     print(scan.query(q, topk, match_threshold_percent))
#     print("Testing Inverted Index")
#     ii = InvertedIndex(test, utilities)
#     ii.build()
#     print("Index Size Inverted: ", ii.index_size())
#     print(ii.query(q, topk, match_threshold_percent))
#     print("Testing Top K Inverted Index")
#     tii = TopKInvertedIndex(test, utilities, 5)
#     tii.build()
#     print("Index Size Top K Inverted: ", tii.index_size())
#     print(tii.query(q, topk, match_threshold_percent))
#     print("Testing Top M Doc")
#     tmd = TopMDoc(test, utilities, 500)
#     tmd.build()
#     print("Index Size Top M Doc: ", tmd.index_size())
#     print(tmd.query(q, topk, match_threshold_percent))
#     print("Testing Top M Doc Set")
#     tmds = TopMDocSet(test, utilities, 500)
#     tmds.build()
#     print("Index Size Top M Doc Set: ", tmds.index_size())
#     print(tmds.query(q, topk, match_threshold_percent))


    # import ir_datasets
    # dataset = ir_datasets.load("beir/nfcorpus/test")
    # corpus = []
    # for doc in dataset.docs_iter():
    #     corpus.append(doc.text)
    
    # utilities = np.random.normal(1000, 100, len(corpus))
    # print("Testing Inverted Index")
    # ii = InvertedIndex(corpus, utilities)
    # ii.build()
    # print("Index Size Inverted: ", ii.index_size())
    # print("Testing Top K Inverted Index")
    # tii = TopKInvertedIndex(corpus, utilities, 5)
    # tii.build()
    # print("Index Size Top K Inverted: ", tii.index_size())
    # print("Testing Top M Doc")
    # tmd = TopMDoc(corpus, utilities, len(corpus)*1000)
    # tmd.build()
    # print("Index Size Top M Doc: ", tmd.index_size())
    # print("Testing Top M Doc Set")
    # tmds = TopMDocSet(corpus, utilities, len(corpus)*1000)
    # tmds.build()
    # print("Index Size Top M Doc Set: ", tmds.index_size())
