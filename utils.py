from gensim.parsing.preprocessing import remove_stopwords, strip_numeric
from string import punctuation
import numpy as np
from storage import mmapDocStore
from typing import List


ADDITIONAL_STOPWORDS = ['i', 'im', 'ive', 'youre', 'youve', 'youll', 'youd',  'shes', 'it', 'its','thatll', 'is', 'are', 'was', 'were', 'be', \
                        'dont', 'shouldve', 'arent', 'cant', 'couldnt',  'didnt', 'doesnt', 'hadnt', 'hasnt', 'havent', 'isnt', \
                        'mightnt', 'shouldnt', 'wasnt', 'werent', 'wont', 'wouldnt', 'them', 'theyre', 'theyve', 'theyll', 'thats', 'theres']

class Preprocessor:
    def __init__(self, stopword_removal=True, punctuation_removal=True, numeric_removal=True):
        self.stopword_removal = stopword_removal
        self.punctuation_removal = punctuation_removal
        self.numeric_removal = numeric_removal
    
    def preprocess_all(self, corpus):
        corpus = [doc.lower() for doc in corpus]
        if self.stopword_removal:
            corpus = [remove_stopwords(doc) for doc in corpus]
        if self.punctuation_removal:
            corpus = [doc.translate(str.maketrans('', '', punctuation)) for doc in corpus]
        if self.numeric_removal:
            corpus = [strip_numeric(doc) for doc in corpus]
        token_lists = [list(set(doc.split())) for doc in corpus]

        return [[token for token in tokens if token not in ADDITIONAL_STOPWORDS] for tokens in token_lists]
    
    def preprocess_item(self, doc):
        doc = doc.lower()
        if self.stopword_removal:
            doc = remove_stopwords(doc)
        if self.punctuation_removal:
            doc = doc.translate(str.maketrans('', '', punctuation))
        if self.numeric_removal:
            doc = strip_numeric(doc)
        return [token for token in doc.split() if token not in ADDITIONAL_STOPWORDS]
    
    def create_new_docstore(self, fpath, processed_corpus: List[List[str]]):
        processed_corpus = [','.join(term_list) for term_list in processed_corpus]
        maxdocsize = max([len(doc) for doc in processed_corpus])
        docstore =  mmapDocStore(fpath, len(processed_corpus), maxdocsize)
        for i, doc in enumerate(processed_corpus):
            docstore.put(i, doc)
        return docstore

    def preprocess_and_store(self, corpus, fpath):
        processed_corpus = self.preprocess_all(corpus)
        docstore = self.create_new_docstore(fpath, processed_corpus)
        return processed_corpus, docstore

class Metrics:
    def mean_jaccard_similarity(self, query, retrieved):
        return np.mean([len(query.intersection(r)) / len(query.union(r)) for r in retrieved])
    
    def mean_intersection_percent_at_k(self, query, retrieved, k):
        if len(retrieved) == 0:
            return 0
        return sum([len(query.intersection(r))/len(query) for r in retrieved])/k

    def prec_rec_at_k(self, retrieved, relevant, k):
        if len(retrieved) == 0:
            return 0, 0
        ret_rel = np.intersect1d(retrieved, relevant).shape[0]
        prec_at_k = ret_rel/k
        rec_at_k = ret_rel/len(relevant)
        return prec_at_k, rec_at_k 
    
    def mean_utility_at_k(self, retrieved, utilities, k):
        if len(retrieved) == 0:
            return 0
        return sum([utilities[i] for i in retrieved])/k