from utils import Preprocessor
from abc import abstractmethod
import numpy as np
import os
import gzip
import json
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, out_path, dset_name):
        self.out_path = out_path
        self.dset_name = dset_name
    
    @abstractmethod
    def load_raw(self, file_path):
        pass
    
    def clean(self, lower_token_filter=3, upper_token_filter=100):
        self.corpus = Preprocessor().preprocess_all(self.corpus)
        self.corpus = [doc for doc in self.corpus if len(doc) >= lower_token_filter and len(doc) <= upper_token_filter] 

        with open(os.path.join(self.out_path, self.dset_name + '_clean' + '.txt'), 'w') as f:
            for doc in self.corpus:
                f.write(','.join(doc) + '\n')
    
    def write_utilities(self, distribution_type='norm', split_factor=2, density_factor=2, spread_factor=4, nmix=5, plot=True):
        N = len(self.corpus)
        if distribution_type == "norm":
            # self.utilities = np.random.normal(200, 25, N)
            self.utilities = np.random.normal(1, 0.1, N)
        elif distribution_type == "binorm":
            self.utilities = np.random.normal(75, 5, N//split_factor)
            self.utilities = np.concatenate((self.utilities, np.random.normal(125, 12.5, N - (N//split_factor))))
        elif distribution_type == "uni":
            self.utilities = np.random.uniform(0.45, 1.55, N)
        elif distribution_type == "normmix":
            starting_mu, starting_sigma = 1, 0.1
            frequencies = [0.5/(density_factor**i) for i in range(nmix)]
            
            frequencies = frequencies + [1 - sum(frequencies)]
            
            if sum(frequencies) > 1:
                raise ValueError("Sum of frequencies is greater than 1")
            
            importance_factors = [spread_factor**i for i in range(nmix)]
            
            self.utilities = np.random.normal(starting_mu, starting_sigma, int(frequencies[0]*N))
            for i in range(1, nmix):
                self.utilities = np.concatenate((self.utilities, np.random.normal(starting_mu*importance_factors[i], starting_sigma*importance_factors[i], int(frequencies[i]*N))))

            remaining_N = N - len(self.utilities)
            self.utilities = np.concatenate((self.utilities, np.random.normal(starting_mu, starting_sigma, remaining_N)))
        else:
            raise NotImplementedError("utility distribution not implemented yet")
        assert np.all(self.utilities >= 0), "Negative utilities present" 
        self.utility_probs = self.utilities/np.sum(self.utilities)
        
        np.save(os.path.join(self.out_path, self.dset_name + '_utilities' + '.npy'), self.utilities)
        if plot:
            plt.hist(self.utilities, bins=50)
            plt.savefig('utility_distribution.png')
            plt.clf()      
    
    def inverted_lookup(self):
        inverted_index = {}
        for i, word_sets in enumerate(self.corpus):
            for word in word_sets:
                if word not in inverted_index:
                    inverted_index[word] = []
                inverted_index[word].append(i)
        self.inverted_index = inverted_index
    
    def write_ngram_queries(self, n, number_queries, minmatches, start=5):
        self.inverted_lookup()
        queries = []
        while len(queries) < number_queries:
            i = np.random.choice(range(len(self.corpus)), size=None, replace=False, p=self.utility_probs)
            words = list(self.corpus[i])
            if len(words) < n + start:
                continue
            word_nmatching_docs = {w: len(self.inverted_index[w]) for w in words}

            word_nmatching_docs = {k: v for k, v in sorted(word_nmatching_docs.items(), key=lambda item: item[1], reverse=True)}
            q = list(dict(list(word_nmatching_docs.items())[start:start+n]).keys())
            nintersect_docs = len(set.intersection(*([set(self.inverted_index[w]) for w in q])))
            if nintersect_docs < minmatches:
                continue
            queries.append(q)
        self.queries = queries

        with open(os.path.join(self.out_path, self.dset_name + '_queries' + '.txt'), 'w') as f:
            for q in queries:
                f.write(','.join(q) + '\n')
        
    def read_queries(self):
        with open(os.path.join(self.out_path, self.dset_name + '_queries' + '.txt'), 'r') as f:
            queries = [list(q.strip().split(',')) for q in f.readlines()]
        return queries
    
    def read_utilities(self):
        return np.load(os.path.join(self.out_path, self.dset_name + '_utilities' + '.npy'))
    
    def read_corpus(self):
        with open(os.path.join(self.out_path, self.dset_name + '_clean' + '.txt'), 'r') as f:
            corpus = [list(doc.strip().split(',')) for doc in f.readlines()]
        return corpus
    
    def make_pipeline(self, in_path, ngram, nqueries, minmatches, distribution_type='norm',
                      split_factor=3, lower_token_filter=3, upper_token_filter=100, start=5,
                      density_factor=2, spread_factor=4, nmix=5):
        self.load_raw(in_path)
        self.clean(lower_token_filter, upper_token_filter)
        self.write_utilities(distribution_type, split_factor, density_factor, spread_factor, nmix)
        self.write_ngram_queries(ngram, nqueries, minmatches, start)
    
class AmazonLoader(DataLoader):
    def __init__(self, out_path, dset_name):
        super().__init__(out_path, dset_name)
    
    def load_raw(self, in_path, config=['reviewText'], verbose=False):
        self.corpus = []
        line_errors = []
        with gzip.open(in_path) as f:
            for i, l in enumerate(f):
                d = json.loads(l.strip())
                try:
                    s = " ".join([d[c] for c in config])
                except:
                    line_errors.append(i)
                    continue
                
                self.corpus.append(s)
        if verbose:
            print("Number of line errors: {}".format(len(line_errors)))
            print("Line errors: {}".format(line_errors))
        

if __name__ == "__main__":
    pass
    # adl = AmazonLoader('data/processed', 'amazon_video_games')
    # adl.make_pipeline('data/raw/Video_Games_5.json.gz', 7, 250, 25)

    # c = adl.read_corpus()
    # u = adl.read_utilities()
    # q = adl.read_queries()

    # assert c == adl.corpus
    # assert (u == adl.utilities).all()
    # assert q == adl.queries

    # adl = AmazonLoader('data/processed', 'amazon_industrial_scientific')
    # adl.make_pipeline('data/raw/Industrial_and_Scientific_5.json.gz', 3, 100, 10, lower_token_filter=3, upper_token_filter=100)
    
    # adl = AmazonLoader('data/processed', 'amazon_musical_instruments')
    # adl.make_pipeline('data/raw/Musical_Instruments_5.json.gz', 5, 10, 25)