from abc import ABC, abstractmethod
import numpy as np

class Store(ABC):
    def __init__(self, fname):
        self.fname = fname
    
    @abstractmethod
    def get(self, i):
        pass
    @abstractmethod
    def put(self, i, doc):
        pass

class mmapDocStore(Store):
    def __init__(self, fname, size, maxdocsize):
        super().__init__(fname)

        self.size = size
        self.maxdocsize = maxdocsize
        self.initalized = False
        
    def get(self, i):
        return str(self.fp[i])
    
    def put(self, i, doc):
        if i == 0:
            if not self.initalized:
                self.initalized = True
                self.fp = np.memmap(self.fname, dtype='U{}'.format(self.maxdocsize), mode='w+', shape=(self.size))
        elif i == 1:
            self.fp = np.memmap(self.fname, dtype='U{}'.format(self.maxdocsize), mode='r+', shape=(self.size))
        self.fp[i] = doc

if __name__ == "__main__":
    l1 = ['pear', 'apple', 'strawberry', 'blueberry', \
     'mango', 'grape', 'orange', 'banana', 'kiwi', 'pineapple', 'passionfruit']
    s1 = ','.join(l1)
    print("S1 len:", len(s1))
    l2 = ['bob', 'the', 'builder']
    s2 = ','.join(l2)
    print("S2 len:", len(s2))

    ds = mmapDocStore("unittest.npy", 2, max(len(s1), len(s1)))
    s = [s1, s2]
    for i, s_i in enumerate(s):
        ds.put(i, s_i)

    l1pr = ds.get(0).split(',')
    print(l1pr, len(l1pr))
    l2pr = ds.get(1).split(',')
    print(l2pr, len(l2pr))

