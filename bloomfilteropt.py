import cvxpy as cp
from abc import ABC, abstractmethod
import numpy as np 


class RoundingScheme(ABC):
    @abstractmethod
    def round(self):
        pass

class FloorRounding(RoundingScheme):
    def round(self, opt):
        return np.floor(opt).astype(int)
    
class GreedyPairing(RoundingScheme):
    def round(self, opt, p, m_prev, eps=0.1):
        N = p.shape[0]
        sort_idxs = np.argsort(p)[::-1]
        p_sorted = p[sort_idxs]
        opt_sorted = opt[sort_idxs]
        ceil_gap = np.ceil(opt_sorted) - opt_sorted
        floor_gap = opt_sorted - np.floor(opt_sorted)
        for i in range(N//2+1):
            for j in range(N-1, N//2, -1):
                gap = floor_gap[j] - ceil_gap[i]
                if p_sorted[j] != -1 and gap < eps and gap > 0:
                    p_sorted[i] = -1
                    p_sorted[j] = -1
                    opt_sorted[i] = np.ceil(opt_sorted[i])
                    opt_sorted[j] = np.floor(opt_sorted[j])
                    break
        mask = np.where(p_sorted != -1)
        opt_sorted[mask] = np.floor(opt_sorted[mask]) 
        opt[sort_idxs] = opt_sorted
        return opt.astype(int)
            
    
class Optimizer(ABC):
    @abstractmethod
    def optimize(self):
        pass

class ConvexJensen(Optimizer):
    def optimize(self, B, p, m_og, n, k, equality_constraint=True, cval='standard', m_prev=None):
        if cval == 'standard':
            c = (1-(1-1/m_og)**(k*n))
        elif cval == 'asymptotic':
            c = 1-np.exp(-k*n/m_og)
        elif cval == 'linearized':
            c = 0.618
        else:
            raise ValueError('cval must be "standard", "asymptotic", or "linearized"')

        N = p.shape[0]
        m_curr = cp.Variable(N)

        if cval == 'linearized':
            objective = cp.Minimize(p @ cp.exp(cp.multiply(np.log(c), cp.multiply(m_curr, 1/n))))
        else:
            objective = cp.Minimize(p @ cp.exp(cp.multiply(np.log(c), cp.multiply(k, cp.multiply(m_curr, 1/m_og)))))
        
        constraints = [m_curr >= 0]

        if m_prev is not None:
            constraints.append(m_curr <= m_prev)
        else:
            constraints.append(m_curr <= m_og)
        
        if equality_constraint:
            constraints.append(cp.sum(m_curr) == B)
        else:
            constraints.append(cp.sum(m_curr) <= B)

        # result = cp.Problem(objective, constraints).solve(verbose=True)
        result = cp.Problem(objective, constraints).solve(verbose=False)
        return m_curr.value
