'''
Relative entropy measures
    taken from https://github.com/centre-for-humanities-computing/newsFluxus/blob/master/src/tekisuto/metrics/entropies.py
    commit 1fb16bc91b99716f52b16100cede99177ac75f55
'''

import numpy as np
from scipy import stats


def kld(p, q):
    """ KL-divergence for two probability distributions
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, (p-q) * np.log10(p / q), 0))


def jsd(p, q, base=np.e):
    '''Pairwise Jensen-Shannon Divergence for two probability distributions  
    '''
    # convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    # normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return stats.entropy(p, m, base=base)/2. + stats.entropy(q, m, base=base)/2.
