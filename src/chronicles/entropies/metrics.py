"""
Relative entropy measures
"""

import numpy as np
from scipy import stats


def kld(p, q):
    """ KL-divergence for two probability distributions
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, (p-q) * np.log2(p / q), 0))


def jsd(p, q, base=2):
    '''Pairwise Jensen-Shannon Divergence for two probability distributions  
    '''
    # convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    # normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return stats.entropy(p, m, base=base)/2. + stats.entropy(q, m, base=base)/2.


def cosine_distance(p, q):
    '''Cosine distance for two vectors
    '''

    p, q = np.asarray(p), np.asarray(q)

    dot_prod = np.dot(p, q)
    magnitude = np.sqrt(p.dot(p)) * np.sqrt(q.dot(q))
    cos_sim = dot_prod / magnitude
    cos_dist = 1 - cos_sim

    return cos_dist
