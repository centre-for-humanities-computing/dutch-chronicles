'''
Test for known bugs caused by wrong python version
'''

import numpy as np


def test_array_indexing():
    '''
    In python 3.7, 
    `sample[np.argsort(idx)]` is required for correct sort. 
    '''
    sample = np.array([1.93, 2.93, 3.93])
    idx = np.array([2, 0, 1])

    rearanged_sample = sample[idx]
    assert rearanged_sample[0] == 2.93


def test_dictionary_iteration():
    # TODO
    pass