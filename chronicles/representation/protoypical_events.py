'''


Course
------
a) Get a bin of representations
b) Calculate distances to other docs
    - with different distance metrics
c) Find prototypical document
    - central?
    - least avg distance?
d) Represent
    - Timebine = prototype document 
    - Take uncertainity (avg distance to other docs / how well does prototype represent?)

'''
# %%
import numpy as np
from sklearn.metrics import pairwise_distances


# %%
def prototype_avg_d(representations, metric='cosine'):
    
    d = pairwise_distances(representations, metric=metric)

    # mean distance to other docs
    avg_d = np.mean(d, 0)
    # std of mean distance to other docs
    std_d = np.std(d, 0)

    # index of minimum avg distance document
    id_min_d = np.argmin(avg_d)

    # get represtention of prototypical doc
    prototype_doc = representations[id_min_d]
    uncertainity = std_d[id_min_d]

    return prototype_doc, uncertainity

