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

# %%
# add some network shit, 
# so that we can track
# - timeseries going between central nodes
# - timeseries going between secondary to n-ary nodes
# - timeseries going between the most deviating nodes

# %%
import re
import datetime
from tqdm import tqdm
import ndjson
import pandas as pd

# representations
ev = np.load(
    '../../../models/representation_final_50_reduced.npy',
    allow_pickle=True
    )

# primitives
path_primitives_daily = '../../../corpus/primitives_220331/primitives_corrected_daily.ndjson'
with open(path_primitives_daily) as fin:
    primitives_daily = ndjson.load(fin)


# %%
# ensure correct formating
bad_date_docs = []
primitives_daily_date_format = []
for doc in primitives_daily:
    try:
        datetime.datetime.strptime(doc['clean_date'], '%Y-%m-%d')
        primitives_daily_date_format.append(doc)
    except ValueError:
        bad_date_docs.append(doc)

# %%
# sort primitives by date
sorted_prim = sorted(
    primitives_daily_date_format,
    key=lambda t: datetime.datetime.strptime(t['clean_date'], '%Y-%m-%d')
    )


# %%
# append representation to primitives
rep_doc_id = [event[0] for event in ev]
rep_vals =[event[1].tolist() for event in ev]

docs_not_represented = []
for doc in tqdm(sorted_prim):
    try:
        idx = rep_doc_id.index(doc['id'])
        corresponding_rep = rep_vals[idx]
        doc['representation'] = corresponding_rep
    except ValueError:
        sorted_prim.remove(doc)
        docs_not_represented.append(doc)



# %%
with open('../../../corpus/primitives_220331/events_repre_reduced.ndjson', 'w') as fout:
    ndjson.dump(sorted_prim, fout)



# %%
