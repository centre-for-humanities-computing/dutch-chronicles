'''
Preproblem
----------
Documents are cosine similarities to topics.
Why not just take their embedding?
[DONE]


Problem statement
-----------------
We have vectors representing text documents. 
They are coordinates in linear space (much like what you get from word2vec).

Each document happened on a day, so our analysis has a daily resolution now. 
But, we want to be able to represent a larger chunk of time, e.g. a week with a single datapoint.
To do this, we decided to group documents that are close to each other in time into timebins -> each timebin contains multiple documents. 
Though not every timebin contains the same ammount of documents! 

A timebin will be represented by its "prototypical document": a single vector (document) that is most representative of the whole timebin (multiple vectors).
In the end we also need a function that extracts the n-th most representative vector from the timebin (not only the top one).

In cases where it makes sense, we also want to keep track of the uncertainity of the representation = how well does a prototypical document represent its timebin?


Solution 1: Spatial distance
----------------------------
I already did this.
Calculates pair-wise distances between vectors using some geometric measure (cosine distance, euclidean, etc.). 
Most representative vector = one with lowest avg distance to all other vectors in a timebin.
Uncertainity = standard deviation of distances from the most representative vector to the others 


Solution 2: Networks
--------------------

1) figure out how to convert the vectors into a network 
    1a) so documents are nodes connected by edges
    1b) possible solution: add nodes one at a time. Connect each new node to the most similar existing node. Keep track of distance (could be used as weight of the edge)
        - pros: network with desirable properitiers – not all nodes are connected
        - cons: unstable. Could get different networks depending on which node you start with.
    
    1c) possible solution 2: add all nodes, connect all. Use distance between vectors as weights

2) how to rank nodes by representativeness?
    2a) perhaps using network measures – eigen centrality? Avg distance to other nodes? Avg path length?

3) how to track uncertainity?
    3a) ???


Solution 3: Entropy
-------------------

1) pretend that the vectors are probability distributions 
    1a) normalize, so that sum(vector) == 1

2) use an information-theoretic measure to measure difference between vectors
    2b) Jensen-Shanon Divergence would be an obvious choice 
    2c) interpretation: given we expect vector P to happen, how suprising is vector Q?

3) rank vectors by representativeness
    3a) lowest avg relative entropy (least surprising) = most represenative?

4) track uncertainity: standard deviation of relative entropies?


Solution 4: model based
-----------------------

1) use a probabilistic model to model a timebin
    1a) ??? 

2) use model to predict each vector, track prediction error for each vector

3) rank vectors by representativeness
    3a) vector with the lowest prediction error = most representative?

4) track uncertanity: just take prediction error


Data format
-----------
List[dict], where dict is a timebin.

dict has following keys:

    timebin : int (may change, but will be single element)
    dates : List[str]
    call_nrs : List[str]
    ids : List[int]
    vectors : List[List[float]]

'''


'''
In this script (old section)
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
    '../../lda/representation_final_50.npy',
    allow_pickle=True
    )

# primitives
path_primitives_daily = '../../data/primitives_220331/primitives_corrected_daily.ndjson'
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
with open('/Users/au582299/Repositories/dutch-chronicles/data/primitives_220331/events_repre.ndjson', 'w') as fout:
    ndjson.dump(sorted_prim, fout)

