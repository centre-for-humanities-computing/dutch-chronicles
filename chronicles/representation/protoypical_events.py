'''
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from entropies.metrics import jsd, kld
from document_vectors import RepresentationHandler


# %%
class PrototypeHandler(RepresentationHandler):
    def __init__(self, model, primitives) -> None:
        self.doc_id2vectoridx = model.doc_id2index
        self.doc_id2primidx = {doc['id']: i for i,
                               doc in enumerate(primitives)}

        self.primitives = primitives
        self.model = model
        self.modeldv = model.model.dv

        self.n_topics = model.get_num_topics()

    @staticmethod
    def get_2d_projection(vectors):
        return PCA(n_components=2).fit_transform(vectors)

    def by_avg_distance(self, doc_ids, doc_rank=0, metric='cosine'):

        vectors = self.find_doc_vectors(doc_ids)
        
        # calc pariwise distances
        d = pairwise_distances(vectors, metric=metric)
        # mean distance to other docs
        avg_d = np.mean(d, 0)
        # std of mean distance to other docs
        std_d = np.std(d, 0)

        # index of document at desired rank
        avg_d_argsort = np.argsort(avg_d)
        doc_idx = int(np.argwhere(avg_d_argsort==doc_rank))

        # get id of prototypical doc
        prototype_doc_id = doc_ids[doc_idx]
        uncertainity = std_d[doc_idx]

        return prototype_doc_id, uncertainity

    def by_distance_to_centroid(self, doc_ids, doc_rank=0):

        vectors = self.find_doc_vectors(doc_ids)
        vecs_2d = self.get_2d_projection(vectors)

        km = KMeans(n_clusters=1)
        km.fit(vecs_2d)

        centroid = km.cluster_centers_
        d_centroid = pairwise_distances(
            X=centroid,
            Y=vecs_2d
        )

        # index of document at desired rank 
        d_centroid_argsort = np.argsort(d_centroid)[0]
        doc_idx = int(np.argwhere(d_centroid_argsort==doc_rank))

        # get id of prototypical doc
        prototype_doc_id = doc_ids[doc_idx]
        uncertainity = km.inertia_

        return prototype_doc_id, uncertainity

    def by_relative_entropy(self, doc_ids, doc_rank=0):

        vectors = self.find_doc_vectors(doc_ids)
        # normalize vectors to be probability distributions
        vectors_prob = np.divide(vectors, vectors.sum())

        # option 1: pairwise relative entropy
        d = pairwise_distances(vectors_prob, metric=jsd)
        # option 2: jsd(doc | avg jsd of the rest)

        pass
