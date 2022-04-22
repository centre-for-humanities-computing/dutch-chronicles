''' 
'''
import numpy as np
import ndjson
from wasabi import msg

from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from top2vec import Top2Vec

from entropies.metrics import jsd, kld


class RepresentationHandler:
    def __init__(self, model, primitives, tolerate_invalid_ids=False):

        self.doc_id2vectoridx = model.doc_id2index
        self.doc_id2primidx = {doc['id']: i for i,
                               doc in enumerate(primitives)}

        self.primitives = primitives
        self.model = model
        self.modeldv = model.model.dv

        self.n_topics = model.get_num_topics()
        self.coerce_key_errors = tolerate_invalid_ids
        self.missing_docid_history = []

    def _warning_coerce_key_errors(self):
        if self.coerce_key_errors:
            msg.warn(
                'Warning: RepresentationHandler.greedy set to False. ' +
                'Invalid doc_ids will be ignored. ' +
                'Please see RepresentationHandler.missing_docid_history to see' +
                'which ones were ignored'
            )
        else:
            pass

    def find_doc_vector(self, doc_id):

        try:
            return self.modeldv[
                self.doc_id2vectoridx[doc_id]
            ]

        except KeyError as error_doc_id_not_found:
            if self.coerce_key_errors:
                # keep track of invalid doc_ids
                self.missing_docid_history.append(doc_id)
                # skip doc_id
                pass
            else:
                raise KeyError(f'document {doc_id} not found in model.')

    def find_doc_vectors(self, doc_ids):

        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]

        # iterate find_doc_vector
        vectors = [self.find_doc_vector(doc_id) for doc_id in doc_ids]
        # list of arrays -> array of shape (docs, topics)
        vectors = np.vstack(vectors)

        # warn, if invalid doc_ids are tolerated
        self._warning_coerce_key_errors()

        return vectors

    def find_document(self, doc_id):

        try:
            return self.primitives[
                self.doc_id2primidx[doc_id]
            ]

        except KeyError as error_doc_id_not_found:
            if self.coerce_key_errors:
                # keep track of invalid doc_ids
                self.missing_docid_history.append(doc_id)
                # skip doc_id
                pass
            else:
                raise KeyError(f'document {doc_id} not found in primitives.')

    def find_documents(self, doc_ids):

        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]

        documents = [self.find_document(doc_id) for doc_id in doc_ids]

        # warn, if invalid doc_ids are tolerated
        self._warning_coerce_key_errors()

        return documents

    def filter_invalid_doc_ids(self, doc_ids):
        '''Try to find document in both primitives and model.
        Returns a list of validated doc_ids

        Parameters
        ----------
        doc_ids : List[int]
            doc_ids to check

        Returns
        -------
        List[int]
            validated doc_ids
        '''

        validated_doc_ids = []
        for doc_id in doc_ids:
            try:
                primidx = self.doc_id2primidx[doc_id]
                vectoridx = self.doc_id2vectoridx[doc_id]
                validated_doc_ids.append(doc_id)
            except KeyError:
                self.missing_docid_history.append(doc_id)
                pass

        return validated_doc_ids

    def find_doc_cossim(self, doc_ids, n_topics=None):

        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]

        # filter invalid doc_ids if approach is greedy
        if not self.coerce_key_errors:
            doc_ids = self.filter_invalid_doc_ids(doc_ids)

        # by default, get n_topics same as the model
        if not n_topics:
            n_topics = self.n_topics

        # get document representations
        if n_topics == self.n_topics:
            tp_ids, tp_vals, tp_words, word_scores = self.model.get_documents_topics(
                doc_ids=doc_ids, reduced=False, num_topics=n_topics)
        else:
            tp_ids, tp_vals, tp_words, word_scores = self.model.get_documents_topics(
                doc_ids=doc_ids, reduced=True, num_topics=n_topics)

        # sort by topic number
        representations = []
        for doc_topic_ids, doc_topic_vals in zip(tp_ids, tp_vals):
            sorted_topic_vals = doc_topic_vals[doc_topic_ids]
            representations.append(sorted_topic_vals)

        # list of arrays -> array of shape (docs, topics)
        representations = np.vstack(representations)

        # warn, if doc_id search is not greedy
        self._warning_not_greedy()

        return representations

    def get_primitives_and_vectors(self, doc_ids):

        event_selection = []
        for doc_id in doc_ids:
            record = self.find_document(doc_id)
            vector = self.find_doc_vector(doc_id)
            vector = vector.tolist()

            record.update({'doc_vec': vector})
            event_selection.append(record)

        return event_selection

    def get_primitives_and_cossims(self, doc_ids, n_topics=None):

        if not n_topics:
            n_topics = self.n_topics

        event_selection = []
        for doc_id in doc_ids:
            record = self.find_document(doc_id)
            cossim = self.find_doc_cossim(doc_id, n_topics)
            cossim = cossim.tolist()

            record.update({'doc_cossim': cossim})
            event_selection.append(record)

        return event_selection

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
        doc_idx = int(np.argwhere(avg_d_argsort == doc_rank))

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
        doc_idx = int(np.argwhere(d_centroid_argsort == doc_rank))

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
