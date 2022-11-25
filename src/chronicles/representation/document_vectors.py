"""
Class for extracting prototypes and operations with document representations.

Requires: document representations acquired with Top2Vec & document IDs allocated in parsing
(chronicles.parser.give_ids)
"""

from Typing import List
import numpy as np
from wasabi import msg

from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from top2vec import Top2Vec


class RepresentationHandler:
    """
    Parameters
    ----------
    model : top2vec.Top2Vec
        fitted Top2Vec model
    primitives : List[dict]
        input documents (stored in new line json).
        Every document is a dict.

    tolerate_invalid_ids : bool
        If Flase, only document IDs found in self.model will be returned, the rest will be
        ignored & added to self.missing_docid_history.
        If True, invalid IDs will cause an error.

    Attributes
    ----------
    doc_id2vectoridx : dict
        mapping of document id to model index
    doc_id2primidx: dict
        mapping of document id to primitive index.
        Automatically generated.
    primitives: List[dict]
        input documents
    model : top2vec.Top2Vec
        fitted Top2Vec model
    n_topics : int
        number of topics found in self.model
    missing_docid_history : List
        list of IDs that were not found in self.model.
    """

    def __init__(self, model: Top2Vec, primitives: List[dict], tolerate_invalid_ids=False):

        self.doc_id2vectoridx = model.doc_id2index
        self.doc_id2primidx = {doc['id']: i for i,
                               doc in enumerate(primitives)}

        self.primitives = primitives
        self.model = model
        self.modeldv = model.model.dv

        self.n_topics = model.get_num_topics()
        self._coerce_key_errors = tolerate_invalid_ids
        self.missing_docid_history = []

    @staticmethod
    def _get_2d_projection(vectors: np.array) -> np.array:
        return PCA(n_components=2).fit_transform(vectors)

    def _warning_coerce_key_errors(self):
        if self._coerce_key_errors:
            msg.warn(
                'Warning: RepresentationHandler.greedy set to False. ' +
                'Invalid doc_ids will be ignored. ' +
                'Please see RepresentationHandler.missing_docid_history to see' +
                'which ones were ignored'
            )
        else:
            pass

    def find_doc_vector(self, doc_id: int) -> np.array:
        """
        Given doc_id, find it's Top2Vec representation
        """

        try:
            return self.modeldv[
                self.doc_id2vectoridx[doc_id]
            ]

        except KeyError as error_doc_id_not_found:
            if self._coerce_key_errors:
                # keep track of invalid doc_ids
                self.missing_docid_history.append(doc_id)
                # skip doc_id
                pass
            else:
                raise KeyError(f'document {doc_id} not found in model.')

    def find_doc_vectors(self, doc_ids: List[int]) -> np.array:
        """
        Iterate over document IDs to find multiple Top2Vec representations
        """

        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]

        # iterate find_doc_vector
        vectors = [self.find_doc_vector(doc_id) for doc_id in doc_ids]
        # list of arrays -> array of shape (docs, topics)
        vectors = np.vstack(vectors)

        # warn, if invalid doc_ids are tolerated
        self._warning_coerce_key_errors()

        return vectors

    def find_document(self, doc_id: int) -> dict:
        """
        Given doc_id, find a document (observation with metadata)
        """

        try:
            return self.primitives[
                self.doc_id2primidx[doc_id]
            ]

        except KeyError as error_doc_id_not_found:
            if self._coerce_key_errors:
                # keep track of invalid doc_ids
                self.missing_docid_history.append(doc_id)
                # skip doc_id
                pass
            else:
                raise KeyError(f'document {doc_id} not found in primitives.')

    def find_documents(self, doc_ids: List[int]):
        """
        Iterate over document IDs to find multiple documents
        """

        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]

        documents = [self.find_document(doc_id) for doc_id in doc_ids]

        # warn, if invalid doc_ids are tolerated
        self._warning_coerce_key_errors()

        return documents

    def filter_invalid_doc_ids(self, doc_ids: List[int]) -> List[int]:
        """
        Try to find document in both primitives and model.

        Args:
            doc_ids : document IDs to check

        Returns:
            validated_doc_ids : document IDs present in both self.primitives & self.model
        """

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

    def find_doc_cossim(self, doc_ids: List[int], n_topics=None) -> np.array:
        """
        Given doc_ids, find the cosine similarities between document vectors and topic centroids.

        Args:
            doc_ids: document IDs to find representations for
            n_topics: desired number of topic centroids (aka representation dimensionality).
                      If specified, hdbscan implemented in Top2Vec will be used to merge topics.
                      By default, None, meaning follow the number of topics the model was originally fitted on.

        Returns:
            representations: cosine similarities of shape (doc_ids, n_topics)
        """

        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]

        # filter invalid doc_ids if approach is greedy
        if not self._coerce_key_errors:
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
        self._warning_coerce_key_errors()

        return representations

    def get_primitives_and_vectors(self, doc_ids: List[int]) -> List[dict]:
        """
        Get a subset of primitives enriched with *vector* representation of documents

        Returns:
            event_selection: records of documents with an extra key-value pair 'doc_vec'
        """

        event_selection = []
        for doc_id in doc_ids:
            record = self.find_document(doc_id)
            vector = self.find_doc_vector(doc_id)
            vector = vector.tolist()

            record.update({'doc_vec': vector})
            event_selection.append(record)

        return event_selection

    def get_primitives_and_cossims(self, doc_ids: List[int], n_topics=None) -> List[dict]:
        """
        Get a subset of primitives enriched with *cosine similarity* representation of documents

        Args:
            doc_ids: document IDs to find representations for
            n_topics: desired number of topic centroids (aka representation dimensionality).
                      If specified, hdbscan implemented in Top2Vec will be used to merge topics.
                      By default, None, meaning follow the number of topics the model was originally fitted on.

        Returns:
            event_selection: records of documents with an extra key-value pair 'doc_cossim'
        """

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

    def prototypes_by_avg_distance(self, doc_ids, doc_rank=0, metric='cosine', reduce_dim=False):
        """
        Find a prototype document in a subset.
        Prototype is the document with the lowest average distance to other documents in the subset.

        Args:
            doc_ids: List[int]
            doc_rank: desired document rank of a prototype (0 = document with the lowest avg distance)
            metric: distance metric to use for calculating distances
            reduce_dim: find prototypes using a 2D representation of the subset?

        Returns:
            prototype_doc_id: int
                ID of prototype document
            uncertainty: float
                standard deviation of the prototypical document's distances to other docs.
        """

        vectors = self.find_doc_vectors(doc_ids)

        if reduce_dim:
            vectors = self._get_2d_projection(vectors)

        # calc pariwise distances
        d = pairwise_distances(vectors, metric=metric)
        # mean distance to other docs
        avg_d = np.mean(d, 0)
        # std of mean distance to other docs
        std_d = np.std(d, 0)

        # index of document at desired rank
        avg_d_argsort = np.argsort(avg_d)
        doc_idx = int(avg_d_argsort[doc_rank])

        # get id of prototypical doc
        prototype_doc_id = doc_ids[doc_idx]
        uncertainty = std_d[doc_idx]

        return prototype_doc_id, uncertainty

    def prototypes_by_distance_to_centroid(self, doc_ids, doc_rank=0):
        """
        Find a prototype document in a subset.
        Prototype is the document with the lowest distance to subset centroid found using KMeans.

        Args:
            doc_ids: List[int]
            doc_rank: desired document rank of a prototype (0 = document with the lowest distance to centroid)

        Returns:
            prototype_doc_id: int
                ID of prototype document
            uncertainty: float
                standard deviation of the prototypical document's distances to other docs.
        """

        vectors = self.find_doc_vectors(doc_ids)
        vecs_2d = self._get_2d_projection(vectors)

        km = KMeans(n_clusters=1)
        km.fit(vecs_2d)

        centroid = km.cluster_centers_
        d_centroid = pairwise_distances(
            X=centroid,
            Y=vecs_2d
        )

        # index of document at desired rank
        d_centroid_argsort = np.argsort(d_centroid)[0]
        doc_idx = int(d_centroid_argsort[doc_rank])

        # get id of prototypical doc
        prototype_doc_id = doc_ids[doc_idx]
        uncertainty = km.inertia_

        return prototype_doc_id, uncertainty
