# %%
import numpy as np
import ndjson

from top2vec import Top2Vec

# # %%
# # get model
# model = Top2Vec.load("/work/62138/models/top2vec/top2vecmodel_50_2")

# # get events
# with open('/work/62138/corpus/primitives_220331/primitives_annotated.ndjson') as fin:
#     primitives = ndjson.load(fin)


# # %%
# # dictionary from doc_id to Top2Vec (also Doc2Vec hopefully) document index
# doc_id2vectoridx = model.doc_id2index

# # dictionary from doc_id to primitives document index
# doc_id2primidx = {doc['id']: i for i, doc in enumerate(primitives)}


# # %%
# test_vector = model.model.dv[doc_id2vectoridx[496]]
# test_record = primitives[doc_id2primidx[496]]


# # %%
# interesting_doc_ids = [121432, 21451, 30035, 119452, 82543]

# event_selection = []
# for doc_id in interesting_doc_ids:
#     record = primitives[doc_id2primidx[doc_id]]
#     vector = model.model.dv[doc_id2vectoridx[doc_id]]
#     vector = vector.tolist()

#     record.update({'doc_vec': vector})
#     event_selection.append(record)


# # %%
# # event selection
# with open('/work/62138/corpus/primitives_220331/220405_selected_events.ndjson', 'w') as fout:
#     ndjson.dump(event_selection, fout)


# %%
class RepresentationHandler:
    def __init__(self, model, primitives):

        self.doc_id2vectoridx = model.doc_id2index
        self.doc_id2primidx = {doc['id']: i for i,
                               doc in enumerate(primitives)}

        self.primitives = primitives
        self.model = model
        self.modeldv = model.model.dv

        self.n_topics = model.get_num_topics()

    def find_doc_vector(self, doc_id):

        return self.modeldv[
            self.doc_id2vectoridx[doc_id]
        ]

    def find_doc_vectors(self, doc_ids):

        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]

        # iterate find_doc_vector
        vectors = [self.find_doc_vector(doc_id) for doc_id in doc_ids]
        # list of arrays -> array of shape (docs, topics)
        vectors = np.vstack(vectors)

        return vectors

    def find_document(self, doc_id):
        return self.primitives[
            self.doc_id2primidx[doc_id]
        ]

    def find_documents(self, doc_ids):

        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]
        
        documents = [self.find_document(doc_id) for doc_id in doc_ids]
        return documents

    def find_doc_cossim(self, doc_ids, n_topics):

        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]

        # get document representations
        if n_topics == self.n_topics:
            tp_ids, tp_vals, tp_words, word_scores = model.get_documents_topics(
                doc_ids=doc_ids, reduced=False, num_topics=n_topics)
        else:
            tp_ids, tp_vals, tp_words, word_scores = model.get_documents_topics(
                doc_ids=doc_ids, reduced=True, num_topics=n_topics)

        # sort by topic number
        representations = []
        for doc_topic_ids, doc_topic_vals in zip(tp_ids, tp_vals):
            sorted_topic_vals = doc_topic_vals[doc_topic_ids]
            representations.append(sorted_topic_vals)

        # list of arrays -> array of shape (docs, topics)
        representations = np.vstack(representations)

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

    def get_primitives_and_cossims(self, doc_ids):

        event_selection = []
        for doc_id in doc_ids:
            record = self.find_document(doc_id)
            cossim = self.find_doc_cossim(doc_id)
            cossim = cossim.tolist()

            record.update({'doc_cossim': cossim})
            event_selection.append(record)
        
        return event_selection
