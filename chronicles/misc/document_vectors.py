# %%
import numpy as np
import ndjson

from top2vec import Top2Vec

# %%
# get model
model = Top2Vec.load("/work/62138/models/top2vec/top2vecmodel_50_2")

# get events
with open('/work/62138/corpus/primitives_220331/primitives_annotated.ndjson') as fin:
    primitives = ndjson.load(fin)


# %%
# dictionary from doc_id to Top2Vec (also Doc2Vec hopefully) document index
doc_id2vectoridx = model.doc_id2index

# dictionary from doc_id to primitives document index
doc_id2primidx = {doc['id']: i for i, doc in enumerate(primitives)}


# %%
test_vector = model.model.dv[doc_id2vectoridx[496]]
test_record = primitives[doc_id2primidx[496]]


# %%
interesting_doc_ids = [121432, 21451, 30035, 119452, 82543]

event_selection = []
for doc_id in interesting_doc_ids:
    record = primitives[doc_id2primidx[doc_id]]
    vector = model.model.dv[doc_id2vectoridx[doc_id]]
    vector = vector.tolist()

    record.update({'doc_vec': vector})
    event_selection.append(record)


# %%
# event selection
with open('/work/62138/corpus/primitives_220331/220405_selected_events.ndjson', 'w') as fout:
    ndjson.dump(event_selection, fout)



# %%
+