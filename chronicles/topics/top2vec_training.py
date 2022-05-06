# %%
import ndjson
import glob
import numpy as np
import pandas as pd
import top2vec
from top2vec import Top2Vec 

# %%

with open('/work/62138/corpus/primitives_220503/primitives_annotated.ndjson') as f:
    data = ndjson.load(f)

# %%

# filter events longer than 50 characters
data_filtered = []
for item in data:
    if len(item['text']) > 50:
        data_filtered.append(item)

# prepare data for top2vec, create two lists of texts and document_ids
corpus = []
document_ids = []

for key in data_filtered:
    text = key['text']
    id = key['id']
    document_ids.append(id)
    corpus.append(text)

# %%

# train model
model = Top2Vec(corpus, document_ids = document_ids)

# %% hierarchically reduce model to 100 topics

model.hierarchical_topic_reduction(100)
# %%

# save model
model.save('/work/62138/models/top2vec/top2vecmodel_220504')
