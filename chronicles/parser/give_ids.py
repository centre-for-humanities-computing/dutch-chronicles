# %%
import os
import ndjson

with open('/Users/au582299/Repositories/dutch-chronicles/data/primitives_220329/primitives_annotated.ndjson') as fin:
    prim_anno = ndjson.load(fin)

with open('/Users/au582299/Repositories/dutch-chronicles/data/primitives_220329/primitives_corrected.ndjson') as fin:
    prim_corr = ndjson.load(fin)


# %%
for doc, i in zip(prim_anno, range(len(prim_anno))):
    doc['id'] = i

prim_corr_texts = [doc['text'] for doc in prim_corr]
prim_corr_call_nrs = [doc['call_nr'] for doc in prim_corr]
prim_corr_fix = [doc for doc in prim_anno if doc['text'] in prim_corr_texts and doc['call_nr'] in prim_corr_call_nrs]

# %%
with open('/Users/au582299/Repositories/dutch-chronicles/data/primitives_220329/primitives_annotated.ndjson', 'w') as fout:
    ndjson.dump(prim_anno, fout)

with open('/Users/au582299/Repositories/dutch-chronicles/data/primitives_220329/primitives_corrected.ndjson', 'w') as fout:
    ndjson.dump(prim_corr_fix, fout)

# %%
