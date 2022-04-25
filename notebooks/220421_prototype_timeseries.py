# %%
import sys
sys.path.append('../chronicles/')

import ndjson
import pandas as pd
from tqdm import tqdm
from top2vec import Top2Vec

from representation import RepresentationHandler
from misc import parse_dates
from entropies import InfoDynamics
from entropies.metrics import jsd, kld, cosine_distance

# %%
# load resources
model = Top2Vec.load("/work/62138/models/top2vec/top2vecmodel_50_2")

with open('/work/62138/corpus/primitives_220331/primitives_annotated_daily.ndjson') as fin:
    primitives = ndjson.load(fin)

# %%
# get subset for the analysis
prims = pd.DataFrame(primitives)
prims = parse_dates(prims['clean_date'], inplace=True, df=prims)

prims = prims.query('year >= 1400 & year <= 1800')
prims = prims.sort_values(by=['year', 'week'])

prims['text_len'] = prims['text'].apply(len)
prims = prims.query('text_len > 50') 

# %%
# weekly doc_id groupings
df_groupings = (prims
    .groupby(['year', 'week'])["id"].apply(list)
    .reset_index()
    .sort_values(by=['year', 'week'])
)

groupings = df_groupings['id'].tolist()

# %%
# get ids of prototypes
rh_weekly = RepresentationHandler(
    model, primitives, tolerate_invalid_ids=False
    )

prototypes_ids = []
prototypes_std = []

for week in tqdm(groupings):
    doc_ids = rh_weekly.filter_invalid_doc_ids(week)

    if doc_ids:

        if len(doc_ids) == 1:
            prot_id = doc_ids[0]
            prot_std = 0

        else:
            prot_id, prot_std = rh_weekly.by_distance_to_centroid(
                doc_ids
            )

        prototypes_ids.append(prot_id)
        prototypes_std.append(prot_std)

prot_vectors = rh_weekly.find_doc_vectors(prototypes_ids)
prot_cossim = rh_weekly.find_doc_cossim(prototypes_ids, n_topics = 100)
prot_docs = rh_weekly.find_documents(prototypes_ids)

# %%
# contextualized cosine distance
im_vectors = InfoDynamics(
    data=prot_vectors,
    window=4,
    time=None,
    normalize=False
)

im_vectors.novelty(meas=cosine_distance)
im_vectors.transience(meas=cosine_distance)
im_vectors.resonance(meas=cosine_distance)

# %%
