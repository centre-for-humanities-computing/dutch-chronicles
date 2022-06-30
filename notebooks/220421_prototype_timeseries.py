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
model = Top2Vec.load("../models/top2vec/top2vecmodel_220504")

with open('../data/primitives_220503/primitives_corrected_daily.ndjson') as fin:
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
            prot_id, prot_std = rh_weekly.by_avg_distance(
                doc_ids,
                metric='cosine'
            )

        prototypes_ids.append(prot_id)
        prototypes_std.append(prot_std)

prot_vectors = rh_weekly.find_doc_vectors(prototypes_ids)
prot_cossim = rh_weekly.find_doc_cossim(prototypes_ids, n_topics = 100)
prot_docs = rh_weekly.find_documents(prototypes_ids)

# %%
# get prims and cossims
prim_cos = rh_weekly.get_primitives_and_cossims((prims['id'].tolist()), n_topics = 100)
# saved as corpus/primitives_220503/events_repre_220503.ndjson

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
# plot
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

nov_fil = gaussian_filter1d(im_vectors.nsignal, sigma=6)
tra_fil = gaussian_filter1d(im_vectors.tsignal, sigma=6)
res_fil = gaussian_filter1d(im_vectors.rsignal, sigma=6)

dates = []
novelty_f = []
transience_f = []
resonance_f = []
std_f = []
for doc, nval, tval, rval, stdval in zip(prot_docs, nov_fil, tra_fil, res_fil, prototypes_std):
    try:
        date = datetime.strptime(doc['clean_date'], '%Y-%m-%d')
        dates.append(date)
        novelty_f.append(nval)
        transience_f.append(tval)
        resonance_f.append(rval)
        std_f.append(stdval)
    except:
        pass

# %%
# novelty
fig, ax = plt.subplots()
sns.lineplot(
    x=dates,
    y=novelty_f,
    ax=ax
)

ax.set_ylim(0.35, 1)

# %%
# transience
fig, ax = plt.subplots()
sns.lineplot(
    x=dates,
    y=transience_f,
    ax=ax
)

ax.set_ylim(0.35, 1)


# %%
# resonance
fig, ax = plt.subplots()
sns.lineplot(
    x=dates,
    y=resonance_f,
    ax=ax
)


# %%
# prototype representativeness uncertainity

sns.lineplot(
    x=dates,
    y=gaussian_filter1d(std_f, sigma=20)
)

# %%
