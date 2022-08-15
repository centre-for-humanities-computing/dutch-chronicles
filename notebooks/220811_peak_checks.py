# %%
import sys
sys.path.append('../chronicles/')

import ndjson
import pandas as pd
import numpy as np

from umap import UMAP
from top2vec import Top2Vec

import seaborn as sns
import matplotlib.pyplot as plt

from entropies import InfoDynamics, jsd
from entropies.afa import adaptive_filter


# %%
TOP2VEC_PATH = "../models/top2vec/top2vecmodel_220504"
PRIMITIVES_PATH = "../data/primitives_220503/primitives_corrected_daily.ndjson"

model = Top2Vec.load(TOP2VEC_PATH)

with open(PRIMITIVES_PATH) as fin:
    primitives = ndjson.load(fin)

# %%
proto_path_100 = '/Users/au582299/Repositories/dutch-chronicles/models/prototype_docs_rank100.ndjson'
proto_path_0 = '/Users/au582299/Repositories/dutch-chronicles/models/prototype_docs_rank0.ndjson'

with open(proto_path_0) as fin:
    prot0 = ndjson.load(fin)
    prot0_df = pd.DataFrame(prot0)

with open(proto_path_100) as fin:
    prot100 = ndjson.load(fin)
    prot100_df = pd.DataFrame(prot100)

# %%
# set up indexing dicts
doc_id2vectoridx = model.doc_id2index
doc_id2primidx = {doc['id']: i for i, doc in enumerate(primitives)}

# vector object
modeldv = model.model.dv

def find_vec(doc_id):
    return modeldv[
        doc_id2vectoridx[doc_id]
        ]

# least representative vectors
vectors100 = [find_vec(doc_id) for doc_id in prot100_df['id'].tolist()]
vectors100 = np.vstack(vectors100)

# most representative vectors
vectors0 = [find_vec(doc_id) for doc_id in prot0_df['id'].tolist()]
vectors0 = np.vstack(vectors0)

# %%
# projection
vectors0_2d = UMAP(random_state=42).fit_transform(X=vectors0)
vectors100_2d = UMAP(random_state=42).fit_transform(X=vectors100)

# %%
# assign back
prot0_df['X'] = vectors0_2d[:, 0]
prot0_df['Y'] = vectors0_2d[:, 1]
prot100_df['X'] = vectors100_2d[:, 0]
prot100_df['Y'] = vectors100_2d[:, 1]

# %%
# sign peaks
prot0_df['year'] = [int(date[0:4]) for date in prot0_df['clean_date'].tolist()]
prot100_df['year'] = [int(date[0:4]) for date in prot100_df['clean_date'].tolist()]

peak_years = [1567, 1568, 1661, 1662, 1663, 1664, 1665, 1746, 1747, 1788, 1789]
prot0_df['in_peak'] = [True if year in peak_years else False for year in prot0_df['year'].tolist()]
prot100_df['in_peak'] = [True if year in peak_years else False for year in prot100_df['year'].tolist()]


# %%
# doc2vec viz: prot0
sns.scatterplot(prot0_df['X'], prot0_df['Y'], hue=prot0_df['in_peak'], alpha=0.1)
plt.title('most representative primitives')

# %%
# doc2vec viz: prot1
sns.scatterplot(prot100_df['X'], prot100_df['Y'], hue=prot100_df['in_peak'], alpha=0.1)
plt.title('last representative primitives')


# %%
# signal without the activation
vectors0_norm = []
for row in vectors0:
    row = row + 20
    norm = row / row.sum()
    vectors0_norm.append(norm)


# %%
im_vectors = InfoDynamics(
        data=vectors0_norm,
        window=30,
        time=None,
        normalize=False
    )

im_vectors.resonance(meas=jsd)

# %%
# dates
import matplotlib.dates as mdates
from datetime import datetime

dates = []
for date in prot0_df['clean_date'].tolist():
    try:
        cl_date = datetime.strptime(date, '%Y-%M-%d')
    except:
        cl_date = datetime.strptime('1900-01-01', '%Y-%M-%d')
    dates.append(cl_date)

# %%
fig, ax = plt.subplots()
plt.plot(dates, im_vectors.nsignal, '.', alpha=0.1)
plt.plot(dates, adaptive_filter(im_vectors.nsignal, span=64), c='red')

ax.format_xdata = mdates.DateFormatter('%Y-%M-%d')

# %%
# share of sources
df_sources = prot0_df.groupby(['call_nr', 'year']).size().reset_index().rename(columns={0:'n'})
df_sources

sns.scatterplot(
    x=df_sources['year'],
    y=np.log2(df_sources['n']),
    hue=df_sources['call_nr'],
    legend=False,
    # alpha=0.2
)
# %%
# number of unique chroniclers per year
# df_calls_per_year = prot0_df.groupby('call_nr')['year'].size()
df_call_overview = prot0_df.groupby(['year', 'call_nr'])['call_nr'].size().to_frame().rename(columns={'call_nr': 'n_docs'}).reset_index()
unique_calls_p_year = df_call_overview.groupby('year')['call_nr'].count().reset_index().rename(columns={'call_nr': 'n_calls'})

# %%
sns.lineplot(
    x=unique_calls_p_year['year'],
    y=unique_calls_p_year['n_calls']
)
plt.ylabel('number of unique sources per year')
plt.title('novelty somewhat coincides with author diversity')
plt.savefig('../fig/unique_sources_over_time.png')

# %%
# sample vector
plt.plot(vectors0[0], '.')
plt.xlabel('dimension')
plt.ylabel('value')
plt.title('sample document vector')
plt.savefig('../fig/sample_vector.png')

# %%
