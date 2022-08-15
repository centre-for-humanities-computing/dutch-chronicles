''' 
Combine top2vec model, dataset of primitives.
Calculate novelty at different windows.
Export a dataframe for analysis.
'''

# %%
import sys
sys.path.append('../chronicles/')

import os
from datetime import datetime
import yaml

import ndjson
import numpy as np
import pandas as pd
from tqdm import tqdm
from wasabi import msg
from top2vec import Top2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import seaborn as sns
import matplotlib.pyplot as plt

from representation import RepresentationHandler
from misc import parse_dates
from entropies import InfoDynamics
from entropies.metrics import jsd, kld, cosine_distance
from entropies.afa import adaptive_filter


# %%
# parameters
TOP2VEC_PATH = "../models/top2vec/top2vecmodel_220504"
PRIMITIVES_PATH = "../data/primitives_220503/primitives_corrected_daily.ndjson"

LG_WINDOWS = False
DOC_RANK = 0
PROTOTYPES = True

if PROTOTYPES:
    outdir = f'../models/2208_novelties/novelty_docrank{DOC_RANK}'
else:
    outdir = f'../models/2208_novelties/novelty_noprototype'

# init output folders
if not os.path.exists(outdir):
    os.mkdir(outdir)

# %%
# load resources
model = Top2Vec.load(TOP2VEC_PATH)

with open(PRIMITIVES_PATH) as fin:
    primitives = ndjson.load(fin)

# %%
# parse dates & get metadata of the subset
prims_unfiltered = pd.DataFrame(primitives)
prims_unfiltered = parse_dates(prims_unfiltered['clean_date'], inplace=True, df=prims_unfiltered)

# text length
prims_unfiltered['n_char'] = prims_unfiltered['text'].str.len()
prims_unfiltered.describe()


# %%
# filtering
prims = prims_unfiltered.copy()
# cut extreme years
prims = prims.query('year >= 1500 & year <= 1820')
prims = prims.sort_values(by=['year', 'week'])
# cut very short & very long docs
prims = prims.query('n_char >= 50 & n_char <= 5000')
prims.describe()

''' 
Despite dropping 6100 documents, 
the mean and median number of characters remains almost unchanged after this filtering.
Also the mean and median of document dating remains almost unchanged.
'''

# %%
# daily doc_id groupings


# %%

if PROTOTYPES:

    df_groupings_day = (prims
        .groupby(['year', 'week', 'day'])["id"].apply(list)
        .reset_index()
        .sort_values(by=['year', 'week', 'day'])
    )

    groupings_day = df_groupings_day['id'].tolist()

    rh_daily = RepresentationHandler(
        model, primitives, tolerate_invalid_ids=False
        )

    prototypes_ids = []
    prototypes_std = []

    msg.info('finding prototypes')
    for week in tqdm(groupings_day):
        doc_ids = rh_daily.filter_invalid_doc_ids(week)

        if doc_ids:

            if len(doc_ids) == 1:
                prot_id = doc_ids[0]
                prot_std = 0
            
            elif DOC_RANK >= len(doc_ids):
                prot_id, prot_std = rh_daily.by_avg_distance(
                    doc_ids,
                    metric='cosine',
                    doc_rank=len(doc_ids)-1
                )

            else:
                prot_id, prot_std = rh_daily.by_avg_distance(
                    doc_ids,
                    metric='cosine',
                    doc_rank=DOC_RANK
                )

            prototypes_ids.append(prot_id)
            prototypes_std.append(prot_std)

    msg.info('extracting vectors')
    prot_vectors = rh_daily.find_doc_vectors(prototypes_ids)
    prot_cossim = rh_daily.find_doc_cossim(prototypes_ids, n_topics = 100)
    prot_docs = rh_daily.find_documents(prototypes_ids)

    # add std & dump prots
    [doc.update({'uncertainity': float(std)}) for doc, std in zip(prot_docs, prototypes_std)]
    with open(f'../models/prototype_docs_rank{DOC_RANK}.ndjson', 'w') as fout:
        ndjson.dump(prot_docs, fout)

    msg.good('done (prototypes, vectors)')

else:

    rh_noproto = RepresentationHandler(
        model, primitives, tolerate_invalid_ids=False
        )

    subset_ids = prims['id'].tolist()
    
    prot_vectors = rh_noproto.find_doc_vectors(subset_ids)
    prot_cossim = rh_noproto.find_doc_cossim(subset_ids, n_topics = 100)
    prot_docs = rh_noproto.find_documents(subset_ids)



# %%
# softmax on vectors
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

prot_vectors_norm = np.array([softmax(vec) for vec in prot_vectors])


# %%
# relative entropy experiments
if LG_WINDOWS:
    window_param_grid = [100, 200, 300, 400, 500, 1000]
else:
    window_param_grid = [1, 5, 10, 20, 30, 40, 50]

system_states = []
for w in tqdm(window_param_grid):

    msg.info(f'infodynamics w {w}')
    # initialize infodyn class
    im_vectors = InfoDynamics(
        data=prot_vectors_norm,
        window=w,
        time=None,
        normalize=False
    )

    # calculate with jensen shannon divergence & save results
    im_vectors.fit_save(
        meas=jsd,
        slice_w=True,
        path=f'../models/novelty_rank{DOC_RANK}/daily_w{w}.json'
        )
    
    # track system state at different windows
    # z-scaler and reshape
    zn = StandardScaler().fit_transform(
        im_vectors.nsignal.reshape(-1, 1)
        )
    zr = StandardScaler().fit_transform(
        im_vectors.rsignal.reshape(-1, 1)
    )

    # fit lm 
    lm = LinearRegression().fit(X=zn, y=zr)
    # track fitted parameters
    regression_res = {
        'window': w,
        'alpha': lm.intercept_[0],
        'beta': lm.coef_[0][0],
        'r_sq': lm.score(X=zn, y=zr)
    }
    system_states.append(regression_res)
    print(f'beta: {lm.coef_[0][0]}')

if LG_WINDOWS:
    outpath = f'../models/novelty_lg_sys_state_rank{DOC_RANK}.ndjson'
else:
    outpath = f'../models/novelty_sys_state_rank{DOC_RANK}.ndjson'

with open(outpath, 'w') as fout:
    ndjson.dump(system_states, fout)

msg.good('done (infodynamics)')


# %%
