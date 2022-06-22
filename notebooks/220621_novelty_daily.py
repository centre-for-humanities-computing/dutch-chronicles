# %%
import sys
sys.path.append('../chronicles/')

from datetime import datetime
import json

import ndjson
import numpy as np
import pandas as pd
from tqdm import tqdm
from wasabi import msg
from top2vec import Top2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from representation import RepresentationHandler
from misc import parse_dates
from entropies import InfoDynamics
from entropies.metrics import jsd, kld, cosine_distance
from entropies.afa import adaptive_filter

# %%
# load resources
model = Top2Vec.load("../models/top2vec/top2vecmodel_220504")

with open('../data/primitives_220503/primitives_corrected_daily.ndjson') as fin:
    primitives = ndjson.load(fin)

# %%
# get subset for the analysis
prims = pd.DataFrame(primitives)
prims = parse_dates(prims['clean_date'], inplace=True, df=prims)

prims = prims.query('year >= 1500 & year <= 1820')
prims = prims.sort_values(by=['year', 'week'])

prims['text_len'] = prims['text'].apply(len)
prims = prims.query('text_len > 50')


# %%
# daily doc_id groupings
df_groupings_day = (prims
    .groupby(['year', 'week', 'day'])["id"].apply(list)
    .reset_index()
    .sort_values(by=['year', 'week', 'day'])
)

groupings_day = df_groupings_day['id'].tolist()

# %%
# get ids of prototypes
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

        else:
            prot_id, prot_std = rh_daily.by_avg_distance(
                doc_ids,
                metric='cosine'
            )

        prototypes_ids.append(prot_id)
        prototypes_std.append(prot_std)

msg.info('extracting vectors')
prot_vectors = rh_daily.find_doc_vectors(prototypes_ids)
prot_cossim = rh_daily.find_doc_cossim(prototypes_ids, n_topics = 100)
prot_docs = rh_daily.find_documents(prototypes_ids)
msg.good('done (prototypes, vectors)')


# %%
# softmax on vectors
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

prot_vectors_norm = np.array([softmax(vec) for vec in prot_vectors])


# %%
# relative entropy experiments

window_param_grid = list(range(2, 51))

system_states = []
for w in tqdm(window_param_grid[-1::]):

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
        path=f'../models/novelty/daily_w{w}.json'
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

msg.good('done (infodynamics)')

# %%
# what is a good sys state
df_sys = pd.DataFrame(system_states)

# %%
# adaptive filtering on desired window
# load signal for desired w
w = 20
with open(f'../models/novelty/daily_w{w}.json') as fin:
    signal = json.load(fin)

# covert to df and add dates
df_signal = pd.DataFrame(signal)
# get a corresponding primitives df
sliced_proto_ids = prototypes_ids[w:-w]
df_proto = prims.query('id == @sliced_proto_ids')
# merge
for col_name in df_proto.columns.tolist():
    df_signal[col_name] = df_proto[col_name].tolist()

# smoothing of signal
for var_name in ['novelty', 'transience', 'resonance']:
    df_signal[f'{var_name}_afa'] = adaptive_filter(df_signal[var_name], span=64)

# date str to datetime
for i, date in df_signal['clean_date'].iteritems():
    try:
        parsed_date = datetime.strptime(date, '%Y-%m-%d')
        df_signal.loc[i, 'parsed_date'] = parsed_date
    except ValueError:
        pass


# %%
# plot smooth novelty
df_viz = df_signal.dropna()

fig = plt.figure(figsize=(10, 6))

plt.plot(
    df_viz['parsed_date'],
    df_viz['novelty'],
    c='grey',
    alpha=0.3
)

plt.plot(
    df_viz['parsed_date'],
    df_viz['novelty_afa'],
)

plt.title(f'Novelty (w={w})')
plt.savefig(f'../models/novelty_fig/novelty_smooth_w{w}.png')

# %%
# plot smooth resonance
fig = plt.figure(figsize=(10, 6))

plt.plot(
    df_viz['parsed_date'],
    df_viz['resonance'],
    c='grey',
    alpha=0.3
)

plt.plot(
    df_viz['parsed_date'],
    df_viz['resonance_afa'],
)

plt.title(f'Resonance (w={w})')
plt.savefig(f'../models/novelty_fig/resonance_smooth_w{w}.png')

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
