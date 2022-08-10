# %%
import sys
sys.path.append('../chronicles/')

from datetime import datetime
import json

import ndjson
import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
from wasabi import msg
from top2vec import Top2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from fastdtw import fastdtw

import seaborn as sns
import matplotlib.pyplot as plt

from representation import RepresentationHandler
from misc import parse_dates
from entropies import InfoDynamics
from entropies.metrics import jsd, kld, cosine_distance
from entropies.afa import adaptive_filter, normalize

# %%
WINDOW = 30
RANK = 0

# %%
# load protoypes & primitives
with open(f'../models/prototype_docs_rank{RANK}.ndjson') as fin:
    prototype_docs = ndjson.load(fin)
    prototypes_ids = [doc['id'] for doc in prototype_docs]
    prototypes_std = [doc['uncertainity'] for doc in prototype_docs]

prims = pd.read_csv('../models/prims.csv')

# %%
# load signal for desired w
def load_signal(w, rank):
    with open(f'../models/novelty_rank{rank}/daily_w{w}.json') as fin:
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
    
    return df_signal

df_signal_30 = load_signal(WINDOW, RANK)

# %%
####
#### short-term signal overview
####

# novelty
plt.plot(
    df_signal_30['year'],
    df_signal_30['novelty'],
    c='grey',
    alpha=0.2
)

plt.plot(
    df_signal_30['year'],
    df_signal_30['novelty_afa']
)

plt.title('novelty at w=30')
plt.savefig('../fig/novelty_w30.png')

# %%
# resonance
plt.plot(
    df_signal_30['year'],
    df_signal_30['resonance'],
    c='grey',
    alpha=0.2
)

plt.plot(
    df_signal_30['year'],
    df_signal_30['resonance_afa'],
    c='green'
)

plt.title('resonance at w=30')
plt.savefig('../fig/resonance_w30.png')


# %%
####
#### check 1: exact years of peaks
####

peak_one = df_signal_30.query('year > 1550').query('year < 1590')
pp1 = peak_one.sort_values(by='novelty_afa', ascending=False).query('novelty_afa > 0.015')
pp1.groupby('year').size()

# peak: 1567, 1568
# around peak: 1559-1571
# drop after that

# %%
peak_two = df_signal_30.query('year > 1640').query('year < 1680')
pp2 = peak_two.sort_values(by='novelty_afa', ascending=False).query('novelty_afa > 0.0155')
pp2.groupby('year').size()

# peak: 1661 to 1665


# %%
peak_three = df_signal_30.query('year > 1720').query('year < 1760')
pp3 = peak_three.sort_values(by='novelty_afa', ascending=False).query('novelty_afa > 0.0105')
pp3.groupby('year').size()

# peak: 1746 to 1747

# %%
peak_four = df_signal_30.query('year > 1760').query('year < 1800')
pp4 = peak_four.sort_values(by='novelty_afa', ascending=False).query('novelty_afa > 0.014')
pp4.groupby('year').size()

# peak: 1788 1789


# %%
####
#### check 2: prototype uncertainty
####

unc_y = sp.stats.zscore(prototypes_std[WINDOW:-WINDOW])
unc_y = adaptive_filter(unc_y, span=64)

nov_y = sp.stats.zscore(df_signal_30['novelty'].tolist())
nov_y = adaptive_filter(nov_y, span=64)

plt.plot(
    df_signal_30['year'],
    unc_y,
    c='red'
)

plt.plot(
    df_signal_30['year'],
    nov_y,
    c='blue'
)

plt.title('novelty happens in event-dense periods, smoothed, z-scaled')
plt.legend(labels=['prototype uncertainty (r=0)', 'novelty (w=30)'])
plt.savefig('../fig/novelty_proto_unc_w30_r0.png')

# %%
# check2: dtw
proto_unc = np.array(prototypes_std[WINDOW:-WINDOW]).reshape(-1, 1)
novelty_30 = np.array(df_signal_30['novelty']).reshape(-1, 1)

proto_unc_z = StandardScaler().fit_transform(proto_unc)
novelty_30_z = StandardScaler().fit_transform(novelty_30)

dist_unc, path_unc = fastdtw(proto_unc_z, novelty_30_z)
dist_unc = int(np.round(dist_unc))

plt.plot(path_unc)
plt.title(f'dtw path from proto uncert. to novelty (d={dist_unc})')


# %%
####
#### check 3: novelty ~ number of documents
####

# nunber of documents
n_docs = (prims
    .groupby('clean_date')
    .size()
    .reset_index(name='n_docs')
    # FIXME remove first point (unequal lengths)
    .iloc[WINDOW+1:-WINDOW]
    ['n_docs']
)

n_docs = np.array(n_docs).reshape(-1, 1)
novelty_30 = np.array(df_signal_30['novelty']).reshape(-1, 1)


n_docs_30_z = StandardScaler().fit_transform(n_docs)
novelty_30_z = StandardScaler().fit_transform(novelty_30)

dist_docs, path_docs = fastdtw(n_docs_30_z, novelty_30_z)
dist_docs = int(np.round(dist_docs))

plt.plot(path_docs)
plt.title(f'dtw path from n_docs to novelty (d={dist_docs})')

# %%
# timeseries

plt.plot(
    df_signal_30['year'],
    adaptive_filter(n_docs_30_z[0:22516]),
    '-',
    c='pink'
)

plt.plot(
    df_signal_30['year'],
    adaptive_filter(novelty_30_z),
    '-',
)

plt.title('novelty vs. number of documents, smoothed, z-scaled')
plt.legend(labels=['number of documents', 'novelty (w=30)'])
plt.savefig('../fig/novelty_ndocs_w30.png')

# %%
####
#### check 4: novelty uncertainity
####

plt.plot(
    df_signal_30['year'],
    normalize(df_signal_30['novelty_afa']),
)

plt.plot(
    df_signal_30['year'],
    # normalize(df_signal_30['novelty_sigma'].tolist()),
    normalize(
        adaptive_filter(df_signal_30['novelty_sigma'].tolist(), span=64)
        ),
    c='red',
)
plt.title('novelty in blue, std(novelty) in red, normalized')

# %%
####
#### check 5: long term novelty 
####

w = 1000
df_signal_1000 = load_signal(w)

# %%
# long novelty
plt.plot(
    df_signal_1000['year'],
    df_signal_1000['novelty'],
    c='grey',
    alpha=0.2
)

plt.plot(
    df_signal_1000['year'],
    df_signal_1000['novelty_afa']
)

plt.title('novelty at w=1000')
plt.savefig('../fig/novelty_w1000.png')

# %%
plt.plot(
    df_signal_1000['year'],
    df_signal_1000['resonance'],
    c='grey',
    alpha=0.2
)

plt.plot(
    df_signal_1000['year'],
    df_signal_1000['resonance_afa'],
    c='green'
)

plt.title('resonance at w=1000')
plt.savefig('../fig/resonance_w1000.png')

# %%
####
#### check 6: document length
####

plt.plot(
    df_signal_30['year'],
    normalize(df_signal_30['text_len'], lower=0),
    c='orange',
    alpha=0.5
)

plt.plot(
    df_signal_30['year'],
    normalize(df_signal_30['novelty_afa'], lower=0)
)

plt.title('novelty happens in text-dense periods')
plt.legend(labels=['document length', 'novelty (w=30)'])
plt.savefig('../fig/doc_length_novelty_w30.png')

# %%
####
#### second most representative document
####

WINDOW = 30
RANK = 1

df_signal_r1 = load_signal(WINDOW, RANK)

# %%
# novelty
plt.plot(
    df_signal_r1['year'],
    df_signal_r1['novelty'],
    c='grey',
    alpha=0.2
)

plt.plot(
    df_signal_r1['year'],
    df_signal_r1['novelty_afa']
)

plt.title('peaks are similar when 2nd most representative doc is picked')
plt.legend(labels=['novelty', 'smoothed novelty'])
plt.savefig('../fig/novelty_r1_w30.png')

# %%
####
#### least representative document
####

WINDOW = 30
RANK = 100

df_signal_r100 = load_signal(WINDOW, RANK)

# %%
# novelty
plt.plot(
    df_signal_r100['year'],
    df_signal_r100['novelty'],
    c='grey',
    alpha=0.2
)

plt.plot(
    df_signal_r100['year'],
    df_signal_r100['novelty_afa']
)

plt.title('...and even when the least representative document is picked')
plt.legend(labels=['novelty', 'smoothed novelty'])
plt.savefig('../fig/novelty_r100_w30.png')

# %%
####
#### resonance vs. novelty
####

plt.plot(
    df_signal_30['novelty'],
    df_signal_30['resonance'],
    '.',
    alpha=0.2
)

plt.title('system state at w=30')
plt.xlabel('novelty')
plt.ylabel('resonance')
plt.savefig('../fig/rn_w30.png')

# %%
