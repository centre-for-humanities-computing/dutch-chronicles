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

import seaborn as sns
import matplotlib.pyplot as plt

from representation import RepresentationHandler
from misc import parse_dates
from entropies import InfoDynamics
from entropies.metrics import jsd, kld, cosine_distance
from entropies.afa import adaptive_filter

# %%
# load protoypes & primitives
with open('../models/prototype_docs.ndjson') as fin:
    prototype_docs = ndjson.load(fin)
    prototypes_ids = [doc['id'] for doc in prototype_docs]
    prototypes_std = [doc['uncertainity'] for doc in prototype_docs]

prims = pd.read_csv('../models/prims.csv')

# %%
# load signal for desired w
w = 30
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
####
#### check 1: exact years of peaks
####

peak_one = df_signal.query('year > 1550').query('year < 1590')
pp1 = peak_one.sort_values(by='novelty_afa', ascending=False).query('novelty_afa > 0.015')
pp1.groupby('year').size()

# peak: 1567, 1568
# around peak: 1559-1571
# drop after that

# %%
peak_two = df_signal.query('year > 1640').query('year < 1680')
pp2 = peak_two.sort_values(by='novelty_afa', ascending=False).query('novelty_afa > 0.0155')
pp2.groupby('year').size()

# peak: 1661 to 1665


# %%
peak_three = df_signal.query('year > 1720').query('year < 1760')
pp3 = peak_three.sort_values(by='novelty_afa', ascending=False).query('novelty_afa > 0.0105')
pp3.groupby('year').size()

# peak: 1746 to 1747

# %%
peak_four = df_signal.query('year > 1760').query('year < 1800')
pp4 = peak_four.sort_values(by='novelty_afa', ascending=False).query('novelty_afa > 0.014')
pp4.groupby('year').size()

# peak: 1788 1789

# %%
####
#### check 2: prototype uncertainity
####

unc_y = sp.stats.zscore(prototypes_std[w:-w])
unc_y = adaptive_filter(unc_y, span=64)

nov_y = sp.stats.zscore(df_signal['novelty'].tolist())
nov_y = adaptive_filter(nov_y, span=64)

plt.plot(
    df_signal['year'],
    unc_y,
    c='red'
)

plt.plot(
    df_signal['year'],
    nov_y,
    c='blue'
)

plt.title('novelty in blue, prototype uncertainity in red (z-scaled)')

# %%
####
#### check 3: novelty uncertainity
####

plt.plot(
    df_signal['year'],
    df_signal['novelty_afa']
)

plt.plot(
    df_signal['year'],
    adaptive_filter(df_signal['novelty_sigma'].tolist(), span=64),
    c='red'
)

# %%
