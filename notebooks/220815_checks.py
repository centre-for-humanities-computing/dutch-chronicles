# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore


# %%
signal_f = pd.read_csv('../models/220815_fulldocs_day/signal.csv').iloc[30:-30, :]
signal_p = pd.read_csv('../models/220815_prototypes_day/signal.csv').iloc[30:-30, :]

signal_week = pd.read_csv('../models/220815_prototypes_week/signal.csv').iloc[30:-30, :]
signal_year = pd.read_csv('../models/220815_prototypes_year/signal.csv').iloc[30:-30, :]


# %%
# novelty prototypes
plt.plot(
    signal_p['year'],
    signal_p['novelty'],
    '.',
    c='grey',
    alpha=0.2
)

plt.plot(
    signal_p['year'],
    signal_p['novelty_afa']
)

# %%
# novelty full
plt.plot(
    signal_f['year'],
    signal_f['novelty'],
    '.',
    c='grey',
    alpha=0.2
)

plt.plot(
    signal_f['year'],
    signal_f['novelty_afa']
)

# %%
# novelty week
plt.plot(
    signal_week['year'],
    signal_week['novelty'],
    '.',
    c='grey',
    alpha=0.2
)

plt.plot(
    signal_week['year'],
    signal_week['novelty_afa']
)

# %%
# novelty week
plt.plot(
    signal_year['year'],
    signal_year['novelty'],
    '.',
    c='grey',
    alpha=0.2
)

plt.plot(
    signal_year['year'],
    signal_year['novelty_afa']
)

# %%
# in one plot
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(signal_f['year'], signal_f['novelty'], '.', c='grey', alpha=0.1)
axs[0, 0].plot(signal_f['year'], signal_f['novelty_afa'])
axs[0, 0].title.set_text('No prototype picking')

axs[0, 1].plot(signal_p['year'], signal_p['novelty'], '.', c='grey', alpha=0.1)
axs[0, 1].plot(signal_p['year'], signal_p['novelty_afa'])
axs[0, 1].title.set_text('Daily prototypes')

axs[1, 0].plot(signal_week['year'], signal_week['novelty'], '.', c='grey', alpha=0.1)
axs[1, 0].plot(signal_week['year'], signal_week['novelty_afa'])
axs[1, 0].title.set_text('Weekly prototypes')

axs[1, 1].plot(signal_year['year'], signal_year['novelty'], '.', c='grey', alpha=0.3)
axs[1, 1].plot(signal_year['year'], signal_year['novelty_afa'])
axs[1, 1].title.set_text('Yearly prototypes')

fig.text(.5, .05, 'novelty in grey, smoothed novelty in blue', ha='center')

plt.savefig('../fig/2208_nov/yearly_novelty_per_prototype_resolution.png')

# %%
# with document length
colnames = ['novelty', 'novelty_afa', 'n_char']

for col in colnames:
    col_z = col + '_z'
    signal_f[col_z] = signal_f[col].transform(zscore)
    signal_p[col_z] = signal_f[col].transform(zscore)
    signal_week[col_z] = signal_f[col].transform(zscore)
    signal_year[col_z] = signal_f[col].transform(zscore)

# %%
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(signal_f['year'], signal_f['n_char_z'], '.', c='orange', alpha=0.3)
axs[0, 0].plot(signal_f['year'], signal_f['novelty_z'], '.', c='grey', alpha=0.1)
# axs[0, 0].plot(signal_f['year'], signal_f['novelty_afa']*200-3)
axs[0, 0].title.set_text('No prototype picking ($R^2 = 0.58)$')

axs[0, 1].plot(signal_p['year'], signal_p['n_char_z'], '.', c='orange', alpha=0.3)
axs[0, 1].plot(signal_p['year'], signal_p['novelty_z'], '.', c='grey', alpha=0.1)
# axs[0, 1].plot(signal_p['year'], signal_p['novelty_afa']*200-3)
axs[0, 1].title.set_text('Daily prototypes ($R^2 = 0.59$)')

axs[1, 0].plot(signal_week['year'], signal_week['n_char_z'], '.', c='orange', alpha=0.3)
axs[1, 0].plot(signal_week['year'], signal_week['novelty_z'], '.', c='grey', alpha=0.2)
# axs[1, 0].plot(signal_week['year'], signal_week['novelty_afa']*200-3)
axs[1, 0].title.set_text('Weekly prototypes ($R^2 = 0.6$)')

axs[1, 1].plot(signal_year['year'], signal_year['n_char_z'], '.', c='orange', alpha=0.5)
axs[1, 1].plot(signal_year['year'], signal_year['novelty_z'], '.', c='grey', alpha=0.5)
# axs[1, 1].plot(signal_year['year'], signal_year['novelty_afa']*200-3)
axs[1, 1].title.set_text('Yearly prototypes ($R^2 = 0.27$)')

fig.text(.5, .05, 'z(novelty) in grey, z(document length) in orange', ha='center')

plt.savefig('../fig/2208_nov/yearly_doclength_by_prototype_resolution.png')


# %%
# are those the same docs?!
ids_full = signal_f['id'].tolist()
ids_daily = signal_p['id'].tolist()
ids_weekly = signal_week['id'].tolist()
ids_yearly = signal_year['id'].tolist()

yearly_docs_in_daily = [doc in ids_daily for doc in ids_yearly]
weekly_docs_in_daily = [doc in ids_daily for doc in ids_weekly]
yearly_docs_in_weekly = [doc in ids_weekly for doc in ids_yearly]
daily_docs_in_full = [doc in ids_full for doc in ids_daily]


# %%
# long docs
signal_f_long_docs = pd.read_csv('../models/220815_fulldocs_long/signal.csv').iloc[30:-30, :]


# %%
###
# straigt signal
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(signal_f.index, signal_f['novelty'], '.', c='grey', alpha=0.1)
axs[0, 0].plot(signal_f.index, signal_f['novelty_afa'])
axs[0, 0].title.set_text('No prototype picking')
sf_ticks = signal_f[signal_f.index % 4500 == 0]['year']
sf_ticks = sf_ticks.append(signal_f[signal_f.index == min(signal_f.index)]['year'])
sf_ticks = sf_ticks.append(signal_f[signal_f.index == max(signal_f.index)]['year'])
axs[0, 0].set_xticks(ticks=sf_ticks.index, labels=sf_ticks.tolist())

axs[0, 1].plot(signal_p.index, signal_p['novelty'], '.', c='grey', alpha=0.1)
axs[0, 1].plot(signal_p.index, signal_p['novelty_afa'])
axs[0, 1].title.set_text('Daily prototypes')
sp_ticks = signal_p[signal_p.index % 3000 == 0]['year']
sp_ticks = sp_ticks.append(signal_p[signal_p.index == min(signal_p.index)]['year'])
# sp_ticks = sp_ticks.append(signal_p[signal_p.index == max(signal_p.index)]['year'])
axs[0, 1].set_xticks(ticks=sp_ticks.index, labels=sp_ticks.tolist())

axs[1, 0].plot(signal_week.index, signal_week['novelty'], '.', c='grey', alpha=0.1)
axs[1, 0].plot(signal_week.index, signal_week['novelty_afa'])
axs[1, 0].title.set_text('Weekly prototypes')
sw_ticks = signal_week[signal_week.index % 1500 == 0]['year']
sw_ticks = sw_ticks.append(signal_week[signal_week.index == min(signal_week.index)]['year'])
# sw_ticks = sw_ticks.append(signal_week[signal_week.index == max(signal_week.index)]['year'])
axs[1, 0].set_xticks(ticks=sw_ticks.index, labels=sw_ticks.tolist())

axs[1, 1].plot(signal_year.index, signal_year['novelty'], '.', c='grey', alpha=0.3)
axs[1, 1].plot(signal_year.index, signal_year['novelty_afa'])
axs[1, 1].title.set_text('Yearly prototypes')

fig.text(.5, .05, 'novelty in grey, smoothed novelty in blue', ha='center')

# plt.savefig('../fig/2208_nov/raw_novelty_per_prototype_resolution.png')


# %%
# novelty long docs
plt.plot(
    signal_f_long_docs['year'],
    signal_f_long_docs['novelty'],
    '.',
    c='grey',
    alpha=0.2
)

plt.plot(
    signal_f_long_docs['year'],
    signal_f_long_docs['novelty_afa']
)
