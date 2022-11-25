# %%
import sys
sys.path.append('..')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from top2vec import Top2Vec
from umap import UMAP
from scipy.stats import zscore
from entropies.afa import adaptive_filter
from representation import RepresentationHandler


# %%
# matplotlib settings
scale = 2.2
plt.rcParams.update({"text.usetex": False,
                    "font.family": "Times New Roman",
                    "font.serif": "serif",
                    "mathtext.fontset": "cm",
                    "axes.unicode_minus": False,
                    "axes.labelsize": 9*scale,
                    "xtick.labelsize": 9*scale,
                    "ytick.labelsize": 9*scale,
                    "legend.fontsize": 9*scale,
                    'axes.titlesize': 14,
                    "axes.linewidth": 1
                    })


# %%
# load signal
signal_p = pd.read_csv('../../models/220815_prototypes_day/signal.csv').iloc[30:-30, :]

# zscore
signal_p['novelty_z'] = zscore(signal_p['novelty'])
signal_p['novelty_afa_z'] = zscore(signal_p['novelty_afa'])
signal_p['novelty_sigma_z'] = zscore(signal_p['novelty_sigma'])

# average signal per year
signal_avg = signal_p.groupby('year').mean().reset_index()
signal_std = signal_p.groupby('year').std().reset_index().fillna(0)


# %%
###
### baseplot: yearly averages
###

'''
Plot shows all datapoints in grey.
Adaptive filter, averaged by year is shown in red. 

'''

step = 40

fig, ax = plt.subplots(figsize=(12, 6))
# raw signal
ax.plot(signal_p['year'], signal_p['novelty_z'], '.', color='grey', alpha=0.1)
# uncertainity
# y1 = signal_avg['novelty_afa_z'] - signal_std['novelty_afa_z']
# y2 = signal_avg['novelty_afa_z'] + signal_std['novelty_afa_z']
# ax.fill_between(signal_std['year'], y1, y2)
# adaptive filter fit
ax.plot(signal_avg['year'], signal_avg['novelty_afa_z'], '-', color='darkred')

ax.set_xlabel('Year')
ax.set_ylabel('$z(\mathbb{N}ovelty)$')

ax.set_xticks(ticks=range(1500, 1800 + step, step))
ax.set_ylim(ymax=4.5)
ax.xaxis.set_label_coords(0.5, -0.15)


plt.tight_layout()
plt.savefig('../../fig/2208_nov/novelty_ts.png', dpi=300)


# %%
###
### baseplot: with labels
###


step = 40

fig, ax = plt.subplots(figsize=(12, 6))
# raw signal
ax.plot(signal_p['year'], signal_p['novelty_z'], '.', color='grey', alpha=0.1)
# adaptive filter fit
ax.plot(signal_avg['year'], signal_avg['novelty_afa_z'], '-', color='darkred')

ax.set_xlabel('Year')
ax.set_ylabel('$z(\mathbb{N}ovelty)$')

ax.set_xticks(ticks=range(1500, 1800 + step, step))
ax.set_ylim(ymax=4.5)
ax.xaxis.set_label_coords(0.5, -0.15)

# peaks
peak0 = signal_p.query('year > 1540 & year < 1580').sort_values(by='novelty_afa_z').tail(1)
peak1 = signal_p.query('year > 1650 & year < 1670').sort_values(by='novelty_afa_z').tail(1)
peak2 = signal_p.query('year > 1740 & year < 1760').sort_values(by='novelty_afa_z').tail(1)
peak3 = signal_p.query('year > 1780 & year < 1800').sort_values(by='novelty_afa_z').tail(1)

# labels
ax.text(peak0['year'] - 8, peak0['novelty_afa_z'] + 0.5, peak0['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
ax.text(peak1['year'] - 8, peak1['novelty_afa_z'] + 0.5, peak1['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
# ax.text(peak2['year'] - 8, peak2['novelty_afa_z'] + 0.5, peak2['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
ax.text(peak3['year'] - 8, peak3['novelty_afa_z'] + 0.5, peak3['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))


plt.tight_layout()
plt.savefig('../../fig/2208_nov/novelty_ts_labeled.png', dpi=300)


# %%
###
### baseplot: with down low labels
###

step = 40

fig, ax = plt.subplots(figsize=(12, 6))
# raw signal
ax.plot(signal_p['year'], signal_p['novelty_z'], '.', color='grey', alpha=0.1)
# adaptive filter fit
ax.plot(signal_avg['year'], signal_avg['novelty_afa_z'], '-', color='darkred')

ax.set_xlabel('Year')
ax.set_ylabel('$z(\mathbb{N}ovelty)$')

ax.set_xticks(ticks=range(1500, 1800 + step, step))
ax.set_ylim(ymax=4.8)
ax.xaxis.set_label_coords(0.5, -0.15)

# peaks
peak0 = signal_p.query('year > 1540 & year < 1580').sort_values(by='novelty_afa_z').tail(1)
peak1 = signal_p.query('year > 1650 & year < 1670').sort_values(by='novelty_afa_z').tail(1)
peak2 = signal_p.query('year > 1740 & year < 1760').sort_values(by='novelty_afa_z').tail(1)
peak3 = signal_p.query('year > 1780 & year < 1800').sort_values(by='novelty_afa_z').tail(1)

# labels
# ax.text(peak0['year'] - 8, 4, peak0['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
# ax.text(peak1['year'] - 8, 4, peak1['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
# ax.text(peak2['year'] - 8, 4, peak2['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
# ax.text(peak3['year'] - 8, 4, peak3['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))


ax.text(peak0['year'] - 8, peak0['novelty_afa_z'] + 2.2, peak0['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
ax.text(peak1['year'] - 8, peak1['novelty_afa_z'] + 2.2, peak1['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
# ax.text(peak2['year'] - 8, peak2['novelty_afa_z'] + 3.6, peak2['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
ax.text(peak3['year'] - 8, peak3['novelty_afa_z'] + 2.3, peak3['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))


plt.tight_layout()
# plt.savefig('../../fig/2208_nov/novelty_ts_labeled_non_overlaping.png', dpi=300)


# %%
###
### dominant topics
###

model = Top2Vec.load("../../models/top2vec/top2vecmodel_220504")
cossims = np.load("../../models/220815_prototypes_day/cossims.npy")

# cossims over time
df_cossims = pd.DataFrame(cossims).iloc[30:-30]
df_tp_viz = pd.concat([signal_p, df_cossims], axis=1)

# aggregate
tp_cols = df_tp_viz.columns[-100:].tolist()
df_tp_viz_yr = df_tp_viz.groupby('year').mean().reset_index()


# %%
###
### topic plot per peak
###
color_palette_tp = {
    0: '#FF595E',
    4: '#FF924C',
    5: '#FFCA3A',
    7: '#8AC926',
    9: '#52A675',
    10: '#1982C4',
    49: '#4267AC',
    56: '#6A4C93'
}

def plot_topics(i_ax, yr, top_tp):
    yrmin = yr - 10
    yrmax = yr + 10
    df_peak = df_tp_viz_yr.query('year >= @yrmin & year <= @yrmax')
    
    tps_back = [tp for tp in tp_cols if tp not in top_tp]

    for tp in tps_back:
        axs[i_ax].plot(df_peak['year'], df_peak[tp], alpha=0.05)
    for tp in top_tp:
        topic_c = color_palette_tp[tp]
        axs[i_ax].plot(df_peak['year'], df_peak[tp], c=topic_c, label=f'Topic {tp}')

    axs[i_ax].axvline(x=yr, linestyle='--', color='black', alpha=0.5)
    axs[i_ax].set_xticks(ticks=range(yrmin, yrmax + step, step))
    axs[i_ax].set_title(f'{yr}', fontsize=10*scale)
    axs[i_ax].set_ylim(ymax=0.30)
    axs[i_ax].legend(loc='upper right', prop={'size': 14})

fig, axs = plt.subplots(1, 3, figsize=(12, 6))
step=5

# first tile
plot_topics(0, yr=1568, top_tp=[10, 7, 9])
# plot_topics(0, yr=1568, top_tp=[10, 7, 9, 12, 8])

# second tile
plot_topics(1, yr=1662, top_tp=[7, 4, 5])
# plot_topics(1, yr=1662, top_tp=[7, 4, 5, 0, 6])

# third tile
plot_topics(2, yr=1789, top_tp=[56, 49, 0])
# plot_topics(2, yr=1789, top_tp=[56, 49, 0, 63, 61])

# fig.supxlabel('Year', fontsize=10*scale, y=-0.02)
# fig.supylabel('Average cosine similarity', fontsize=8*scale, x=0)
plt.tight_layout()
plt.savefig('../../fig/2208_nov/peak_topics.png', dpi=300)

# %%

# %%
# ####################
# ### experimental ###
# ####################




# # %%
# ###
# ### topics viz: mine
# ###

# cossims = np.load("../../models/220815_prototypes_day/cossims.npy")[30:-30]
# vectors = np.load("../../models/220815_prototypes_day/vectors.npy")[30:-30]

# # %%
# # cossims over time
# df_cossims = pd.DataFrame(cossims)
# for i in range(100):
#     ts_topic = adaptive_filter(df_cossims[i])
#     plt.plot(signal_p['year'], ts_topic, alpha=0.1)


# # %%
# ###
# ### topic centroids viz
# ###
# model = Top2Vec.load("../../models/top2vec/top2vecmodel_220504")

# n_topics = 100

# # %%
# topic_centroids = model.topic_vectors_reduced
# X = np.vstack([topic_centroids, vectors])

# umap = UMAP(random_state=42)
# X_2d =  umap.fit_transform(X)

# topic_centroids_2d = X_2d[0:n_topics]
# vectors_2d = X_2d[n_topics::]

# # %%
# plt.plot(vectors_2d[:, 0], vectors_2d[:, 1], '.', c='grey', alpha=0.1)
# plt.plot(topic_centroids_2d[:, 0], topic_centroids_2d[:, 1], '.', c='blue', markersize=10)



# # %%
# ###
# ### which documents are peaks
# ###

# import plotly.express as px

# px.line(
#     signal_p.reset_index(),
#     x='year',
#     y='novelty_afa',
#     hover_data=['id', 'index']
# )


# # %%
# ###
# ### representative words
# ###

# ids_peak1 = list(range(1776, 1798+1))

# words_peak1 = pd.DataFrame([])
# for doc_id in ids_peak1:
#     words, sims = model.search_words_by_vector(vectors[doc_id], num_words=50)
#     doc_words = pd.DataFrame({'id': doc_id, 'word': words, 'sim': sims})
#     # doc_words['id'] = doc_id
#     words_peak1 = words_peak1.append(doc_words)

# # %%
# words_peak1.groupby('word')['sim'].mean().reset_index().sort_values(by='sim', ascending=False).head(20)

# # words_peak1.sort_values(by='sim', ascending=False)

# # %%
# ###
# ### reduced cossims over time
# ###

# n_topics = 20

# model = Top2Vec.load("../../models/top2vec/top2vecmodel_220504")
# model.hierarchical_topic_reduction(num_topics=n_topics)

# fake_prim = []
# for val in signal_p['id'].tolist():
#     fake_prim.append({'id': val})

# rh = RepresentationHandler(
#     model=model,
#     primitives=fake_prim,
#     tolerate_invalid_ids=False
# )

# cossims_reduced = rh.find_doc_cossim(signal_p['id'].tolist(), n_topics=n_topics)

# # %%
# df_cossims_reduced = pd.DataFrame(cossims_reduced)
# for i in range(n_topics):
#     ts_topic = adaptive_filter(df_cossims_reduced[i])
#     # ts_topic = df_cossims_reduced[i]
#     plt.plot(signal_p['year'], ts_topic, alpha=0.1)


# # %%
