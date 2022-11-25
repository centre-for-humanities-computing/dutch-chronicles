# %%
import sys
from turtle import width
sys.path.append('..')

import datetime as dt
import ndjson
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
###
### pick data source for plotting
###

# primitives corrected daily
with open('../../data/primitives_220503/primitives_corrected_daily.ndjson') as fin:
    prim_cor_day = ndjson.load(fin)
    df_prim_cor_day = pd.DataFrame(prim_cor_day)

# annotated no date
with open('../../data/primitives_220503/primitives_annotated.ndjson') as fin:
    prim_an = ndjson.load(fin)

# signal used for the analysis
df_signal = pd.read_csv('../../models/220815_prototypes_day/signal.csv')


# %%
###
### primitives annotated
###

for doc in prim_an:
    year = 0000
    if isinstance(doc['date'], list):
        try:
            first_date = doc['date'][0]
            year = int(first_date[0:4])
        except:
            pass
    doc.update({'year': year})

df_an_yrly_dp = pd.DataFrame(prim_an)
df_an_yrly_dp['n_events'] = 1
df_an_yrly_dp = df_an_yrly_dp.query('year >= 1500 & year <= 1820')

df_an_yrly_dp = df_an_yrly_dp.groupby('year')['n_events'].sum().reset_index()



# %%
###
### primitives corrected
###

df_cor_yrly_dp = df_prim_cor_day.copy()

df_cor_yrly_dp['year'] = df_cor_yrly_dp['clean_date'].str[0:4].astype(int)
df_cor_yrly_dp['n_events'] = 1
df_cor_yrly_dp = df_cor_yrly_dp.query('year >= 1500 & year <= 1820')

df_cor_yrly_dp = df_cor_yrly_dp.groupby('year')['n_events'].sum().reset_index()


# %%
###
### prototypes
###

df_proto_yrly_dp = df_signal.copy()

df_proto_yrly_dp['n_events'] = 1
df_proto_yrly_dp = df_proto_yrly_dp.groupby('year')['n_events'].sum().reset_index()


# %%
###
### big plot 1
###

# subplots: left side annotated (with labels) & right side corrected and prototypes

# color pallette:
# https://coolors.co/palette/582f0e-7f4f24-936639-a68a64-b6ad90-c2c5aa-a4ac86-656d4a-414833-333d29

step = 40

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
axs[0].bar(df_an_yrly_dp['year'], df_an_yrly_dp['n_events'], width=1.2, color='#A68A64')
axs[0].set_title('Annotated', fontsize=10*scale)

axs[1].bar(df_cor_yrly_dp['year'], df_cor_yrly_dp['n_events'], width=1.2, color='#C2C5AA')
axs[1].bar(df_proto_yrly_dp['year'], df_proto_yrly_dp['n_events'], width=1.2, color='#333D29')
axs[1].set_title('Corrected & Prototypes', fontsize=10*scale)
axs[1].set_ylim(ymax=1100)


# label coords
peak1 = df_an_yrly_dp.query('year > 1540 & year < 1580').sort_values(by='n_events').tail(1)
peak2 = df_an_yrly_dp.query('year > 1650 & year < 1680').sort_values(by='n_events').tail(1)
peak3 = df_an_yrly_dp.query('year > 1680 & year < 1710').sort_values(by='n_events').tail(1)
peak4 = df_an_yrly_dp.query('year > 1740 & year < 1760').sort_values(by='n_events').tail(1)
peak5 = df_an_yrly_dp.query('year > 1780 & year < 1820').sort_values(by='n_events').tail(1)


# year labels
# axs[0].set_ylim(ymax=2600)
# axs[0].text(peak1['year'] - 11, peak1['n_events'] + 100, peak1['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
# axs[0].text(peak2['year'] - 11, peak2['n_events'] + 100, peak2['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
# axs[0].text(peak3['year'] - 11, peak3['n_events'] + 100, peak3['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
# axs[0].text(peak4['year'] - 11, peak4['n_events'] + 100, peak4['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
# axs[0].text(peak5['year'] - 11, peak5['n_events'] + 100, peak5['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))


for ax in axs: 
    ax.set_xticks(ticks=range(1500, 1800 + step, step))
    ax.xaxis.set_label_coords(0.5, -0.1)

fig.supxlabel('Year', fontsize=10*scale, y=-0.02)
fig.supylabel('Number of events', fontsize=10*scale, x=0)
plt.tight_layout()
plt.savefig('../../fig/2208_nov/document_frequency_annotated_corrected.png', bbox_inches='tight')



# %%
###
### big plot 2
###

# only corrected & prototypes shown, peaks highlighted

# color pallette:
# https://coolors.co/palette/582f0e-7f4f24-936639-a68a64-b6ad90-c2c5aa-a4ac86-656d4a-414833-333d29

step = 40

fig, axs = plt.subplots(figsize=(12, 6))

axs.bar(df_cor_yrly_dp['year'], df_cor_yrly_dp['n_events'], width=1.2, color='#C2C5AA')
axs.bar(df_proto_yrly_dp['year'], df_proto_yrly_dp['n_events'], width=1.2, color='#333D29')


# label coords
peak0 = df_cor_yrly_dp.query('year > 1530 & year < 1550').sort_values(by='n_events').tail(1)
peak1 = df_cor_yrly_dp.query('year > 1560 & year < 1580').sort_values(by='n_events').tail(1)
peak2 = df_cor_yrly_dp.query('year > 1620 & year < 1650').sort_values(by='n_events').tail(1)
peak3 = df_cor_yrly_dp.query('year > 1660 & year < 1690').sort_values(by='n_events').tail(1)
peak4 = df_cor_yrly_dp.query('year > 1690 & year < 1710').sort_values(by='n_events').tail(1)
peak5 = df_cor_yrly_dp.query('year > 1730 & year < 1760').sort_values(by='n_events').tail(1)
peak6 = df_cor_yrly_dp.query('year > 1780 & year < 1800').sort_values(by='n_events').tail(1)
peak7 = df_cor_yrly_dp.query('year > 1810 & year < 1820').sort_values(by='n_events').tail(1)


# year labels
axs.set_ylim(ymax=1300)
axs.text(peak0['year'] - 8, peak0['n_events'] + 60, peak0['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
axs.text(peak1['year'] - 8, peak1['n_events'] + 60, peak1['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
axs.text(peak2['year'] - 8, peak2['n_events'] + 60, peak2['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
axs.text(peak3['year'] - 8, peak3['n_events'] + 60, peak3['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
axs.text(peak4['year'] - 8, peak4['n_events'] + 60, peak4['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
axs.text(peak5['year'] - 8, peak5['n_events'] + 60, peak5['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
axs.text(peak6['year'] - 8, peak6['n_events'] + 60, peak6['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
axs.text(peak7['year'] - 8, peak7['n_events'] + 60, peak7['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))


axs.set_xticks(ticks=range(1500, 1800 + step, step))
axs.xaxis.set_label_coords(0.5, -0.1)

axs.set_xlabel('Year')
axs.set_ylabel('Number of events')
plt.tight_layout()
plt.savefig('../../fig/2208_nov/document_frequency_main_all_peak.png', bbox_inches='tight', dpi=300)


# %%
###
### big plot 3
###

# same as before, just less peaks highlighted

step = 40

fig, axs = plt.subplots(figsize=(12, 6))

axs.bar(df_cor_yrly_dp['year'], df_cor_yrly_dp['n_events'], width=1.2, color='#C2C5AA')
axs.bar(df_proto_yrly_dp['year'], df_proto_yrly_dp['n_events'], width=1.2, color='#333D29')


# label coords
peak0 = df_cor_yrly_dp.query('year > 1530 & year < 1550').sort_values(by='n_events').tail(1)
peak1 = df_cor_yrly_dp.query('year > 1560 & year < 1580').sort_values(by='n_events').tail(1)
peak2 = df_cor_yrly_dp.query('year > 1620 & year < 1650').sort_values(by='n_events').tail(1)
peak3 = df_cor_yrly_dp.query('year > 1660 & year < 1690').sort_values(by='n_events').tail(1)
peak4 = df_cor_yrly_dp.query('year > 1690 & year < 1710').sort_values(by='n_events').tail(1)
peak5 = df_cor_yrly_dp.query('year > 1730 & year < 1760').sort_values(by='n_events').tail(1)
peak6 = df_cor_yrly_dp.query('year > 1780 & year < 1800').sort_values(by='n_events').tail(1)
peak7 = df_cor_yrly_dp.query('year > 1810 & year < 1820').sort_values(by='n_events').tail(1)


# year labels
axs.set_ylim(ymax=1300)
# axs.text(peak0['year'] - 8, peak0['n_events'] + 60, peak0['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
axs.text(peak1['year'] - 8, peak1['n_events'] + 60, peak1['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
# axs.text(peak2['year'] - 8, peak2['n_events'] + 60, peak2['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
axs.text(peak3['year'] - 8, peak3['n_events'] + 60, peak3['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
axs.text(peak4['year'] - 8, peak4['n_events'] + 60, peak4['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
axs.text(peak5['year'] - 8, peak5['n_events'] + 60, peak5['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
axs.text(peak6['year'] - 8, peak6['n_events'] + 60, peak6['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))
# axs.text(peak7['year'] - 8, peak7['n_events'] + 60, peak7['year'].values[0], fontsize=9*scale, bbox=dict(facecolor='white', alpha=0.5))


axs.set_xticks(ticks=range(1500, 1800 + step, step))
axs.xaxis.set_label_coords(0.5, -0.1)

axs.set_xlabel('Year')
axs.set_ylabel('Number of events')
plt.tight_layout()
plt.savefig('../../fig/2208_nov/document_frequency_main_less_peak.png', bbox_inches='tight')

# %%
