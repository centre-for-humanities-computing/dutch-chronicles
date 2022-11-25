# %%
import numpy as np
import pandas as pd
from tqdm import tqdm

from top2vec import Top2Vec
from umap import UMAP
from top2vec import Top2Vec

import matplotlib.pyplot as plt


# %%
# plt settings
scale = 1.8
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
model = Top2Vec.load("../../models/top2vec/top2vecmodel_220504")
vectors = np.load("../../models/220815_prototypes_day/vectors.npy")[30:-30]


# %%
def calc_inertia(vectors, topic_centroids):

    distances = []
    for vect in vectors:
        d = np.linalg.norm(vect - topic_centroids, axis=1)
        d_closest_centroid = d.min()
        distances.append(d_closest_centroid)

    inertia = sum(distances)
    assert np.array(distances).min() > 0

    return inertia


# %%
out = []
for n_topics in tqdm([400, 200, 100, 50, 40, 30, 20, 10, 5]):
    model.hierarchical_topic_reduction(num_topics=n_topics)

    topic_centroids = model.topic_vectors_reduced
    X = np.vstack([topic_centroids, vectors])
    umap = UMAP(random_state=42)
    X_2d =  umap.fit_transform(X)
    topic_centroids_2d = X_2d[0:n_topics]
    vectors_2d = X_2d[n_topics::]

    inertia = calc_inertia(vectors_2d, topic_centroids_2d)
    out.append({'n_topics': n_topics, 'inertia': inertia})

df_out = pd.DataFrame(out)
df_out.to_csv('inertias.csv', index=False)

# %%
df_out = pd.read_csv('inertias.csv')

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(df_out['n_topics'], df_out['inertia'])
ax.plot(df_out['n_topics'], df_out['inertia'], '.')
ax.set_xlabel('n topics')
ax.set_ylabel('inertia')
ax.set_xticks(ticks=[5, 50, 100, 200, 400])

plt.tight_layout()
plt.savefig('../../fig/2208_nov/topics_elbow.png', dpi=150)


# %%
