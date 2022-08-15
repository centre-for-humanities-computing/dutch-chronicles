''' 
Combine top2vec model, dataset of primitives.
Find prototypes.
Calculate novelty at different windows.
Export a results.


Parameters (yaml)
-----------------
paths: 
(loading necessary resources)

    top2vec : str
        path a trained top2vec model
    primitives : str
        path to documents
        assumes .ndjson & existing fields 
        `text`:str,
        `id`:str,
        `clean_date`:str
    outdir : str
        path to directory in which results will be dumped
        and new subfolders created


filter:
(limits for pd.query when subsetting the dataset)
(all queries are <= or >=)

    min_year : int
    max_year : int
    min_nchar : int
    max_nchar : int


representataion:
(what representations to export)

    softmax : bool
        apply softmax to document representations?
    export_vec : bool
        export doc2vec representations?
    export_docsim : bool
        export cosince similarities to 100 topic centroids?


prototypes:
(how to pick prototypical documents)

    find_prototypes : bool
        switch to bypass prototype searching 
        if True, only prototypical documents will be used for novelty calculation
        if False, all documents are used.
    resolution : str
        what time resolution to group documents on
        either 'year' or 'week' or 'day'
    doc_rank : int
        when ordered by average distance, document of which rank to pick as prototype
        if 0, the document with LOWEST avg distance will be picked.
        if 1, the doc with SECOND LOWEST avg dist
        if {doc_rank > len(group)} the doc with HIGHEST avg dist will be picked.


novelty:
(parameters for caluclating relative entropies)

    windows : List[int]
        w parameters to iterate though in the novelty calculation.
        w is the number of preceeding/following documents the focus document should be compared to.

'''

# %%
import os
import yaml
import argparse

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
from util import softmax


def main(param):
    # load resources
    model = Top2Vec.load(param['paths']['top2vec'])

    with open(param['paths']['primitives']) as fin:
        primitives = ndjson.load(fin)

    # parse dates & get metadata of the subset
    prims_unfiltered = pd.DataFrame(primitives)
    prims_unfiltered = parse_dates(
        prims_unfiltered['clean_date'], inplace=True, df=prims_unfiltered)

    # text length
    prims_unfiltered['n_char'] = prims_unfiltered['text'].str.len()
    prims_unfiltered.describe()

    msg.info('subset description')
    print(prims_unfiltered.describe())

    # filtering
    prims = prims_unfiltered.copy()
    minyear = param['filter']['min_year']
    maxyear = param['filter']['max_year']
    minnchar = param['filter']['min_nchar']
    maxnchar = param['filter']['max_nchar']

    # cut extreme years
    prims = prims.query('year >= @minyear & year <= @maxyear')
    prims = prims.sort_values(by=['year', 'week'])
    # cut very short & very long docs
    prims = prims.query('n_char >= @minnchar & n_char <= @maxnchar')
    prims.describe()

    msg.info('filtered subset description')
    print(prims.describe())

    # switch: pick prototypes if desired
    if param['prototypes']['find_prototypes']:

        # find what resolution to group on
        grouping_levels = ['year', 'week', 'day']
        last_level_idx = grouping_levels.index(
            param['prototypes']['resolution'])
        grouping_levels = grouping_levels[:last_level_idx + 1]

        # group by day
        df_groupings = (prims
                        .groupby(grouping_levels)["id"].apply(list)
                        .reset_index()
                        .sort_values(by=grouping_levels)
                        )

        groupings_ids = df_groupings['id'].tolist()

        rh_daily = RepresentationHandler(
            model, primitives, tolerate_invalid_ids=False
        )

        prototypes_ids = []
        prototypes_std = []

        msg.info('finding prototypes')
        for group in tqdm(groupings_ids):
            # take group
            doc_ids = rh_daily.filter_invalid_doc_ids(group)

            # check for empty group
            if doc_ids:
                # single document in a group = prototype with 0 uncertainity
                if len(doc_ids) == 1:
                    prot_id = doc_ids[0]
                    prot_std = 0

                # if doc_rank is higher than group size pick...
                # ...the last possible document as prototype
                elif param['prototypes']['doc_rank'] >= len(doc_ids):
                    prot_id, prot_std = rh_daily.by_avg_distance(
                        doc_ids,
                        metric='cosine',
                        doc_rank=len(doc_ids)-1
                    )

                # any other case (multiple docs in group & doc_rank < group size)
                # pick doc with desired rank as prototype
                else:
                    prot_id, prot_std = rh_daily.by_avg_distance(
                        doc_ids,
                        metric='cosine',
                        doc_rank=param['prototypes']['doc_rank']
                    )

                prototypes_ids.append(prot_id)
                prototypes_std.append(prot_std)

        msg.info('extracting vectors')
        prot_vectors = rh_daily.find_doc_vectors(prototypes_ids)
        prot_cossim = rh_daily.find_doc_cossim(prototypes_ids, n_topics=100)
        prot_docs = rh_daily.find_documents(prototypes_ids)

        # add uncertainity to doc dump
        [doc.update({'uncertainity': float(std)})
         for doc, std in zip(prot_docs, prototypes_std)]


    else:
        # no prototypes = extract all document vectors

        rh_noproto = RepresentationHandler(
            model, primitives, tolerate_invalid_ids=False
        )

        subset_ids = prims['id'].tolist()
        valid_subset_ids = rh_noproto.filter_invalid_doc_ids(subset_ids)

        prot_vectors = rh_noproto.find_doc_vectors(valid_subset_ids)
        prot_cossim = rh_noproto.find_doc_cossim(valid_subset_ids, n_topics=100)
        prot_docs = rh_noproto.find_documents(valid_subset_ids)


    # dump section
    # paths
    path_prototypes = os.path.join(param['paths']['outdir'], "prototypes.ndjson")
    path_vector = os.path.join(param['paths']['outdir'], "vectors.npy")
    path_cossim = os.path.join(param['paths']['outdir'], "cossims.npy")
    # dump prototypes
    with open(path_prototypes, 'w') as fout:
        ndjson.dump(prot_docs, fout)
    # dump doc2vec representations
    np.save(path_vector, prot_vectors)
    # dump cosine similarities to topic centroids
    np.save(path_cossim, prot_cossim)
    

    msg.good('done (prototypes, vectors)')


    # softmax on vectors
    if param['representation']['softmax']:
        prot_vectors = np.array([softmax(vec) for vec in prot_vectors])


    # relative entropy experiments
    system_states = []
    for w in tqdm(param['novelty']['windows']):

        msg.info(f'infodynamics w {w}')
        # initialize infodyn class
        im_vectors = InfoDynamics(
            data=prot_vectors,
            window=w,
            time=None,
            normalize=False
        )

        # calculate with jensen shannon divergence & save results
        # base=2 must be hard defined in entropies.metrics
        im_vectors.fit_save(
            meas=jsd,
            slice_w=False,
            path=os.path.join(param['paths']['outdir'], f'novelty_w{w}.ndjson')
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
        lm = LinearRegression(fit_intercept=False).fit(X=zn, y=zr)
        # track fitted parameters
        regression_res = {
            'window': w,
            'alpha': lm.intercept_,
            'beta': lm.coef_[0][0],
            'r_sq': lm.score(X=zn, y=zr)
        }
        system_states.append(regression_res)
        print(f'beta: {lm.coef_[0][0]}')

    with open(os.path.join(param['paths']['outdir'], 'infodynamics_system_states.ndjson'), 'w') as fout:
        ndjson.dump(system_states, fout)

    msg.good('done (infodynamics)')


# %%
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--yml')
    args = vars(ap.parse_args())

    # with open(args['settings']) as fin:
    with open(args['yml']) as fin:
        param = yaml.safe_load(fin)

    # init output folder
    if not os.path.exists(param['paths']['outdir']):
        os.mkdir(param['paths']['outdir'])

    main(param)
