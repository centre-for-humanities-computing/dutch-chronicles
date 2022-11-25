"""
Add IDs to events.
Overwrites original files!
Requires both annotated & corrected corpus
"""
import ndjson
from tqdm import tqdm


def main(prim_anno, prim_corr):
    """
    Give IDs to events

    Parameters
    ----------
    prim_anno : List[dict]
        primitives_annotated - parsed annotated chronicles
    prim_corr : List[dict]
        primitives_corrected - parsed corrected chronicles

    Returns
    -------
    List[dict], List[dict]
        two files with an added field 'id',
        ids in both files are compatible.
    """

    for doc, i in zip(prim_anno, range(len(prim_anno))):
        doc['id'] = i

    prim_corr_texts = [doc['text'] for doc in prim_corr]
    prim_corr_call_nrs = [doc['call_nr'] for doc in prim_corr]
    prim_corr_fix = [doc for doc in tqdm(prim_anno) if doc['text']
                     in prim_corr_texts and doc['call_nr'] in prim_corr_call_nrs]

    assert(len(prim_corr) == len(prim_corr_fix))

    return prim_anno, prim_corr_fix


if __name__ == "__main__":

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-ap', '--annotatedpath')
    ap.add_argument('-cp', '--correctedpath')
    args = vars(ap.parse_args())

    with open(args['annotatedpath']) as fin:
        prim_anno = ndjson.load(fin)

    with open(args['correctedpath']) as fin:
        prim_corr = ndjson.load(fin)

    prim_anno_ids, prim_corr_ids = main(prim_anno, prim_corr)

    with open(args['annotatedpath'], 'w') as fout:
        ndjson.dump(prim_anno_ids, fout)

    with open(args['correctedpath'], 'w') as fout:
        ndjson.dump(prim_corr_ids, fout)
