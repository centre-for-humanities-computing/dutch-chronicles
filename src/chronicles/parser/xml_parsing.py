"""
Main script of parser/

Uses delimitation_strategies to parse XML files.
"""

import os
import re
from typing import List

import ndjson
from tqdm import tqdm
from wasabi import msg
from bs4 import BeautifulSoup

from delimitation_strategies import delimitation_experiment1
from delimitation_strategies import delimitation_experiment2


def parse_chronicle(path: str, delimitation_strategy: int) -> List[dict]:
    """
    Import one XML file and turn into json using one
    of two document delimitation strategies

    Parameters
    ----------
    path : str
        filepath to xml file
    delimitation_strategy : int
        1 or 2 (see documentation in `chronicles.parser.delimitation_strategies`)

    Returns
    -------
    List[dict]
        parsed chronicle with 'date', 'call_nr' and 'text' attributes
    """

    with open(path, 'r') as f_in:
        soup = BeautifulSoup(f_in, 'lxml')

    # extract call_nr: {YYYY}_{LOCATION_TAG}_{AUTHOR_TAG}
    title_tags = soup.find_all('title')
    # one call_nr must be present in file (see tests/test_parsing)
    call_nr = title_tags[0].get_text()

    increments = soup.find_all('l')

    if delimitation_strategy == 1:
        res = delimitation_experiment1(increments)
    elif delimitation_strategy == 2:
        res = delimitation_experiment2(increments)

    # add source
    # HACK i really don't like that it's inplace
    [doc.update({'call_nr': call_nr}) for doc in res]

    return res


def document_to_string(doc_list: List[dict]) -> List[dict]:
    """
    extra preprocessing step in parsing xml chronicle:
        - convert text fields (List) to a single string
        - merge multi-line words (tagged using the "¬")

    Parameters
    ----------
    doc_list : List[dict]
        output of `parse_chronicle()`

    Returns
    -------
    List[dict]
        enhanced parsed chronicle
    """
    # hyphen pattern
    pat_line_break = re.compile(r'¬\s+', re.UNICODE)
    # trailing hyphen pattern (does not delimit words)
    pat_trailing_line_break = re.compile(r'¬', re.UNICODE)
    # specials characters used by annotators, but we're ignoring them
    pat_unreadable = re.compile(r'#|@', re.UNICODE)

    if not isinstance(doc_list, list):
        doc_list = [doc_list]

    updated_doc_list = []
    for doc in doc_list:
        # turn into a single string
        doc['text'] = ' '.join(doc['text'])

        # first pass: merge multiline words
        while re.search(pat_line_break, doc['text']):
            # remove space after linebreak
            doc['text'] = re.sub(pat_line_break, '', doc['text'])

        # second pass: get rid of trailing hyphens (no space after them)
        while re.search(pat_trailing_line_break, doc['text']):
            # remove hyphens that do not delimit words
            # (= hyphen at the end of document)
            doc['text'] = re.sub(pat_trailing_line_break, '', doc['text'])

        # third pass: get rid of special characters used by annotators
        while re.search(pat_unreadable, doc['text']):
            doc['text'] = re.sub(pat_unreadable, '', doc['text'])

        updated_doc_list.append(doc)

    return updated_doc_list


def main(data_dir: str, delimitation_strategy: int) -> List[dict]:
    """
    Parse a XML chronicle corpus

    Parameters
    ----------
    data_dir : str
        path to directory where .xml files are located
    delimitation_strategy : int
        1 or 2 (see documentation in `chronicles.parser.delimitation_strategies`)

    Returns
    -------
    List[dict]
        all chronicles parsed to a single object
    """

    xml_paths = [os.path.join(data_dir, path)
                 for path in os.listdir(data_dir) if path.endswith('.xml')]

    all_events = []
    for path in tqdm(xml_paths):
        try:
            chron = parse_chronicle(path, delimitation_strategy)
            chron_up = document_to_string(chron)
            all_events.extend(chron_up)
        except:
            msg.fail(f'file failed: {path}')

    return all_events


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--datadir', type=str, default=None,
                    help='path to directory where .xml files are located')
    ap.add_argument('-s', '--strategy', type=int, default=1,
                    help='1 or 2 (see documentation in `chronicles.parser.delimitation_strategies`)')
    ap.add_argument('-o', '--outpath', type=str, default=None,
                    help='filepath to dump parsed chronicles')
    args = vars(ap.parse_args())

    extracted = main(
        data_dir=args['datadir'],
        delimitation_strategy=args['strategy']
    )

    with open(args['outpath'], 'w') as fout:
        ndjson.dump(extracted, fout)
