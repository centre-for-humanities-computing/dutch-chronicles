# %%
import os
import pandas as pd
import numpy as np
import re
import ndjson
from wasabi import msg

from bs4 import BeautifulSoup
from tqdm import tqdm
import lxml

# %%
def extract_primitives(path, document_increment):
    '''
    Get all date annotations from a chronicle
    '''

    with open(path, 'r') as f_in:
        soup = BeautifulSoup(f_in, 'lxml')

        # extract call_nr: {YYYY}_{LOCATION_TAG}_{AUTHOR_TAG}
        title_tags = soup.find_all('title')
        # check only one call_nr is present in file
        assert len(title_tags) == 1
        call_nr = title_tags[0].get_text()

        # page numbers
        page_nrs = [page_nr['n'] for page_nr in soup.find_all('pb')]

        primitives = []
        for page_nr, increment in zip(page_nrs, soup.find_all(document_increment)):
            text_lines = [line.get_text() for line in increment.find_all('l')]

            # catch multiple dates in a single increment
            dates_in_increment = increment.find_all('datum')
            if isinstance(dates_in_increment, list):

                date_lines = []
                for line in dates_in_increment:
                    if line.has_attr('datum'):
                        date_tag_type = 'datum'
                    elif line.has_attr('when'):
                        date_tag_type = 'when'
                    else: 
                        # exception when date is tagged, but not annotated
                        date_tag_type = None
                    
                    if date_tag_type:
                        one_date = line[date_tag_type]
                        date_lines.append(one_date)

            else:
                date_lines = dates_in_increment['datum']

            # date_lines = [line['datum'] for line in increment.find_all('datum')]
            # person_lines = date_lines = [line.get_text() for line in increment.find_all('person')]
            # loc_lines = date_lines = [line.get_text() for line in increment.find_all('loc')]

            primitives.append({
                'call_nr': call_nr,
                'page': page_nr,
                'text': text_lines,
                'date': date_lines
            })
    
    return primitives


# get filepaths
# data_dir = '../../data/corpus_220222_corrected'
data_dir = 'data/corpus_220222_corrected'

xml_paths = [os.path.join(data_dir, path)
             for path in os.listdir(data_dir) if path.endswith('.xml')]

chronicles = []
for path in tqdm(xml_paths):
    try:
        chron = extract_primitives(path, document_increment='p')
        chronicles.extend(chron)
    except:
        msg.fail(f'file failed: {path}')

with open('data/primitives_220303/primitives.ndjson', 'w') as fout:
    ndjson.dump(chronicles, fout)