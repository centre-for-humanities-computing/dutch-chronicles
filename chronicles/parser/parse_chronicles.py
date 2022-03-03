# %%
import os
import glob
import pandas as pd
import numpy as np
import re
from wasabi import msg

from bs4 import BeautifulSoup
from tqdm import tqdm
import lxml


'''
old nltk script
    unwanted: does tokenization
    quality not reviewed & not trusted

bs4 parsing
    on a test file, lxml does not make a difference
    parsing with BeautifulSoup(IN, 'lxml') == BeautifulSoup(IN)
'''

# %%
##
## one doc test: lxml
##
# docs_lxml = []
# path = '/Users/au582299/Repositories/chronicles/data/corpus_220222_corrected/1567_Amst_Bies.xml'
# with open(path, 'r') as f_in:
#     soup = BeautifulSoup(f_in, 'lxml')
#     words = []
#     linelist = [''.join(l.findAll(text=True)) for l in soup.findAll('l')]
#     wordstring = ' '.join(linelist)
#     wordstring_without_hyphen = re.sub(r"((¬#?) ?)", "", wordstring)
#     docs_lxml.append(wordstring_without_hyphen)

# %%
###
### one doc: extract primitives
###
'''
- extract title
- extract {document_unit}
    - extract text
        - join linebreaks ¬
    - extract tags
        - datum
        - person
        - loc
'''
document_increment = 'p'
path = '/Users/au582299/Repositories/dutch-chronicles/data/corpus_220222_corrected/1791_Purm_Louw_03.xml'

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
data_dir = '../../data/corpus_220222_corrected'
xml_paths = [os.path.join(data_dir, path)
             for path in os.listdir(data_dir) if path.endswith('.xml')]

chronicles = []
for path in tqdm(xml_paths):
    msg.info(path)
    chron = extract_primitives(path, document_increment='p')
    chronicles.append(chron)

