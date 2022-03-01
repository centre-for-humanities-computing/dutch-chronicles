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
# docs = []
# for doc in glob.glob('../data/corpus_220222_corrected/test/*.xml'):
#     with open(doc, 'r') as tei:
#         soup = BeautifulSoup(tei, 'lxml')
#         words = []
#         linelist = [''.join(l.findAll(text=True)) for l in soup.findAll('l')]
#         wordstring = ' '.join(linelist)
#         wordstring_without_hyphen = re.sub(r"((¬#?) ?)", "", wordstring)
#         docs.append(wordstring)

# %%
###
### line breaks
###
line_breaks = []
for doc in tqdm(glob.glob('../../data/corpus_220222_corrected/*.xml')):
    with open(doc, 'r') as tei:
        soup = BeautifulSoup(tei, 'lxml')
        words = []
        linelist = [''.join(l.findAll(text=True)) for l in soup.findAll(r'¬')]
        line_breaks.append(linelist)


# %%
##
## one doc test: lxml
##
docs_lxml = []
path = '/Users/au582299/Repositories/chronicles/data/corpus_220222_corrected/1567_Amst_Bies.xml'
with open(path, 'r') as f_in:
    soup = BeautifulSoup(f_in, 'lxml')
    words = []
    linelist = [''.join(l.findAll(text=True)) for l in soup.findAll('l')]
    wordstring = ' '.join(linelist)
    wordstring_without_hyphen = re.sub(r"((¬#?) ?)", "", wordstring)
    docs_lxml.append(wordstring_without_hyphen)

# %%
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
path = '/Users/au582299/Repositories/dutch-chronicles/data/corpus_220222_corrected/1567_Amst_Bies.xml'

with open(path, 'r') as f_in:
    soup = BeautifulSoup(f_in, 'lxml')

    # extract call_nr: {YYYY}_{LOCATION_TAG}_{AUTHOR_TAG}
    title_tags = soup.find_all('title')
    # check only one call_nr is present in file
    assert len(title_tags) == 1
    call_nr = title_tags[0].get_text()

    # page numbers
    page_nrs = [page_nr['n'] for page_nr in soup.find_all('pb')]

    docs = []
    for page_nr, increment in zip(page_nrs, soup.find_all(document_increment)):
        text_lines = [line.get_text() for line in increment.find_all('l')]
        date_lines = [line['datum'] for line in increment.find_all('datum')]
        # person_lines = date_lines = [line.get_text() for line in increment.find_all('person')]
        # loc_lines = date_lines = [line.get_text() for line in increment.find_all('loc')]

        docs.append({
            'call_nr': call_nr,
            'page': page_nr,
            'text': text_lines,
            'date': date_lines
        })





    # linelist = [''.join(l.findAll(text=True)) for l in soup.findAll('l')]
    # wordstring = ' '.join(linelist)
    # wordstring_without_hyphen = re.sub(r"((¬#?) ?)", "", wordstring)
    # docs_lxml.append(wordstring_without_hyphen)

# %%
