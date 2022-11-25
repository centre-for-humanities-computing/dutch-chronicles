# %%
import os
import glob
import pandas as pd
import numpy as np
import re
from wasabi import msg

import bs4
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
# ahoj
def detect_date_locator(tag):
    '''
    line : bs4.element.Tag
        can behave like a list 
    '''
    # get correct tag to use
    if tag.has_attr('datum'):
        locator = 'datum'
    elif tag.has_attr('when'):
        locator = 'when'
    else:
        locator = None
    
    return locator

def extract_date_attr(tag):
    '''
    line : bs4.element.Tag
        can behave like a list 
    '''

    # extract, depending if list or not
    dates = []
    for date in tag:
        locator = detect_date_locator(date)
        if locator:
            dates.append(date[locator])

    return dates


# %%
###
### one case document parser – experiment 1
###

path = '/Users/au582299/Repositories/dutch-chronicles/data/corpus_220222_corrected/1791_Purm_Louw_03.xml'

with open(path, 'r') as f_in:
    soup = BeautifulSoup(f_in, 'lxml')

    increments = soup.find_all('l')

# if you solve it for increment[2], then we're doing good

# increment_lines = increments[3].find_all('l')

docs = []
partial_doc = {}
for i, line in enumerate(increments):

    # detect dates in line
    # TODO: finding 'when'
    dates_found = line.find_all('datum')

    # if datum is found, start a new doc
    if dates_found:
        doc_date = extract_date_attr(dates_found)

        # add old doc
        docs.append(partial_doc)

        # start a new doc
        partial_doc = {}
        text_content = [i.get_text() for i in line]
        partial_doc['date'] = doc_date
        partial_doc['text'] = text_content

    else:
        if partial_doc:
            text_content = [i.get_text() for i in line]
            partial_doc['text'].extend(text_content)
        else:
            partial_doc = {}
            text_content = [i.get_text() for i in line]
            partial_doc['date'] = 'NaN'
            partial_doc['text'] = text_content

    if i+1 == len(increments):
        docs.append(partial_doc)

'''
Segment on lines, not pages (events can go between pages)

Missing last documents
'''
# %%

# %%
def delimitation_experiment1(increment):
    '''
    text corresponding to a primitive -> `{date_1} document_1 {date_2} document_2`

    Parameters
    ----------
    increment : bs4.element.ResultSet
        iterable of bs4.element.Tag
    '''

    assert(isinstance(increment, bs4.element.ResultSet))
    assert(isinstance(increment[0], bs4.element.Tag))

    docs = []
    partial_doc = {}
    for i, line in enumerate(increments):

        # detect dates in line
        # TODO: finding 'when'
        dates_found = line.find_all('datum')

        # if datum is found, start a new doc
        if dates_found:
            doc_date = extract_date_attr(dates_found)

            # add old doc
            docs.append(partial_doc)

            # start a new doc
            partial_doc = {}
            text_content = [i.get_text() for i in line]
            partial_doc['date'] = doc_date
            partial_doc['text'] = text_content

        else:
            if partial_doc:
                text_content = [i.get_text() for i in line]
                partial_doc['text'].extend(text_content)
            else:
                partial_doc = {}
                text_content = [i.get_text() for i in line]
                partial_doc['date'] = 'NaN'
                partial_doc['text'] = text_content

        if i+1 == len(increments):
            docs.append(partial_doc)


        return docs



def chroincle_loader(path, increment_tag):

    with open(path, 'r') as f_in:
        soup = BeautifulSoup(f_in, 'lxml')

    # extract call_nr: {YYYY}_{LOCATION_TAG}_{AUTHOR_TAG}
    title_tags = soup.find_all('title')
    # one call_nr must be present in file (see tests/test_parsing)
    call_nr = title_tags[0].get_text()

    # get a list of page numbers. len(page_nrs) is the number of pages in a chronicle.
    page_nrs = [page_nr['n'] for page_nr in soup.find_all('pb')]

    # get document increments to itterate over
    increments = soup.find_all(increment_tag)

    # iterate over page numbers & increments
    primitives_chronicle = []
    for page_nr, increment in zip(page_nrs, increments):
        doc = delimitation_experiment1(increment.find_all('l'))
        primitives_chronicle.extend(doc)
    
    return primitives_chronicle


# %%
# test run: works for just lines?
path = '/Users/au582299/Repositories/dutch-chronicles/data/corpus_220222_corrected/1791_Purm_Louw_03.xml'
incr_tag = 'l'
a = chroincle_loader(path, incr_tag)

# %%
# test run: works for pages?
path = '/Users/au582299/Repositories/dutch-chronicles/data/corpus_220222_corrected/1791_Purm_Louw_03.xml'
incr_tag = 'p'
a = chroincle_loader(path, incr_tag)

# %%