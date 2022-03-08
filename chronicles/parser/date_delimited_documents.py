'''
Document segmentation into primitives, where documents are delimited by dates.

Text corresponding to a primitive -> `{date_1} document_1 {date_2} document_2`

'''
# %%
import os
import ndjson
from turtle import resetscreen
from wasabi import msg

import bs4
from bs4 import BeautifulSoup
from tqdm import tqdm
import lxml


def detect_date_locator(tag):
    '''
    Help function, finding key to extract date tag from an annotated field.

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
    Help function, actually extract date labels

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


def delimitation_experiment1(increments):
    '''
    text corresponding to a primitive -> `{date_1} document_1 {date_2} document_2`

    Parameters
    ----------
    increment : bs4.element.ResultSet
        iterable of bs4.element.Tag
    '''
    # check input is not completely outlandish
    assert(isinstance(increments, bs4.element.ResultSet))
    assert(isinstance(increments[0], bs4.element.Tag))

    docs = [] # dump for the results
    partial_doc = {} # create an empty extraction doc that gives a False
    for i, line in enumerate(increments):

        # detect dates in line
        dates_found = line.find_all('datum')

        # if datum is found, start a new doc
        if dates_found:
            # use date tag extraction function to handle exception
            doc_date = extract_date_attr(dates_found)

            # add cache (PREVIOUS doc) to results
            docs.append(partial_doc)

            # TODO
            # here there should be splitting of the text in a single line
            # text before date tag -> old partial_doc
            # text after date tag -> new partial_doc

            # TODO
            # also joining words that are on multiple lines...

            # start a NEW doc (bc date was found)
            partial_doc = {}
            text_content = [i.get_text() for i in line]
            partial_doc['date'] = doc_date
            partial_doc['text'] = text_content

            # last increment in the file
            # sometimes needs to be added to last document worked on manually
            # this happens, when last increment contains a date
            #     a new partial doc is created in if dates_found:
            #     but, it is never added to the results list
            #
            #     TODO: when these two lines move to the back, 
            #     number of documents increases by one (1791_Purm_Louw_03.xml).
            #     Don't know why.
            if i+1 == len(increments):
                docs.append(partial_doc)

        else:
            # if there's something in cache 
            # aka PREVIOUS doc is not empty = being worked on
            if partial_doc:
                # add text lines to it
                text_content = [i.get_text() for i in line]
                partial_doc['text'].extend(text_content)
            
            # if there's nothing in cache
            # this happens for first lines of a file, before a date has appears
            else:
                # start a fresh doc/cache (seems unnessssary)
                partial_doc = {}
                # extract and add
                text_content = [i.get_text() for i in line]
                # fake date. TODO input something else?
                partial_doc['date'] = 'NaN_before_date'
                partial_doc['text'] = text_content

    return docs


def parse_chronicle(path):

    with open(path, 'r') as f_in:
        soup = BeautifulSoup(f_in, 'lxml')

    # extract call_nr: {YYYY}_{LOCATION_TAG}_{AUTHOR_TAG}
    title_tags = soup.find_all('title')
    # one call_nr must be present in file (see tests/test_parsing)
    call_nr = title_tags[0].get_text()

    increments = soup.find_all('l')
    res = delimitation_experiment1(increments)
    # add source
    # HACK i really don't like that it's inplace
    [doc.update({'call_nr': call_nr}) for doc in res]

    return res


# %%
data_dir = 'data/corpus_220222_corrected'

xml_paths = [os.path.join(data_dir, path)
             for path in os.listdir(data_dir) if path.endswith('.xml')]

all_events = []
for path in tqdm(xml_paths):
    try:
        chron = parse_chronicle(path)
        all_events.extend(chron)
    except:
        msg.fail(f'file failed: {path}')

with open('data/primitives_220308/primitives.ndjson', 'w') as fout:
    ndjson.dump(all_events, fout)
