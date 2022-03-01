import re
import os
from bs4 import BeautifulSoup
import lxml

# get filepaths
data_dir = '../data/corpus_220222_corrected'
xml_paths = [os.path.join(data_dir, path)
             for path in os.listdir(data_dir) if path.endswith('.xml')]


def test_document_increments():
    '''
    All files must have the same number of document increments.

    Document increments
    -------------------
    pb : page numbers
        no content

    p : paragraphs
        content

    lg : lages?
        content
        nested inside p (paragraps)

    problem -> len(p) == len(pb) == len(lg)?
    '''

    equal_units_paths = []
    inequal_units_paths = []
    for doc in xml_paths:
        with open(doc, 'r') as f_in:
            soup = BeautifulSoup(f_in, 'lxml')

            pages = [page for page in soup.find_all('lg')]
            paragraphs = [par for par in soup.find_all('p')]
            page_nrs = [page_nr for page_nr in soup.find_all('pb')]

            try:
                len(pages) == len(paragraphs) == len(page_nrs)
                equal_units_paths.append(doc)
            except:
                inequal_units_paths.append(doc)

    assert len(equal_units_paths) == len(xml_paths)


def test_lxml_representation_agnosticism():
    '''
    problem -> BeautifulSoup(IN, 'lxml') == BeautifulSoup(IN)

    should pass, because without explicitely setting lxml as the parser,
    bs4 gueses it. 
    If lxml is installed, running this test should produce following warning:

    GuessedAtParserWarning: 
    No parser was explicitly specified, 
    so I'm using the best available HTML parser for this system ("lxml"). 
    This usually isn't a problem, but if you run this code on another system, 
    or in a different virtual environment, 
    it may use a different parser and behave differently.
    '''

    equal_parsing_paths = []
    inequal_parsing_paths = []
    for doc in xml_paths:
        with open(doc, 'r') as f_in:
            soup_bs4_base = BeautifulSoup(f_in)
            soup_lxml = BeautifulSoup(f_in, 'lxml')

            try:
                soup_bs4_base == soup_lxml
                equal_parsing_paths.append(doc)
            except:
                inequal_parsing_paths.append(doc)
    
    assert len(equal_parsing_paths) == len(xml_paths)


def test_one_title_per_doc():
    '''
    There can only be one title (call_nr) per doc
    '''
    pass


def test_linebreaks():
    '''
    Have multi-line words been joined? ¬

    Some chronicles were already digitized (no line breaks present).
    '''
    pass