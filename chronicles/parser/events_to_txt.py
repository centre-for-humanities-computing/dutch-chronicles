# %%
import os
import ndjson
import re
import shutil
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
                try:
                    text_content = [i.get_text() for i in line]
                except:
                    print(line)
                # fake date. TODO input something else?
                partial_doc['date'] = 'NaN_before_date'
                partial_doc['text'] = text_content

    return docs


def parse_chronicle(path):

    with open(path, 'r', encoding='utf-8') as f_in:
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

# create documents per event, and merge hyphen

data_dir = '/Users/alielassche/documents/github/chronicling-topics/corpus/corpus_220222_annotated'
error_dir = '/Users/alielassche/documents/github/chronicling-topics/corpus/errors'
os.chdir('/Users/alielassche/documents/github/chronicling-topics/corpus/txt')
xml_paths = [os.path.join(data_dir, path)
             for path in os.listdir(data_dir) if path.endswith('.xml')]

all_events = []
for path in tqdm(xml_paths):
    try:
        chron = parse_chronicle(path)
        for i in range(len(chron)):
            try:
                chrontxt = chron[i]['text']
                chrontxt = ' '.join(chrontxt)
                chrontxt = re.sub(r"((¬#?) ?)", "", chrontxt)
                filename = str(chron[i]['call_nr']) + '_' + str(chron[i]['date'])[2:12] + '.txt'
                with open(filename, 'a', encoding='utf-8') as fileout:
                    fileout.write(chrontxt)
            except:
                continue
    except:
        msg.fail(f'file failed: {path}')
        shutil.copy(path, error_dir)

    
# %%
import glob
import nltk
import re
import string
import shutil
import csv
from nltk import FreqDist

TOKENIZER = nltk.tokenize.word_tokenize

def is_punct(t):
    return re.match(f'[{string.punctuation}]+$', t) is not None

# %%

# filter documents with more than 10 words and make mfw list with 250 most common words

words_total = []
n = 0
new_dir = ('/work/corpus/txt_10')


os.chdir('/work/corpus/txt')
for doc in glob.glob("*.txt"):
    text = open(doc, "r", encoding="utf-8").read()
    words = []
    for sentence in TOKENIZER(text, language="dutch"):
        words.extend([w.lower() for w in sentence.split() if not is_punct(w)])
        words_total.extend([w.lower() for w in sentence.split() if not is_punct(w)])
    if len(words) >= 10:
        n += 1
        shutil.copy(doc, new_dir)
print(n)

fd = FreqDist(words_total)
mfw = fd.most_common(250)

# %%
MFW = csv.writer(open('mfw_events.csv', 'w', encoding='utf-8'))
for key, count in mfw:
    MFW.writerow([key, count])

# %%

import glob
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

corpus_path = '/work/corpus/txt_10/*.txt'
stopwords_path = '/work/corpus/stoplist.txt'
stopwords = [s.lower() for s in open(stopwords_path, 'r', encoding='utf-8').read().splitlines()]

remove_stopwords = lambda x: [word.lower() for word in x if word.lower() not in stopwords and not is_punct(word) and len(word) > 1 and word.isalpha()]

# %%

texts = glob.glob(corpus_path, recursive=False)
tokenized_texts = [TOKENIZER(open(text, "r", encoding="utf-8").read(), language="dutch") for text in texts]
tokenized_texts = [remove_stopwords(text) for text in tokenized_texts]

# %%

no_below = 2
no_above = 1

n_topics = 20
iterations = 2000
eval_every = 3

# %%

dictionary = Dictionary(tokenized_texts)
dictionary.filter_extremes(no_below=no_below, no_above=no_above)
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

# %%
lda = LdaModel( corpus=corpus,
                id2word=dictionary,
                num_topics=n_topics, 
                iterations=iterations
                )

# %%

# calculate coherence score

cm = CoherenceModel(model=lda, corpus=corpus, texts=tokenized_texts, dictionary=dictionary, coherence ='c_v')
coherence = cm.get_coherence()
print('\nCoherence Score: ', coherence)

# %%

# calculate coherence score for range

def compute_coherence_values(dictionary, corpus, texts, limit, start, step):

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics, 
                iterations=iterations,  
                )
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# %%
model_list, coherence_values = compute_coherence_values(
                                dictionary=dictionary,
                                corpus=corpus,
                                texts=tokenized_texts,
                                start=10, limit=55, step=5)

limit=55; start=10; step=5
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Number of topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.savefig('/work/dutch-chronicles/output/coherence_values.pdf')
