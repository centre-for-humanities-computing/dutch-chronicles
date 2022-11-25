"""
Segment documents in XML files
"""
from typing import List
import bs4


def detect_date_locator(tag):
    """
    Help function, finding key to extract date tag from an annotated field.

    line : bs4.element.Tag
        can behave like a list
    """
    # get correct tag to use
    if tag.has_attr('datum'):
        locator = 'datum'
    elif tag.has_attr('when'):
        locator = 'when'
    else:
        locator = None

    return locator


def extract_date_attr(tag):
    """
    Help function, actually extract date labels

    line : bs4.element.Tag
        can behave like a list
    """

    # extract, depending on if list or not
    dates = []
    for date in tag:
        locator = detect_date_locator(date)
        if locator:
            dates.append(date[locator])

    return dates


def delimitation_experiment1(increments) -> List[dict]:
    """
    text corresponding to a primitive -> `{date_1} document_1 {date_2} document_2`

    Parameters
    ----------
    increments : bs4.element.ResultSet
        iterable of bs4.element.Tag
    """
    # check input is not completely outlandish
    assert (isinstance(increments, bs4.element.ResultSet))
    assert (isinstance(increments[0], bs4.element.Tag))

    docs = []  # dump for the results
    partial_doc = {}  # create an empty extraction doc that gives a False
    for i, line in enumerate(increments):

        # detect dates in line
        dates_found = line.find_all('datum')

        # if datum is found, start a new doc
        if dates_found:
            # use date tag extraction function to handle exception
            doc_date = extract_date_attr(dates_found)

            if partial_doc:
                # add cache (PREVIOUS doc) to results
                docs.append(partial_doc)

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
            if i + 1 == len(increments):
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
                # start a fresh doc/cache (seems unnecessary)
                partial_doc = {}
                # extract and add
                text_content = [i.get_text() for i in line]
                # fake date
                partial_doc['date'] = 'NaN_before_date'
                partial_doc['text'] = text_content

    return docs


def delimitation_experiment2(increments) -> List[dict]:
    """
    text corresponding to a primitive -> `{date_1} document_1 {date_2} document_2`
    Starts a new event if date tag is e.g. in the middle of a line.

    Parameters
    ----------
    increments : bs4.element.ResultSet
        iterable of bs4.element.Tag
    """
    # check input is not completely outlandish
    assert(isinstance(increments, bs4.element.ResultSet))
    assert(isinstance(increments[0], bs4.element.Tag))

    docs = []  # dump for the results
    partial_doc = {}  # create an empty extraction doc that gives a False
    for i, line in enumerate(increments):

        # detect dates in line
        dates_found = line.find_all('datum')
        # extract content of a line to iterate over
        line_contents = [i for i in line]

        # if datum is found, start a new doc
        if dates_found:
            # in increment, at which positions do date-tags appear?
            date_indices = [line_contents.index(tag) for tag in dates_found]

            # HANDLING INCREMENTS NOT STARTING WITH A DATE-TAG
            # if increment does not start with a date-tag,
            # the content until the first date-tag must be appended to previous doc 
            if 0 not in date_indices:
                # extract lines before first date-tag
                chunk_prev_doc = line_contents[0:min(date_indices)]
                # add chunk to previous doc
                if partial_doc:
                    partial_doc['text'] = partial_doc['text'] = chunk_prev_doc
                else:
                    partial_doc = {'date': 'NaN_before_date', 'text': chunk_prev_doc}

                # prepare the rest of the increment -> starts with a date-tag
                # take first date-tag and following lines
                line_contents = line_contents[min(date_indices)::]
                # recalculate indices (because line_contents changed in previous step)
                date_indices = [line_contents.index(tag) for tag in dates_found]

            # get rid of partial doc
            docs.append(partial_doc)

            # HANDLING MULTIPLE DATE-TAGS IN ONE INCREMENT
            while date_indices:
                # doc starts with the most recent unseen date-tag
                # because at the end of the while loop, 
                # date_indices[0] will be popped from the list
                doc_start_index = date_indices[0]

                # in case there are multiple date-tags (docs) remaining, 
                # take content between two date-tags
                if len(date_indices) > 1:
                    doc_end_index = date_indices[1]
                    line_contents_subset = line_contents[doc_start_index:doc_end_index]

                # if there is only one date-tag remaining, 
                # take the rest of the line_content
                else:
                    line_contents_subset = line_contents[doc_start_index::]

                # start a new document on the subset
                partial_doc = {}
                text_content = [i.get_text() for i in line_contents_subset]
                partial_doc['date'] = extract_date_attr(dates_found)[0]
                partial_doc['text'] = text_content

                if len(date_indices) > 1:
                    # add chunk to doc list
                    docs.append(partial_doc)
                # remove current date-tag from index list
                date_indices.pop(0)
                # remove current date-tag from tag list
                dates_found.pop(0)

            # use date tag extraction function to handle exception
            doc_date = extract_date_attr(dates_found)

            # last increment in the file
            # sometimes needs to be added to last document worked on manually
            # this happens, when last increment contains a date
            #     a new partial doc is created in if dates_found:
            #     but, it is never added to the results list
            if i + 1 == len(increments):
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
                # start a fresh doc/cache (seems unnecessary)
                partial_doc = {}
                # extract and add
                text_content = [i.get_text() for i in line]
                # fake date
                partial_doc['date'] = 'NaN_before_date'
                partial_doc['text'] = text_content

    return docs
