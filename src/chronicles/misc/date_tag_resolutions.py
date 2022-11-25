"""
In a parsed file with primitives, clean the datetag so that
- only events with daily resolution are considered
- only one date-tag per event is allowed

Arbitrary rule for selecting representative date-tags:
first date-tag with daily resolution is considered representative of the event.

Uncertainty of how representative a selected date-tag is will
be tracked in the newly created 'date_uncertainty' field
"""

import re
import ndjson


def extract_daily_tag(date_tag_list):

    # patterns
    pat_year_resolution = re.compile(r'\d{4}-xx-xx', flags=re.IGNORECASE)
    pat_month_resolution = re.compile(r'\d{4}-\d{2}-xx', flags=re.IGNORECASE)
    pat_day_resolution = re.compile(r'\d{4}-\d{2}-\d{2}', flags=re.IGNORECASE)

    # confirm that input is list (not to iterate though single characters)
    if not isinstance(date_tag_list, list):
        date_tag_list = [date_tag_list]

    # find out what resolutions tags are in the list
    resolutions = []
    for tag in date_tag_list:
        if pat_year_resolution.match(tag):
            resolution = 'year'
        elif pat_month_resolution.match(tag):
            resolution = 'month'
        elif pat_day_resolution.match(tag):
            resolution = 'day'
        else:
            resolution = 'broken'

        resolutions.append(resolution)

    # get only the first day tag from the list
    if 'day' in resolutions:
        # extract day tag to represent the document
        day_id = resolutions.index('day')
        clean_date_tag = date_tag_list[day_id]

        # date-tag certainty
        # highest degree – only one tag and it's day
        if len(resolutions) == 1 and all(r == 'day' for r in resolutions):
            uncertainty = 'unambiguous'
        # multiple daily tags in a single doc
        elif len(resolution) > 1 and all(r == 'day' for r in resolutions):
            uncertainty = 'multiple day events in document'
        # multiple resolutions in a single doc
        elif len(resolutions) > 1 and not all(r == 'day' for r in resolutions):
            uncertainty = 'varying resolutions'
        else:
            uncertainty = 'unknown degree'

        return clean_date_tag, uncertainty

    else:
        return None, None


def main(primitives):

    primitives_daily = []
    for doc in primitives:
        # find day tag
        clean_date_tag, uncertainty = extract_daily_tag(doc['date'])

        # if day tag is available
        if clean_date_tag:
            doc['clean_date'] = clean_date_tag
            doc['date_uncertainty'] = uncertainty
            primitives_daily.append(doc)
        # if no day tag is found, skip document
        else:
            pass

    return primitives_daily


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inputpath')
    ap.add_argument('-o', '--outputpath')
    args = vars(ap.parse_args())

    with open(args['inputpath']) as fin:
        primitives = ndjson.load(fin)

    primitives_clean = main(primitives)

    with open(args['outputpath'], 'w') as fout:
        ndjson.dump(primitives_clean, fout)
