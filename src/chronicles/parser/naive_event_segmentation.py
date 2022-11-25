"""
Naive event segmentation:
    - extract date tags
    - label them by specificity (yearly / montly / daily)
"""
import re
import pandas as pd
from bs4 import BeautifulSoup


def extract_primitives(path, document_increment):
    """
    Get all date annotations from a chronicle
    """

    with open(path, 'r') as f_in:
        soup = BeautifulSoup(f_in, 'lxml')

        # extract call_nr: {YYYY}_{LOCATION_TAG}_{AUTHOR_TAG}
        title_tags = soup.find_all('title')
        # one call_nr must be present in file (see tests/test_parsing)
        call_nr = title_tags[0].get_text()

        # page numbers
        page_nrs = [page_nr['n'] for page_nr in soup.find_all('pb')]

        primitives = []
        # there must be same number of different document increments (see tests/test_parsing)
        increments_list = soup.find_all(document_increment)
        for page_nr, increment in zip(page_nrs, increments_list):
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

                    one_date = line[date_tag_type] if date_tag_type else line.get_text(
                    )
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


def extract_dates_resolution(primitives):
    pat_year_resolution = re.compile(r'\d{4}-xx-xx', flags=re.IGNORECASE)
    pat_month_resolution = re.compile(r'\d{4}-\d{2}-xx', flags=re.IGNORECASE)
    pat_day_resolution = re.compile(r'\d{4}-\d{2}-\d{2}', flags=re.IGNORECASE)

    date_primitives = []
    for page in primitives:

        for tag in page['date']:

            if pat_year_resolution.match(tag):
                resolution = 'year'
            elif pat_month_resolution.match(tag):
                resolution = 'month'
            elif pat_day_resolution.match(tag):
                resolution = 'day'
            else:
                resolution = 'broken'

            date_record = {
                'call_nr': page['call_nr'],
                'date': tag,
                'resolution': resolution
            }
            date_primitives.append(date_record)

    df_date = pd.DataFrame(date_primitives)

    return df_date

