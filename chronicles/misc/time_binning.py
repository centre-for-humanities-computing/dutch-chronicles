'''
Group list of dict by date
'''
# %%
import re
import ndjson
import pandas as pd

# %%
# sample run
with open('/Users/au582299/Repositories/dutch-chronicles/data/primitives_220329/primitives_annotated.ndjson') as fin:
    primitives = ndjson.load(fin)

# # HACK consider only first date per primitive
# primitives_one_day_per_doc = []
# for doc in primitives:
#     if len(doc['date']) > 1:
#         doc['date'] = doc['date'][0]
#     else:
#         pass

#     primitives_one_day_per_doc.append(doc)

# %%
# TODO drop invalid dates using naive_event_segmentation.py

def extract_dates_resolution(primitives):
    pat_year_resolution = re.compile(r'\d{4}-xx-xx', flags=re.IGNORECASE)
    pat_month_resolution = re.compile(r'\d{4}-\d{2}-xx', flags=re.IGNORECASE)
    pat_day_resolution = re.compile(r'\d{4}-\d{2}-\d{2}', flags=re.IGNORECASE)

    date_primitives = []
    for doc in primitives:

        for tag in doc['date']:

            if pat_year_resolution.match(tag):
                resolution = 'year'
            elif pat_month_resolution.match(tag):
                resolution = 'month'
            elif pat_day_resolution.match(tag):
                resolution = 'day'
            else:
                resolution = 'broken'

            date_record = {
                'call_nr': doc['call_nr'],
                'doc_id': doc['id'],
                'date': tag,
                'resolution': resolution
            }
            date_primitives.append(date_record)

    df_date = pd.DataFrame(date_primitives)

    return df_date


a = extract_dates_resolution(primitives)
a_day = a.query('resolution == "day"').drop_duplicates()
day_ids = a_day['doc_id'].tolist()

# %%
primitives_day = [primitives[i] for i in day_ids]

# make sure there is on datetag per file
for doc in primitives_day:
    if len(doc['date']) > 1 and isinstance(doc['date'], list):
        doc['date'] = doc['date'][0]
    else:
        pass

# %%
import datetime

datetime.datetime.strptime(primitives_day[0]['date'], '%Y-%m-%d')

# %%
# see dates
ex_dates = [doc['date'] for doc in primitives_day]
df = pd.DataFrame(ex_dates)

# %%
