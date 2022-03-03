# %%
from itertools import groupby
import re

import ndjson
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# %%
# get primitives
with open('../data/primitives_220303/primitives.ndjson') as fin:
    primitives = ndjson.load(fin)

# %%
'''
Split date tags into:
- day specific
- month specific
- year specific
'''
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

# %%
# resolution expl
df_date.groupby(['resolution']).describe()

# %%
# broken tags expl
(df_date
    .groupby(['resolution', 'call_nr'])
    .describe()
    .query('resolution == "broken"')
    .sort_values(by=[('date', 'count')], ascending=False)
)

# %%
# dates across chronicles
(df_date
    .query('resolution == "day"')
    .groupby('date')
    .size()
    .to_frame(name='count')
    .sort_values(by='count', ascending=False)
    .query('count > 1')
)

# %%
# broken chronicle 1: gent police
# non-standardized annotation (e.g. 1003 instead of 1003-xx-xx )
(df_date
    .query('call_nr == "1668_Gent_Bill_07"')
    .query('resolution == "broken"')
    .groupby('date')
    .size()
    .to_frame(name='count')
    .sort_values(by='count', ascending=False)
)
# %%
# broken chronicle 2: brussels
# tagged, but empty attributes (e.g. jaer LXX)
(df_date
    .query('call_nr == "1602_Brus_Pott"')
    .query('resolution == "broken"')
    .groupby('date')
    .size()
    .to_frame(name='count')
    .sort_values(by='count', ascending=False)
)

# %%
###
### years: unique vs non-unique density
###

# get daily dates
df_date_daily = df_date.query('resolution == "day"')

# summarize date occurances
days = (df_date_daily
    .groupby('date')
    .size()
    .to_frame(name='count')
    .reset_index()
)
# tag unique
days['unique'] = [False if n > 1 else True for n in days['count'].tolist()]

# get rid of ridiculous dates 
days['year'] = [int(re.match(r'\d{4}', tag).group(0)) for tag in days['date'].tolist()]
days = days.query('year >= 1400 & year < 1800')

## convert to datetime
## DOESN'T WORK: OUT OF BOUNDS. datetime64 starts in 1677
# dates_datetime = []
# invalid_dates = []
# for tag in days['date'].tolist():
#     try:
#         parsed_tag = pd.to_datetime(tag, format='%Y-%m-%d')
#         dates_datetime.append(parsed_tag)
#     except ValueError:
#         dates_datetime.append(None)
#         invalid_dates.append(tag)

# %%
# days_yearly_cnct = days.groupby('year')
props_unique_yearly = []
for year_tag, year_data in days.groupby('year'):
    n_records = len(year_data)
    proportion_unique = sum(year_data['unique']) / n_records
    props_unique_yearly.append({
        'year': year_tag,
        'n_dates': n_records,
        'proportion_unique': proportion_unique, 
    })

props_unique_yearly = pd.DataFrame(props_unique_yearly)

# plot
sns.barplot(
    x=props_unique_yearly['year'],
    y=props_unique_yearly['n_dates']
)

# %%
###
### decades: unique vs non-unique density
###
days['decade'] = [re.match(r'\d{4}', tag).group(0) for tag in days['date'].tolist()]
days['decade'] = [int(year[0:3]) for year in days['decade'].tolist()]

props_unique_decade= []
for decade_tag, decade_data in days.groupby('decade'):
    n_records = len(decade_data)
    proportion_unique = sum(decade_data['unique']) / n_records
    props_unique_decade.append({
        'decade': decade_tag,
        'n_dates': n_records,
        'proportion_unique': proportion_unique, 
    })

props_unique_decade = pd.DataFrame(props_unique_decade)

# %%
### just number of records
plt.figure(figsize=(10, 6))
sns.barplot(
    x=props_unique_decade['decade'],
    y=props_unique_decade['n_dates'],
    color='#A7CECB'
)
plt.xticks(rotation=90)

# %%
### just proportion
plt.figure(figsize=(10, 6))
props_unique_decade['top'] = 1
sns.barplot(
    x=props_unique_decade['decade'],
    y=props_unique_decade['top'],
    color='#AE5377'
)
sns.barplot(
    x=props_unique_decade['decade'],
    y=props_unique_decade['proportion_unique'],
    color='#F2D1C9'
)
plt.xticks(rotation=90)

# %%
### 
plt.figure(figsize=(10, 6))
props_unique_decade['top'] = 1
sns.barplot(
    x=props_unique_decade['decade'],
    y=props_unique_decade['n_dates'],
    color='#AE5377'
)
sns.barplot(
    x=props_unique_decade['decade'],
    y=props_unique_decade['n_dates'] * props_unique_decade['proportion_unique'],
    color='#F2D1C9'
)
plt.legend()
plt.xticks(rotation=90)
plt.title('')

# %%
