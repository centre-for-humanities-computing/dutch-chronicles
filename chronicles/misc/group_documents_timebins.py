'''
Select a temporal resolution & group documents into bins of that size.
Output should be a list of document ids 


Possible solutions
------------------

Pandas hack : add a thousand years to all timestamps so it can handle it

Str hack : iterate over YYYY-MM?
'''
# %%
# the method



# %%
# use case
import datetime
import ndjson
import pandas as pd

with open('/Users/au582299/Repositories/dutch-chronicles/data/primitives_220331/primitives_corrected_daily.ndjson') as fin:
    primitives = ndjson.load(fin)

# %%
# define intervals
datetag = datetime.datetime.strptime(primitives[0]['clean_date'], '%Y-%m-%d')
datetag.isocalendar().week

# %%
# pandas hack
stamps = [doc['clean_date'] for doc in primitives]

stamps_hacked = []
for tag in stamps:
    hacked_year = int(tag[0:4]) + 500
    hacked_tag = str(hacked_year) + tag[4::]
    stamps_hacked.append(hacked_tag)

# %%
df_prim = pd.DataFrame(primitives)
df_prim['date_hacked'] = pd.to_datetime(stamps_hacked)

# %%
df_resampled = (
    df_prim[['date_hacked', 'id']]
    .resample('1W')
    .sum()
    .rename(columns={0: 'doc_ids'})
)
# %%
