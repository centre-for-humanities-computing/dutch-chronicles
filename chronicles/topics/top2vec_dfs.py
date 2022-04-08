import ndjson
import numpy as np
import pandas as pd

import tslearn
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# import primitives with corrected date and topic weights
with open('../../corpus/primitives_220331/events_repre_reduced.ndjson') as f:
    events = ndjson.load(f)

df_events = pd.DataFrame(events)
df_events.shape

# remove primitives that have no topic weights (shorter than 50 characters)
df_events.dropna(subset=['representation'], inplace=True)

# transpose representations to 100 separate columns
representations = pd.DataFrame(df_events.representation.to_list(), index = df_events.index)

# merge events and representations and save to csv
total = pd.merge(left=df_events, right=representations, left_index=True, right_index=True).drop(columns='representation')
total.to_csv('/work/62138/corpus/total.csv')

# scaling and smoothening
total['year'] = total['clean_date'].str[:4]
total['year'] = pd.to_numeric(total['year'])
total = total[(total['year'] >= 1450) & (total['year'] <=1800)]

columns = list(range(0,100))
year_mean = total.groupby('year')[columns].agg('mean').reset_index()
year_mean_scaled = year_mean.copy().drop(columns=columns)

for column in year_mean.iloc[:, 1:101]:
    scaled = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform([year_mean[column].values])
    year_mean_scaled[column] = scaled[0].flatten().tolist()

year_mean.to_csv('/work/62138/corpus/year_mean.csv')
year_mean_scaled.to_csv('/work/62138/corpus/year_mean_scaled.csv')