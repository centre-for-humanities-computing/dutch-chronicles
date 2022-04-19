"""Pipeline for grouping chronicles by date"""
import pandas as pd
import ndjson
import datetime
from process_dates import add_week, split_date

with open("primitives_corrected_daily.ndjson") as fin:
    primitives = ndjson.load(fin)

df_primitives = pd.DataFrame(primitives)

weeks = add_week(df_primitives["clean_date"])

df_primitives["weeks"] = weeks

years, months, days = split_date(df_primitives["clean_date"])
df_primitives["year"] = years
df_primitives["month"] = months
df_primitives["day"] = days

grouped_by_year = df_primitives.groupby("year")["id"].apply(list)
grouped_by_year = grouped_by_year.reset_index()
grouped_by_year = grouped_by_year.rename(columns={"id": "List_of_ids"})
grouped_by_year.head()
