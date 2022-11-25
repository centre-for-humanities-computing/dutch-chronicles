"""Functions for splitting dates and extracting week numbers"""
import pandas as pd
import datetime
from typing import List, Union


def add_week(data: Union[List[str], pd.DataFrame]) -> List[int]:
    """
    Identifies the week number based on a given date.

    Args:
        data(Union[List[str], pd.DataFrame]): A list or dataframe column with dates to split.
    Returns:
        weeks([List[int]): A list with week numbers.
    """

    weeks = []

    for date in data:
        try:
            datetag = datetime.datetime.strptime(date, "%Y-%m-%d")
            week = datetag.isocalendar()[1]
            weeks.append(week)
        except ValueError:
            weeks.append(0)

    return weeks


def split_date(
    data: Union[List[str], pd.DataFrame]
) -> Union[List[str], List[str], List[str]]:
    """
    Splits date into day, month and year.

    Args:
        data(Union[List[str], pd.DataFrame]): A list or dataframe column with dates to split.
    Returns:
        years, months, days(Union[List[str], List[str], List[str]]): Lists with years, months and days.
    """

    years = []
    months = []
    days = []

    for date in data:
        splitted_date = date.split("-")
        years.append(int(splitted_date[0]))
        months.append(int(splitted_date[1]))
        days.append(int(splitted_date[2]))

    return years, months, days


def parse_dates(data: Union[List[str], pd.DataFrame], inplace=False, df=None):
    '''Parse YYYY-MM-DD dates.
    Returns lists of year, month, week & day

    Parameters
    ----------
    data : List[dict]
        primitives
    inplace : bool
        append date column to an existing dataframe?
    df : pd.DataFrame
        if inplace, dates will be appended to this df

    '''
    weeks = add_week(data)
    years, months, days = split_date(data)

    if inplace:
        df['year'] = years
        df['month'] = months
        df['week'] = weeks
        df['day'] = days

        return df

    else:
        return years, months, weeks, days
