import pandas as pd
from constants import *
import json
from datetime import date, datetime
from api import tmdb
from constants import rf_model_column_names
from typing import List

with open('data/holidays.json', 'r') as file:  
    holidays = json.load(file)

catalog_df = pd.read_pickle("data/catalog_df.pkl")

def is_public_holiday(date: datetime, holidays):
    for canton in holidays['cantons']:
        # Check if the date is a public holiday in the canton
        if date in [day['date'] for day in holidays['cantons'][canton]['public_holidays']]:
            return True
        
        # Check if the date falls within any school vacation period
        for vacation in holidays['cantons'][canton]['school_vacations']:
            start = vacation['start']
            end = vacation['end']
            #Convert start and end dates to Timestamps
            start = pd.Timestamp(vacation['start'])
            end = pd.Timestamp(vacation['end'])
            if start <= date <= end:
                return True
            
    return False

def get_season(month: int):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
    
def context2dict(context: dict) -> dict:
    d = {}
    if context['day'] is not None:
        #day = context['day']
        #day = datetime.strptime(context['day'], "%Y-%m-%d")
        #d['public_holiday'] = 1 if is_public_holiday(context['day'], holidays) else 0
        #d['weekday'] = day.weekday()
        d['weekday'] = DAY2WEEKDAY[context['day']]
        
    #else:
        #d['public_holiday'] = 0 # If specific date not given assume not public holiday
        #d['weekday'] = DAY2WEEKDAY[context['weekday']]
        #day = date.today()

    d['hour'] = TIMES2HOURS[context['time']]
    #d['month'] = day.month
    d['month'] = MONTHS.index(context['month'])+1 # Starting at January is 1
    #d['year'] = day.year
    #d['doy'] = day.timetuple().tm_yday
    d['is_weekend'] = 1 if d['weekday'] > 3 else 0
    #d['public_holiday'] = 1 if is_public_holiday(day, holidays) else 0
    d['season'] = get_season(d['month'])
    #d['season'] = context['season']

    return d

def get_film_features(title: str) -> dict:
    # Call TMDB API
    tmdb_id = tmdb.find_best_match(title)
    if tmdb_id is not None:
        film_features = tmdb.get_movie_features(tmdb_id)
        return film_features
    else:
        return None

def build_model_input(context: dict, movie_id: int = None, title: str = None) -> pd.DataFrame:
    """
    Use movie_id when searching within the catalog_df.
    Use title when searching TMDB for film features."""

    assert movie_id is not None or title is not None, "Either movie_id or title must be provided"
    context_dict = context2dict(context)
    context_df = pd.DataFrame.from_dict([context_dict])

    prefix = 'season_'
    season_dummies = [x for x in rf_model_column_names if prefix in x]
    for season in season_dummies:
        if context_df['season'][0] in season:
            context_df[season] = 1
        else: 
            context_df[season] = 0
    if title is not None:
        film_features = get_film_features(title)
    else:
        film_features = catalog_df.iloc[movie_id].to_dict()
        
    film_features['genre_names'] = [genre['name'] for genre in film_features['genres']]
    ff_df = pd.DataFrame.from_dict([film_features])

    ff_df["release_date_dt"] = pd.to_datetime(
        ff_df["release_date"],
        format="%Y-%m-%d",
        )

    ff_df["release_year"]  = ff_df["release_date_dt"].dt.year
    ff_df['age'] = pd.Timestamp('today').year - ff_df["release_year"]

    ff_df = ff_df.drop(columns=['release_date', 'release_date_dt'])

    old_column_names = ff_df.columns.tolist() + context_df.columns.tolist()
    new_column_names = [x for x in rf_model_column_names if x not in old_column_names]

    # Set all dummies to null values
    for c in new_column_names:
        ff_df[c] = 0

    # Add genre dummies
    prefix = 'genre_'
    for g in ff_df['genre_names'][0]:
        col_name = prefix + g
        ff_df[col_name] = 1

    # Add original language dummy
    s = 'original_language_'+ff_df['original_language']
    ff_df[s] = 1

    # Put context and film features together
    full_df = pd.concat([ff_df, context_df], axis=1)
    diff_cols = [x for x in full_df.columns if x not in rf_model_column_names]
    #print(f"Columns in full_df not in rf_model_column_names: {diff_cols}")
    full_df = full_df[rf_model_column_names]

    #print(f"additional feature: {[x for x in full_df.columns.tolist() if x not in rf_model_column_names]}")
    assert len(full_df.columns) == len(rf_model_column_names), f"Problem with dataframe columns size {len(full_df.columns)} not matching size rf_model_column_names {len(rf_model_column_names)}"
    full_df = full_df[rf_model_column_names]

    return full_df

def build_model_input_batch(contexts: List[dict], movie_ids: List[int] = None, titles: List[str] = None) -> pd.DataFrame:
    """
    Use movie_ids when searching within the catalog_df.
    Use titles when searching TMDB for film features.
    """
    assert len(contexts) == len(movie_ids) or len(contexts) == len(titles), "Contexts must match movie_ids or titles length"
    
    inputs = []
    for i, context in enumerate(contexts):
        if movie_ids is not None:
            movie_id = movie_ids[i]
            title = None
        else:
            movie_id = None
            title = titles[i]
        inputs.append(build_model_input(context, movie_id=movie_id, title=title))
    
    return pd.concat(inputs, ignore_index=True)
