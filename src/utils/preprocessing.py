import numpy as np
import pandas as pd
from api.tmdb import find_best_match, get_movie_title, get_movie_features

def preprocess_featured_movies(featured_movies_df):
    # Adult = False as default
    featured_movies_df['adult'] = np.where(
        featured_movies_df['adult'].isna(),         # condition per-row
        False,                                      # value if True
        featured_movies_df['adult']                 # value if False
    )

    # Missing original_language put as 'unknown'
    featured_movies_df['original_language'] = np.where(
        featured_movies_df['original_language'].isna(),         # condition per-row
        'unknown',                                              # value if True
        featured_movies_df['original_language']                 # value if False
    )

    featured_movies_df['genres'] = featured_movies_df['genres'].apply(
        lambda x: x if isinstance(x, list) or isinstance(x, np.ndarray) else []
    )

    # Missing release_date put as '1900-01-01'
    featured_movies_df['release_date'] = np.where(
        featured_movies_df['release_date'].isna(),         # condition per-row
        '1900-01-01',                                      # value if True
        featured_movies_df['release_date']                 # value if False
    )

    # Add a missing_release_date flag
    featured_movies_df.loc[:, 'missing_release_date'] = np.where(
        featured_movies_df['release_date'].isna(),  # condition per-row
        False,                                      # value if True
        True                                        # value if False
    )
    featured_movies_df.loc[:, 'missing_release_date'] = featured_movies_df.loc[:, 'missing_release_date'].apply(lambda s: False if s == '' else True)
    featured_movies_df.loc[:, 'release_date'] = featured_movies_df.loc[:, 'release_date'].apply(lambda s: '1900-01-01' if s == '' else s)

    # Missing Revenue put as 0 similarly to TMDB API
    featured_movies_df['revenue'] = np.where(
        featured_movies_df['revenue'].isna(),         # condition per-row
        0,                                            # value if True
        featured_movies_df['revenue']                 # value if False
    )

    # missing tmdb id flag
    featured_movies_df.loc[:, 'missing_tmdb'] = np.where(
        featured_movies_df['tmdb_id'].isna(),  # condition per-row
        True,                                  # value if True
        False                                  # value if False
    )

    # Add vote average as zero
    featured_movies_df.loc[:, 'vote_average'] = featured_movies_df['vote_average'].fillna(0)

    # Add last_diff_rating as zero
    featured_movies_df.loc[:, 'last_diff_rating'] = featured_movies_df['last_diff_rating'].fillna(0)

    #  Add popularity zero 
    featured_movies_df.loc[:, 'popularity'] = featured_movies_df['popularity'].fillna(0)

    # Separate Movies and TV Shows
    featured_movies_df.loc[:, 'is_movie'] = True

    # Age of the movie
    dates = pd.to_datetime(featured_movies_df['release_date'])
    featured_movies_df.loc[:, 'movie_age'] = (pd.Timestamp.now() - dates).dt.days // 365

    return featured_movies_df

def search_movie_id_row(row):
    title = row['title']
    #known_runtime = row['duration_min']
    row['tmdb_id'] = find_best_match(title)
    if row['tmdb_id'] is not None:
        row['processed_title'] = get_movie_title(row['tmdb_id']) # Keep title from TMDB for consistency throughout all channels
        row['missing_tmdb_id'] = False
    else:
        row['processed_title'] = row['title'] # Fallback to original title if no match found
        row['missing_tmdb_id'] = True
    return row

def enrich_movie_feature_row(row, return_duration = False):
    tmdb_id = row["tmdb_id"]
    if pd.notna(tmdb_id):
        feats = get_movie_features(tmdb_id, return_duration)
        for k, v in feats.items():
            row[k] = v
    if return_duration:
        row['duration_min'] = row['runtime']
    return row

def match_catalog_ids(historical_data_df: pd.DataFrame, title_to_id: dict):
    # First: match via TMDB ID
    historical_data_df['catalog_id'] = historical_data_df['tmdb_id']

    # Second: fallback for rows where TMDB ID is missing or didn't match
    missing_mask = historical_data_df['catalog_id'].isna()
    historical_data_df.loc[missing_mask, 'catalog_id'] = historical_data_df.loc[missing_mask, 'title'].map(title_to_id)

    # If the movie is still missing a catalog_id, it is not in the catalog
    missing_mask_catalog = historical_data_df['catalog_id'].isna()
    historical_data_df.loc[missing_mask_catalog, 'catalog_id'] =  -1 # Assign -1 for missing catalog_id

    return historical_data_df

def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

def flatten_schedule(nested, title_to_id) -> pd.DataFrame:
    
    # Build df from scraped future competitor showings into appendable df to hsitorical_df
    rows = []
    for channel, weeks in nested.items():
        for week, shows in weeks.items():
            for show in shows:
                row = show.copy()
                row['channel'] = channel
                row['week'] = week
                rows.append(row)
    df = pd.DataFrame(rows)

    # normalize time: TF1 uses "21.15" meaning "21:15", M6 uses "21:10"
    time_fixed = df['time'].str.replace('.', ':', regex=False)
    df['hour'] = pd.to_numeric(time_fixed.str.split(':').str[0], errors='coerce').astype('Int64')

    # parse dates (TF1 is DD/MM/YYYY, M6 is YYYY-MM-DD)
    df['date'] = df['date'].apply(robust_parse)
    df['day'] =  df['date'].dt.day
    df['weekday'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['weekday']>4
    df['month'] =  df['date'].dt.month
    df['season'] = df['date'].apply(get_season)
    df['rt_m'] = 0.0
    df['content_class_key'] = '71'
    df = df.apply(search_movie_id_row, axis=1) # find tmdb ids and change corresponding 'processed_title' and 'missing_tmdb_id'
    df = df.apply(lambda row: enrich_movie_feature_row(row, return_duration=True), axis=1) # get movie features corresponding to found tmdb ids
    df = match_catalog_ids(df, title_to_id)
    
    # reorder df columns to historical_df column order
    df = df[['title', 'date', 'content_class_key', 'channel', 'duration_min', 'hour', 'day',
       'weekday', 'is_weekend', 'month', 'rt_m', 'tmdb_id', 'season',
       'processed_title', 'missing_tmdb_id', 'catalog_id']]
    
    return df



def robust_parse(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    # try fast explicit formats first
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            continue
