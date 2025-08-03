import json
from pathlib import Path

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
TIMES = ["morning", "afternoon", "evening", "night"]
SEASONS = ["spring", "summer", "fall", "winter"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


TIMES2HOURS = {"morning": 10, "afternoon": 16, "evening": 18, "night": 21}
DAY2WEEKDAY = {"mon": 0, "tue":1, "wed":2, "thu":3, "fri":4, "sat":5, "sun":6}
#{"Spring": 0, "Summer": 1, "Fall": 2, "Winter": 3}


rf_model_column_names = ['hour', 'weekday', 'is_weekend', 'duration_min', 'month', 'adult',
       'missing_release_date', 'revenue', 'missing_tmdb', 'vote_average',
       'popularity', 'is_movie', 'genre_Action', 'genre_Animation',
       'genre_Aventure', 'genre_Comédie', 'genre_Crime', 'genre_Documentaire',
       'genre_Drame', 'genre_Familial', 'genre_Fantastique', 'genre_Guerre',
       'genre_Histoire', 'genre_Horreur', 'genre_Musique', 'genre_Mystère',
       'genre_Romance', 'genre_Science-Fiction', 'genre_Thriller',
       'genre_Téléfilm', 'genre_Western', 'movie_age', 'channel_France 2',
       'channel_France 3', 'channel_M6_T_PL', 'channel_RTS 1', 'channel_RTS 2',
       'channel_TF1_T_PL', 'season_fall', 'season_spring', 'season_summer',
       'season_winter', 'original_language_ab', 'original_language_ar',
       'original_language_da', 'original_language_de', 'original_language_en',
       'original_language_es', 'original_language_fa', 'original_language_fi',
       'original_language_fr', 'original_language_he', 'original_language_hi',
       'original_language_hr', 'original_language_is', 'original_language_it',
       'original_language_ja', 'original_language_ko', 'original_language_nl',
       'original_language_no', 'original_language_pl', 'original_language_pt',
       'original_language_ru', 'original_language_sv', 'original_language_th',
       'original_language_unknown', 'original_language_xx']

rf_model_column_names_context_features = ['hour', 'weekday', 'is_weekend', 'month', 'season_fall',
       'season_spring', 'season_summer', 'season_winter', 'channel_France 2', 'channel_France 3', 'channel_M6_T_PL',
       'channel_RTS 1', 'channel_RTS 2', 'channel_TF1_T_PL']

rf_model_column_names_film_features = [x for x in rf_model_column_names if x not in rf_model_column_names_context_features]

genre_columns = [col for col in rf_model_column_names if col.startswith("genre")]
original_language_columns = [col for col in rf_model_column_names if col.startswith("original_language")]

# Competititon Channels
COMPETITOR_CHANNELS = [
    "France 2", "France 3", "M6_T_PL", "TF1_T_PL"
]

INTEREST_CHANNELS = ['RTS 1', 'RTS 2']

MAX_AUDIENCE_RATING = 366

""""
MAB_FEATURE_NAMES = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                   'month_sin', 'month_cos', 'season_spring', 'season_summer', 'season_autumn', 'season_winter',
                   'norm_revenue', 'norm_vote_avg', 'norm_popularity', 'norm_duration',
                     'norm_movie_age', 'genre_Action', 'genre_Animation', 'genre_Aventure',
                     'genre_Comédie', 'genre_Crime', 'genre_Documentaire', 'genre_Drame',
                     'genre_Familial', 'genre_Fantastique', 'genre_Guerre', 'genre_Histoire',
                     'genre_Horreur', 'genre_Musique', 'genre_Mystère', 'genre_Romance',
                     'genre_Science-Fiction', 'genre_Thriller', 'genre_Téléfilm',
                     'genre_Western', 'original_language_ab', 'original_language_ar',
                     'original_language_da', 'original_language_de', 'original_language_en',
                     'original_language_es', 'original_language_fa', 'original_language_fi',
                     'original_language_fr', 'original_language_he', 'original_language_hi',
                     'original_language_hr', 'original_language_is', 'original_language_it',
                     'original_language_ja', 'original_language_ko', 'original_language_nl',
                     'original_language_no', 'original_language_pl', 'original_language_pt',
                     'original_language_ru', 'original_language_sv', 'original_language_th',
                     'original_language_unknown', 'original_language_xx',
                     'reward_audience', 'reward_competition', 'reward_diversity', 'reward_novelty','reward_rights']
"""

MAB_FEATURE_NAMES = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                   'month_sin', 'month_cos', 'season_spring', 'season_summer', 'season_autumn', 'season_winter',
                   'norm_revenue', 'norm_vote_avg', 'norm_popularity', 'norm_duration',
                     'norm_movie_age', 'genre_Action', 'genre_Animation', 'genre_Aventure',
                     'genre_Comédie', 'genre_Crime', 'genre_Documentaire', 'genre_Drame',
                     'genre_Familial', 'genre_Fantastique', 'genre_Guerre', 'genre_Histoire',
                     'genre_Horreur', 'genre_Musique', 'genre_Mystère', 'genre_Romance',
                     'genre_Science-Fiction', 'genre_Thriller', 'genre_Téléfilm',
                     'reward_audience', 'reward_competition', 'reward_diversity', 'reward_novelty','reward_rights']

path = Path("data/constants/movie_catalog_columns.json")
movie_catalog_columns = json.loads(path.read_text())

BASE_CUSTOM_ID = 'XF_000_'

context_feature_names = ['prime_time', 'late_night', 'afternoon', 'morning', 'monday', 'tuesday', 'wednesday', 
                         'thursday', 'friday', 'saturday', 'sunday', 'is_weekend', 'spring', 'summer', 'autumn', 'winter']

"""
movie_feature_names = ['norm_revenue', 
       'norm_vote_avg', 'norm_popularity', 'norm_duration',
       'norm_movie_age', 'genre_Action', 'genre_Animation', 'genre_Aventure',
       'genre_Comédie', 'genre_Crime', 'genre_Documentaire', 'genre_Drame',
       'genre_Familial', 'genre_Fantastique', 'genre_Guerre', 'genre_Histoire',
       'genre_Horreur', 'genre_Musique', 'genre_Mystère', 'genre_Romance',
       'genre_Science-Fiction', 'genre_Thriller', 'genre_Téléfilm',
       'genre_Western']
"""

# Without norm popularity
movie_feature_names = ['norm_revenue', 
       'norm_vote_avg', 'norm_duration',
       'norm_movie_age', 'genre_Action', 'genre_Animation', 'genre_Aventure',
       'genre_Comédie', 'genre_Crime', 'genre_Documentaire', 'genre_Drame',
       'genre_Familial', 'genre_Fantastique', 'genre_Guerre', 'genre_Histoire',
       'genre_Horreur', 'genre_Musique', 'genre_Mystère', 'genre_Romance',
       'genre_Science-Fiction', 'genre_Thriller', 'genre_Téléfilm',
       'genre_Western']