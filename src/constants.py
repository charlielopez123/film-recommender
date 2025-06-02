DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
TIMES = ["morning", "afternoon", "evening", "night"]
SEASONS = ["Spring", "Summer", "Fall", "Winter"]

TIMES2HOURS = {"morning": 10, "afternoon": 16, "evening": 18, "night": 21}
DAY2WEEKDAY = {"mon": 0, "tue":1, "wed":2, "thu":3, "fri":4, "sat":5, "sun":6}
#{"Spring": 0, "Summer": 1, "Fall": 2, "Winter": 3}


rf_model_column_names = ['hour', 'weekday', 'is_weekend', 'duration', 'adult',
       'missing_release_date', 'revenue', 'missing_tmdb', 'vote_average',
       'popularity', 'is_movie', 'genre_Action', 'genre_Animation',
       'genre_Aventure', 'genre_Comédie', 'genre_Crime', 'genre_Documentaire',
       'genre_Drame', 'genre_Familial', 'genre_Fantastique', 'genre_Guerre',
       'genre_Histoire', 'genre_Horreur', 'genre_Musique', 'genre_Mystère',
       'genre_Romance', 'genre_Science-Fiction', 'genre_Thriller',
       'genre_Téléfilm', 'genre_Western', 'age', 'release_year', 'season_Fall',
       'season_Spring', 'season_Summer', 'season_Winter',
       'original_language_ab', 'original_language_ar', 'original_language_da',
       'original_language_de', 'original_language_en', 'original_language_es',
       'original_language_fa', 'original_language_fi', 'original_language_fr',
       'original_language_is', 'original_language_it', 'original_language_ja',
       'original_language_no', 'original_language_pl', 'original_language_pt',
       'original_language_ru', 'original_language_sh', 'original_language_tl',
       'original_language_unknown', 'original_language_xx']

rf_model_column_names_context = ['hour', 'weekday', 'is_weekend', 'season_Fall', 'season_Spring', 'season_Summer', 'season_Winter']


rf_model_column_names_film = [x for x in rf_model_column_names if x not in rf_model_column_names_context]