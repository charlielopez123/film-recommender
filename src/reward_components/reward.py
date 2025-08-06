import pandas as pd
import numpy as np
from typing import List, Dict
from constants import *
import utils
from envs.context import Context
from competition.competitor import CompetitorDataManager
import sys
from datetime import datetime, timedelta

import utils.date_formatting

class RewardCalculator:
    def __init__(self, movie_catalog: pd.DataFrame, audience_model: None, competition_manager: CompetitorDataManager = None,
                  memory = None, historical_df = None, scaler_dict: Dict = None, active_channel: str = 'RTS 1'):
        """
        Initialize the reward calculator with a movie catalog
        
        Args:
            movie_catalog: DataFrame containing movie metadata (id, genre, country, year, etc.)
        """
        self.movie_catalog = movie_catalog
        self.audience_model = audience_model  
        self.competition_manager = competition_manager
        self.memory = memory
        self.historical_df = historical_df
        self.scaler_dict = scaler_dict
        self.active_channel= active_channel
        # Filter for your channel's programming decisions
        self.interest_channel_historical_df = self.historical_df[
            self.historical_df['channel'] == active_channel
        ].copy()

    def calculate_total_reward(self, movie_id, context, air_date, historical_row = None, available_historical_df = None, verbose = False):
        """
        Calculate multi-objective reward from historical performance data
        
        Args:
            context: Context
            competitor_context: What competitors were showing at the same time
            programming_history: Recent programming history for diversity/novelty calculation
        
        Returns:
            Combined reward score (0.0 to 1.0)
        """
        
        movie = self.movie_catalog.loc[movie_id]
        
        # Initialize reward components
        rewards = {}
        
        # 1. AUDIENCE APPEAL REWARD
        if historical_row is not None:
            air_date = historical_row['date']
            audience = historical_row.get('rt_m')
            audience_reshape = np.array([audience]).reshape(-1, 1) # Reshape for scaler compatibility
        else:
            audience  = self._calculate_audience_reward(context, movie)
            audience_reshape = (audience).reshape(-1, 1)

        rewards['audience'] = float(self.scaler_dict['rt_m'].transform(audience_reshape).squeeze())
            
        # 2. COMPETITIVE ADVANTAGE REWARD
        rewards['competition'] = self._calculate_competition_reward(air_date, movie)
        
        # 3. DIVERSITY REWARD
        rewards['diversity'] = self._calculate_diversity_reward(movie)
        
        # 4. NOVELTY REWARD
        rewards['novelty'] = self._calculate_novelty_reward(air_date, movie, interest_historical_row=historical_row, available_historical_df=available_historical_df)

        # 5. RIGHTS URGENCY REWARD
        rewards['rights'] = self._calculate_rights_urgency_reward(air_date, movie, historical_row)
        
        if verbose:
            #if use_history:
                #print('-----------------------')
            if rewards['novelty'] != 0.8 or rewards['competition'] != 0.5:
                print('-----------------------')
                print(rewards)
        # Combine rewards with weights
        reward_weights = {
            'audience': 0.4,       # Prioritizes strong viewership, but with room for public service trade-offs
            'competition': 0.15,    # Avoids clashing with competitors, but not overly reactive
            'diversity': 0.1,      # Promotes balanced representation across genres, demographics, etc.
            'novelty': 0.2,        # Encourages programming that is fresh or less repetitive
            'rights': 0.15         # Incentivizes using content with soon-to-expire rights
        }

        if rewards['audience'] > 4 or rewards['audience'] < -0.8:
            print('----------------------------')
            print(f"Case: {rewards['audience']}")
            print(audience_reshape)

        total_reward = sum(rewards[component] * reward_weights[component] for component in rewards.keys())
        if total_reward < 0.05:
            print(total_reward)
        return total_reward, rewards

    def build_model_input(self, context: Context, movie: pd.Series) -> pd.DataFrame:
        """
        row: historical_df row
        movie: catlalog_df movie row with movie features
        """
        
        model_movie_input = pd.DataFrame({
            'duration_min': [movie['duration_min']],
            'adult': [movie['adult']],
            'missing_release_date': [movie['missing_release_date']],
            'revenue': [movie['revenue']],
            'missing_tmdb': [movie['missing_tmdb']],
            'is_movie': [movie['missing_tmdb']],
            'movie_age': [movie['movie_age']],
            'popularity': [movie['popularity']],
            'vote_average': [movie['vote_average']],
        })

        genre_prefix = 'genre_'
        lang_prefix = 'original_language_'

        # Identify expected dummy columns from model
        genre_dummies = [x for x in rf_model_column_names if x.startswith(genre_prefix)]
        lang_dummies = [x for x in rf_model_column_names if x.startswith(lang_prefix)]

        # Add all dummy columns with 0 initially
        for col in genre_dummies + lang_dummies:
            model_movie_input[col] = 0

        # Set matching genre dummies to 1
        movie_genre_list = [genre['name'] for genre in movie['genres']]
        for genre_col in genre_dummies:
            if genre_col.removeprefix(genre_prefix) in movie_genre_list:
                model_movie_input[genre_col] = 1

        #Set matching original language dummy to 1
        if 'original_language' in movie:
            lang_code = movie['original_language']
            lang_col_name = f"{lang_prefix}{lang_code}"
            if lang_col_name in model_movie_input.columns:
                model_movie_input[lang_col_name] = 1


        model_context_input = pd.DataFrame({
            'hour': [context.hour],
            'weekday': [context.day_of_week],
            'is_weekend': [1 if context.day_of_week > 4 else 0],
            'month': [context.month]
        })

        # One-hot encoding for season
        season_prefix = 'season_'
        season_cols = [col for col in rf_model_column_names if col.startswith(season_prefix)]

        # Initialize season columns with 0
        for season_col in season_cols:
            model_context_input[season_col] = 0

        # Set the active season dummy to 1
        active_season_col = f"{season_prefix}{context.season}"
        if active_season_col in model_context_input.columns:
            model_context_input[active_season_col] = 1 

        # One-hot encoding for channel
        channel_prefix = 'channel_'
        channel_cols = [col for col in rf_model_column_names if col.startswith(channel_prefix)]

        # Initialize channel columns with 0
        for channel_col in channel_cols:
            model_context_input[channel_col] = 0

        # Set the active channel dummy to 1
        active_channel_col = f"{channel_prefix}{self.active_channel}"
        if active_channel_col in model_context_input.columns:
            model_context_input[active_channel_col] = 1

        model_input = pd.concat([model_movie_input, model_context_input], axis=1)
        model_input = model_input[rf_model_column_names] # The feature names should match those that were passed during fit.
                                                         # Feature names must be in the same order as they were in fit.
        return model_input

    def _calculate_audience_reward(self, context, movie) -> float:
        """Calculate reward based on audience appeal factors"""
        model_input = self.build_model_input(context, movie)
        
        assert list(model_input.columns) == rf_model_column_names, (
                f'difference in features:\n'
                f'  only in model_input: '
                f'{set(model_input.columns) - set(rf_model_column_names)}\n'
                f'  only in rf_model: '
                f'{set(rf_model_column_names) - set(model_input.columns)}'
            )
        pred_audience_rating = self.audience_model.predict(model_input)
        
        return pred_audience_rating

    def _calculate_competition_reward(self, date, movie, verbose = False) -> float:
        """Calculate reward based on competitive advantage"""
        date = utils.date_formatting.to_datetime_format(date)
        movie_competitor_context = self.competition_manager.get_movie_competitor_context(movie.name, date)
        competition_reward = 0

        #if len(movie_competitor_context.competitor_showings) > 0:
            #print(f"id: {movie.name}, num_comp_showings: {len(movie_competitor_context.competitor_showings)}")
        if len(movie_competitor_context.competitor_showings) == 0 or movie.name == -1: # if no competitor showings or movie not recognised from catalog
            return 0.0  # Neutral if no competitor info
        
        air_date = date
        if verbose:
            print("-------------------------------")
            print("Potential Competition Overlap:")
            #print(f"movie_id: {row['movie_id']} \n processed_title: {row['processed_title']} \n Air date: {air_date} ")
        
        
        # Get competing programs
        for comp_showing in movie_competitor_context.competitor_showings:
            comp_date = comp_showing['air_date']
            date_diff_days = (utils.date_formatting.to_datetime_format(air_date) - utils.date_formatting.to_datetime_format(comp_date)).days


            # SAME WEEK BEFORE COMPETITION (best window)
            if -7 <= date_diff_days < 0:
                reward = 1 * np.exp(date_diff_days / 2)  # Peak ~0.75 near -2, ~1 near -1
            # ONE TO FOUR WEEKS BEFORE
            elif -21 < date_diff_days < -7:
                reward = 0.158 * np.exp((date_diff_days + 14) / 10) + 0.1  # smoother decay
            # SAME DAY
            elif date_diff_days == 0:
                reward = -0.3
            # SAME WEEK AFTER (strong penalty)
            elif 0 < date_diff_days <= 7:
                reward = -1.0
            # ONE TO THREE WEEKS AFTER (linearly improving)
            elif 7 < date_diff_days <= 21:
                reward = -1.0 + (date_diff_days - 7) * (1.0 / 30)
            # ONE TO SIX MONTHS AFTER (mild exponential penalty)
            elif 21 < date_diff_days <= 180:
                reward = -0.5 * np.exp(-(date_diff_days - 21) / 60)  # slow decay toward 0
            else:
                reward = 0  # No effect if too far before or after

            if verbose:
                if date_diff_days < 0:
                    print(f"Showing on {comp_showing['channel']} {comp_date} -> {date_diff_days} days, after our showing")
                else:
                    print(f"Showing on {comp_showing['channel']} {comp_date} -> {date_diff_days} days, before our showing")

        competition_reward += reward

        scaled_reward = (competition_reward + 1.0) / 2.0

        return scaled_reward
    
    #TODO# Calculate competitive advantages
    def _calculate_competitive_advantage():
            pass
            
            #advantages = []
            #
            #for comp_movie in competing_movies:
            #    # Genre differentiation
            #    genre_advantage = self._calculate_genre_advantage(movie, comp_movie)
            #    
            #    # Quality advantage  
            #    quality_advantage = self._calculate_quality_advantage(movie, comp_movie)
            #    
            #    # Target audience advantage
            #    audience_advantage = self._calculate_audience_advantage(movie, comp_movie, row)
            #    
            #    # Combine advantages for this competitor
            #    comp_advantage = (genre_advantage * 0.4 + 
            #                    quality_advantage * 0.4 + 
            #                    audience_advantage * 0.2)
            #    advantages.append(comp_advantage)
            #
            ## Average advantage against all competitors
            #competition_reward = np.mean(advantages)
            #
            #return np.clip(competition_reward, 0.0, 1.0)

    def _calculate_diversity_reward(self, movie) -> float:
        """Calculate reward for programming diversity"""
        
        if self.memory is None or len(self.memory) <= 1:
            return 0.3  # Neutral if no history
        
        # Look at recent programming (e.g., last 25 movies)
        recent_programming = self.memory[-1:-25:-1] 

        # Calculate diversity metrics
        diversity_scores = []
        
        # 1. Genre diversity
        current_genres = [genre['name'] for genre in movie['genres']]
        recent_genres = [[genre['name'] for genre in self.movie_catalog.loc[movie_id]['genres']] for movie_id in recent_programming]
        

        genre_diversity = self._calculate_genre_diversity(current_genres, recent_genres)
        diversity_scores.append(genre_diversity)
        
        # 2. Country/Language diversity
        current_lang = movie['original_language']
        recent_langs = [self.movie_catalog.loc[movie_id]['original_language'] for movie_id in recent_programming]
        
        language_diversity = self._calculate_language_diversity(current_lang, recent_langs, decay = 0.9)
        diversity_scores.append(language_diversity)
        
        # 3. Era diversity (decade)
        current_era = utils.date_formatting.to_datetime_format(movie['release_date']).year//10
        recent_eras = [utils.date_formatting.to_datetime_format(self.movie_catalog.loc[movie_id]['release_date']).year//10 for movie_id in recent_programming]
    
        era_diversity = self._calculate_era_diversity(current_era, recent_eras)
        diversity_scores.append(era_diversity)

        # 4. Audience Rating Diversiy
        current_rt = movie['last_diff_rating_7']
        recent_rts = [self.movie_catalog.loc[movie_id]['last_diff_rating_7'] for movie_id in recent_programming]
        audience_rt_diversity = self._calculate_audience_rating_diversity(current_rt, recent_rts)
        diversity_scores.append(audience_rt_diversity)

        #TODO add parental control diversity
        # 4. Rating diversity (family-friendly vs mature)
        #recent_ratings = [self.movie_catalog.loc[hist_row['movie_id']].get('content_rating', 'PG') 
        #                for hist_row in recent_programming]
        #rating_diversity = self._calculate_rating_diversity(movie.get('content_rating', 'PG'), recent_ratings)
        #diversity_scores.append(rating_diversity)
        
        diversity_reward = np.mean(diversity_scores)
        return diversity_reward


    def _calculate_novelty_reward(
        self,
        air_date,
        movie: pd.Series,
        interest_historical_row: pd.Series = None,
        available_historical_df: pd.DataFrame = None
    ) -> float:
    
        """Calculate reward for programming novelty/freshness."""
        # 0) Early exit if no memory
        if self.memory is None or len(self.memory) == 0:
            return 0.75  # High novelty if no history

        # Ensure air_date is a Timestamp
        air_date = pd.to_datetime(air_date)

        # 1) Channel-specific repetition & time-decay score
        times_shown = int(movie.get('times_shown', 0))
        if times_shown == 0:
            channel_score = 1.0
        else:
            # Base repetition penalty
            rep_score = max(0.0, 1.0 - 0.25 * times_shown)
            # Linear time-decay over 365 days
            days_since = movie.get('date_last_diff', np.nan)
            if pd.isna(days_since):
                time_decay = 1.0
            else:
                days_since = air_date - movie['date_last_diff']
                days_since = min(days_since.days, 365)
                time_decay = days_since / 365.0
            channel_score = rep_score * time_decay

        # Collect scores
        novelty_scores = [channel_score]

        # 2) Cross-channel historical repetition
        if interest_historical_row is not None and available_historical_df is not None:
            last_showing = None
            # Look back over last 100 historical showings
            history_slice = available_historical_df.iloc[-100:]
            for idx, hist_row in history_slice.iterrows():
                if idx == movie.name:
                    hist_date = pd.to_datetime(hist_row['date'])
                    if last_showing is None or hist_date > last_showing:
                        last_showing = hist_date
            if last_showing is None:
                cross_score = 1.0
            else:
                days = (air_date - last_showing).days
                cross_score = min(1.0, days / 365.0)
            novelty_scores.append(cross_score)

        # Final novelty reward is the mean of all components
        novelty_reward = np.mean(novelty_scores)
        return float(np.clip(novelty_reward, 0.0, 1.0))

    def _calculate_novelty_reward2(self, air_date, movie, interest_historical_row: pd.Series = None, available_historical_df = None) -> float:
        """Calculate reward for programming novelty/freshness"""
        
        if self.memory is None or len(self.memory) == 0:
            return 0.75  # High novelty if no history
        
        novelty_scores = []
        
        if movie['times_shown'] == 0: # Times shown for interest channel
            repetition_score = 1.0  # Never shown before
        else:
            # Check if this exact movie was shown recently
            air_date = utils.date.to_datetime_format(air_date)

            repetition_score = 1.0 - 0.25 * movie['times_shown']

            days_since = movie.get('date_last_diff', np.nan)

            if pd.isna(days_since):
                time_decay = 1.0        # never shown → full novelty
            else:
                days_since = min(days_since, 365)
                time_decay = days_since / 365.0

            # 3) Combine linearly
            novelty_reward = repetition_score * time_decay

        if interest_historical_row is not None:
            # Movie repetition penalty
            last_showing = None
            
            for _, hist_row in available_historical_df.iloc[-100 : -1].iterrows(): # Check within last 100 movie showings across all channels
                if hist_row.name == movie.name:  # if movie has been shown
                    hist_date = pd.to_datetime(hist_row['date'])
                    if last_showing is None or hist_date > last_showing:
                        last_showing = hist_date

            if last_showing is None:
                repetition_score = 1.0  # Never shown before
            else:
                days_since = (air_date - last_showing).days
                repetition_score = min(1.0, days_since / 365.0)  
        
        novelty_scores.append(repetition_score)
        
        #TODO# 2. Director/Actor novelty
        """
        #recent_directors = set()
        #recent_actors = set()
        #
        #cutoff_date = air_date - pd.Timedelta(days=30)  # Last month
        #for hist_row in programming_history:
        #    if pd.to_datetime(hist_row['air_date']) >= cutoff_date:
        #        hist_movie = self.movie_catalog.loc[hist_row['movie_id']]
        #        recent_directors.add(hist_movie.get('director', ''))
        #        recent_actors.update(hist_movie.get('main_actors', '').split(','))
        #
        #director_novelty = 1.0 if movie.get('director', '') not in recent_directors else 0.3
        #actor_novelty = 1.0 if not any(actor.strip() in recent_actors 
        #                            for actor in movie.get('main_actors', '').split(',')) else 0.5
        #
        #novelty_scores.extend([director_novelty, actor_novelty])
        #
        ## 3. Thematic novelty (avoid similar themes too close together)
        #recent_themes = []
        #cutoff_date = air_date - pd.Timedelta(days=14)  # Last 2 weeks
        #
        #for hist_row in programming_history:
        #    if pd.to_datetime(hist_row['air_date']) >= cutoff_date:
        #        hist_movie = self.movie_catalog.loc[hist_row['movie_id']]
        #        recent_themes.extend(hist_movie.get('themes', '').split(','))
        #
        #movie_themes = movie.get('themes', '').split(',')
        #theme_overlap = len(set(movie_themes) & set(recent_themes)) / max(len(movie_themes), 1)
        #theme_novelty = 1.0 - theme_overlap
        #
        #novelty_scores.append(theme_novelty)
        #
        """

        novelty_reward = np.mean(novelty_scores)
        return np.clip(novelty_reward, 0.0, 1.0)

    def _calculate_rights_urgency_reward(self, air_date, movie, interest_historical_row: pd.Series = None) -> float:
        end_date = movie['end_rights']

        diff_days = (utils.date_formatting.to_datetime_format(end_date) - utils.date_formatting.to_datetime_format(air_date)).days
        

        # 0-30 days (critical urgency)
        if 0 <= diff_days <= 30: #Critical urgency
            reward = 1
        # 30-90 days (high urgency)
        elif 30 < diff_days <= 90:
            reward = 0.8  # smoother decay
        # 90-365 days (medium urgency)
        elif 90 < diff_days < 365//2:
            reward = 0.6
        # 6 months to a year
        elif 365//2 < diff_days < 365:
            reward = 0.3
        # year to 2 years
        elif 365 < diff_days < 2*365:
            reward = 0.1
        else:
            reward = 0  # No effect if too far before or after
        return reward




    # Helper functions for competitive analysis
    def _calculate_genre_advantage(self, our_movie, comp_movie) -> float:
        """Calculate advantage based on genre differentiation"""
        our_genres = set(our_movie.get('genre', '').split(','))
        comp_genres = set(comp_movie.get('genre', '').split(','))
        
        # Reward for different genres (counter-programming)
        overlap = len(our_genres & comp_genres) / max(len(our_genres | comp_genres), 1)
        return 1.0 - overlap  # Higher reward for less overlap

    def _calculate_quality_advantage(self, our_movie, comp_movie) -> float:
        """Calculate advantage based on relative quality"""
        our_rating = our_movie.get('imdb_rating', 5.0)
        comp_rating = comp_movie.get('imdb_rating', 5.0)
        
        # Normalize to 0-1 scale
        if comp_rating > 0:
            quality_ratio = our_rating / comp_rating
            return min(1.0, quality_ratio / 2.0)  # Cap at 1.0, advantage if 2x better
        return 0.5

    def _calculate_audience_advantage(self, our_movie, comp_movie, context_row) -> float:
        """Calculate advantage for target audience capture"""
        target_audience = context_row.get('target_audience', 'general')
        
        our_fit = self._get_audience_fit_score(our_movie, target_audience)
        comp_fit = self._get_audience_fit_score(comp_movie, target_audience)
        
        if comp_fit > 0:
            return our_fit / comp_fit
        return our_fit

    def _get_audience_fit_score(self, movie, target_audience) -> float:
        """Score how well a movie fits a target audience"""
        # This would be implemented based on your audience segmentation
        # Example implementation:
        
        audience_scores = {
            'family': movie.get('family_friendly_score', 0.5),
            'young_adult': movie.get('young_adult_appeal', 0.5),
            'mature': movie.get('mature_themes_score', 0.5),
            'general': 0.5  # Neutral for general audience
        }
        
        return audience_scores.get(target_audience, 0.5)

    def _calculate_genre_diversity(self,
        current_genres: List[str],
        recent_genres: List[List[str]],
        decay: float = 0.9) -> float:
        """
        Calculate genre diversity score based on overlap frequency and recency.

        Args:
            current_genres: List of genres of the current movie.
            recent_genres: List of genre lists of recently shown movies, ordered from most recent to oldest.
            decay: Exponential decay factor for recency weighting (e.g., 0.9).

        Returns:
            A float in [0, 1], where 1 = completely novel, 0 = heavily repetitive.
        """
        if not recent_genres or not current_genres: # If no recent movie genres shown or current movie genres shown default ot 0.5
            return 0.5

        current_genres_set = set(current_genres)
        genre_penalty = 0.0
        max_penalty = 0.0 # Normalize penalty by maximum possible penalty

        # Iterate over recent movies
        for i, genres in enumerate(recent_genres):
            recency_weight = decay ** i  # More recent = higher weight
            for genre in genres:
                if genre in current_genres_set:
                    genre_penalty += recency_weight
            max_penalty += recency_weight * len(current_genres_set)

        # Avoid division by zero
        if max_penalty == 0:
            return 1.0

        normalized_penalty = genre_penalty / max_penalty
        return 1.0 - normalized_penalty  # Higher = more diverse

    def _calculate_language_diversity(self,
        current_lang: str,
        recent_langs: List[str],
        decay: float = 0.9) -> float:
        """
        Calculate country diversity score with recency-based decay penalty.

        Args:
            current_lang: Language of the current movie.
            recent_langs: List of countries of recently shown movies (most recent first).
            decay: Recency decay factor (e.g., 0.9).

        Returns:
            A float in [0, 1], where 1 = highly diverse (never/long ago shown), 0 = heavily repetitive.
        """
        
        if not recent_langs:
            return 1.0

        
        penalty = 0.0
        max_penalty = 0.0

        for i, language in enumerate(recent_langs):
            
            weight = decay ** (i+1)
            if language == current_lang:
                penalty += weight
            max_penalty += weight

        diversity = 1.0 - (penalty / max_penalty if max_penalty > 0 else 0)
        return diversity

    def _calculate_era_diversity(self,
    current_era: int,
    recent_eras: List[int],
    decay: float = 0.9,
    distance_penalty_factor: float = 1.0
    ) -> float:
        """
        Calculates era diversity based on recency and distance of decade.

        Args:
            current_era: Current movie's decade (e.g., 1990).
            recent_eras: List of decades of recent movies (most recent first).
            decay: Recency weighting factor (e.g., 0.9).
            distance_penalty_factor: Multiplier for how much to penalize close decades.

        Returns:
            A float in [0, 1] where 1 = highly diverse, 0 = repetitive.
        """
        if not recent_eras:
            return 1.0

        penalty = 0.0
        max_penalty = 0.0

        for i, era in enumerate(recent_eras):
            recency_weight = decay ** i
            # Smaller distance = higher penalty
            distance = abs(current_era - era)
            distance_penalty = max(0, 1 - (distance_penalty_factor * distance))

            # Clamp between 0 and 1
            distance_penalty = min(1.0, max(0.0, distance_penalty))

            penalty += recency_weight * distance_penalty
            max_penalty += recency_weight  # full penalty if same era

        return 1.0 - (penalty / max_penalty if max_penalty > 0 else 0)

    def _calculate_audience_rating_diversity(self,
        current_rating: float,
        recent_ratings: List[float],
        decay: float = 0.9,
        distance_penalty_factor: float = 0.5
    ) -> float:
        """
        Compute a diversity score based on how similar the current audience rating is to recent ones.

        Args:
            current_rating: Current movie's audience rating (e.g., 7.5).
            recent_ratings: List of recent audience ratings (most recent first).
            decay: Recency weighting (0.9 = sharp decay, 0.99 = smoother).
            distance_penalty_factor: Scale factor for how harshly to penalize small rating differences.

        Returns:
            A float in [0, 1], where 1 = high diversity (novel performance), 0 = repetitive performance level.
        """
        if not recent_ratings:
            return 1.0

        penalty = 0.0
        max_penalty = 0.0

        for i, recent_rating in enumerate(recent_ratings):
            recency_weight = decay ** i
            distance = abs(current_rating - recent_rating)

            # Diversity is low if the distance is small
            distance_penalty = max(0.0, 1.0 - distance_penalty_factor * distance / MAX_AUDIENCE_RATING)  # 10 = scale of audience rating spread
            penalty += recency_weight * distance_penalty
            max_penalty += recency_weight  # worst-case: same rating

        return 1.0 - (penalty / max_penalty if max_penalty > 0 else 0)
