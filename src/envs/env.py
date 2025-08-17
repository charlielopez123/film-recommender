import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from competition.competitor import CompetitorDataManager
from reward_components.reward import RewardCalculator
import sys
from constants import *
from envs.context import *
import torch
import utils
from tqdm.notebook import tqdm
from constants import rf_model_column_names
import utils.date_formatting
from contextual_thompson import ContextualThompsonSampler, sigmoid
from model_architectures.curator import CuratorNetwork
import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but SimpleImputer was fitted with feature names"
)


class TVProgrammingEnvironment:
    """
    Contextual Multi-Armed Bandit Environment for TV Programming
    """
    
    def __init__(self, 
                 movie_catalog: pd.DataFrame,  # Your TMDB DataFrame
                 historical_df: Optional[pd.DataFrame] = None,
                 reward_weights: Optional[Dict[str, float]] = None,
                 audience_model: Optional[object] = None,
                 cts_path: str = "models/ts_state_tick_match.npz",
                 curator_model_path = 'models/curator_model.pth'):
        
        
        title_to_id = movie_catalog.set_index('title')['catalog_id'].to_dict()

        self.movie_catalog = movie_catalog.set_index('catalog_id')
        self.historical_df = historical_df#.set_index('catalog_id')
        self.audience_model = audience_model
        self.current_date = datetime.now()
        self.memory_size = 100  # Max memory size
        self.memory = []
        self.active_channel = 'RTS 1'
        self.available_movies = self.movie_catalog.copy().index.tolist()
        self.context_dim = 16
        self.movie_dim = 24

        # Competitor data manager
        self.competition_historical_df = self.historical_df[self.historical_df['channel'].isin(COMPETITOR_CHANNELS)]
        print("Setting up CompetitorDataManager...")
        self.competitor_manager = CompetitorDataManager(self.competition_historical_df, title_to_id_mapping=title_to_id)


        print("Setting up Scalers...")
        """
        self.scaler_dict = {
        "revenue": StandardScaler().fit(self.movie_catalog[['revenue']]), # Scaler expects on 2D array
        "vote_average": StandardScaler().fit(self.movie_catalog[['vote_average']]),
        "popularity": StandardScaler().fit(self.movie_catalog[['popularity']]),
        "duration": StandardScaler().fit(self.movie_catalog[['duration_min']]),
        "movie_age": RobustScaler().fit(self.movie_catalog[['movie_age']]),
        "rt_m": MinMaxScaler().fit(self.historical_df[['rt_m']]),
        }
        """

        self.scaler_dict = {
            "revenue": make_safe_positive_pipeline(log_compress=True).fit(
                self.movie_catalog[["revenue"]].replace([np.inf, -np.inf], np.nan)
            ),
            "popularity": make_safe_positive_pipeline(log_compress=True).fit(
                self.movie_catalog[["popularity"]].replace([np.inf, -np.inf], np.nan)
            ),
            "movie_age": make_safe_positive_pipeline(log_compress=True).fit(
                self.movie_catalog[["movie_age"]].replace([np.inf, -np.inf], np.nan)
            ),
            "duration": make_safe_positive_pipeline(log_compress=False).fit(
                self.movie_catalog[["duration_min"]].replace([np.inf, -np.inf], np.nan)
            ),
            "vote_average": make_safe_positive_pipeline(log_compress=False).fit(
                self.movie_catalog[["vote_average"]].replace([np.inf, -np.inf], np.nan)
            ),  # or MinMaxScaler if you prefer
            "rt_m": make_safe_positive_pipeline(log_compress=True).fit(
                self.historical_df[["rt_m"]].replace([np.inf, -np.inf], np.nan)
            ),
        }

        # Reward Calculator
        print("Setting up RewardCalculator...")
        self.reward = RewardCalculator(self.movie_catalog, 
                                       self.audience_model, 
                                       self.competitor_manager, 
                                       self.memory, 
                                       self.historical_df,
                                       self.scaler_dict)
            
        
        # Default reward weights
        self.reward_weights = reward_weights or {
            'audience': 0.4,
            'cultural': 0.25,
            'diversity': 0.2,
            'novelty': 0.15,
            'urgency': 0.1
        }
        
        # Track programming history
        self.programming_history: List[Tuple[Context, str, float]] = []
        self.curator_feedback_history: List[CuratorFeedback] = []
        
        # State tracking
        self.context_features_cache = {}

        # Setup Contextual Thompson Sampler
        print("Setting up ContextualThompsonSampler...")
        self.cts = ContextualThompsonSampler.load(Path(cts_path))

        # Setup Curator Network
        print("Setting up CuratorNetwork...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.curator_model = CuratorNetwork(context_dim = self.context_dim, movie_dim = self.movie_dim).to(self.device)
        self.curator_model.load_state_dict(torch.load(curator_model_path))

   
    def get_available_movies(self, date: datetime, context: Context):
        """Get movies available for the given context (rights not expired)"""
        # Assuming your DataFrame has 'rights_expiry' column as datetime
        #available_mask = self.movie_catalog['rights_expiry'] > self.current_date
        #return self.movie_catalog[available_mask].index.tolist()
        # TODO
        # Filter movies based on rights expiry
        # Can't show movies whom have passed the number of showings available
        available_mask = ((self.movie_catalog['end_rights'] > date) &
                            (self.movie_catalog['start_rights'] < date) &
                            (self.movie_catalog['available_num_diff'] > 0)
                            )
        self.available_movies = self.movie_catalog[available_mask].index.tolist()
    
    def get_context_features(self, context: Union[Context, Tuple]) -> np.ndarray:

        """
        Convert context to feature vector from either a given Context or previous context_cache_key

        feature_vector = [time_slot_hot (4,), day_of_week_hot (7,), is_weekend, season_one_hot(4,1)] -> shape: (16, 1)
        """
        # Cache key for efficiency
        if context is Tuple:
            cache_key = context
        else:
            cache_key = (context.hour, context.day_of_week, 
                        context.month, context.season.value)
        
        #if cache_key in self.context_features_cache:
        #    return self.context_features_cache[cache_key], cache_key
        
        features = []

        # Slot hour of showing into 4 different TimeSlot
        try:
            time_slot_value = self.get_time_slot(context.hour).value
        except:
            print(context.hour)
            sys.exit(1)
        time_slots = [time_slot.value for time_slot in TimeSlot]
        
        time_slot_features = [1 if time_slot == time_slot_value else 0 
                          for time_slot in time_slots]
        features.extend(time_slot_features)

        # Day-of-week one-hot (0=Monday ... 6=Sunday)
        dow_one_hot = [1 if context.day_of_week == i else 0 for i in range(7)]
        features.extend(dow_one_hot)
        
        # Weekend flag
        is_weekend = 1 if context.day_of_week >= 5 else 0
        features.append(is_weekend)
        
        # Season one-hot
        seasons = [season.value for season in Season]
        season_features = [1 if season == context.season.value else 0 
                          for season in seasons]
        features.extend(season_features)
        
        feature_vector = np.array(features, dtype=np.float32)
        self.context_features_cache[cache_key] = feature_vector
        return feature_vector, cache_key
    
    def get_context_features_cyclical(self, context: Union[Context, Tuple]) -> np.ndarray:
        """
        Convert context to feature vector from either a given Context or previous context_cache_key

        feature_vector = [hour_sin, hour_cos, day_of_week_sin, day_of_week_cos,
                   month_sin, month_cos, season_one_hot] -> shape: (10, 1)
        """
        # Cache key for efficiency
        if context is Tuple:
            cache_key = context
        else:
            cache_key = (context.hour, context.day_of_week, 
                        context.month, context.season.value)
        
        if cache_key in self.context_features_cache:
            return self.context_features_cache[cache_key], cache_key
        
        features = []
        
        # Hour cyclical encoding
        features.extend([
            np.sin(2 * np.pi * context.day_of_week / 27),
            np.cos(2 * np.pi * context.day_of_week / 27)
        ])
        
        # Day of week cyclical encoding
        features.extend([
            np.sin(2 * np.pi * context.day_of_week / 7),
            np.cos(2 * np.pi * context.day_of_week / 7)
        ])
        
        # Month cyclical encoding  
        features.extend([
            np.sin(2 * np.pi * context.month / 12),
            np.cos(2 * np.pi * context.month / 12)
        ])
        
        # Season one-hot
        seasons = [season.value for season in Season]
        season_features = [1 if season == context.season.value else 0 
                          for season in seasons]
        features.extend(season_features)
        
        feature_vector = np.array(features, dtype=np.float32)
        self.context_features_cache[cache_key] = feature_vector
        return feature_vector, cache_key


    def create_context_from_date(self, date: str, hour: int) -> Context:
        """Create Context object from historical data row"""
        
        air_date = utils.date_formatting.to_datetime_format(date)
        
        # Map month to season
        month = air_date.month
        if month in [3, 4, 5]:
            season = Season.SPRING
        elif month in [6, 7, 8]:
            season = Season.SUMMER
        elif month in [9, 10, 11]:
            season = Season.AUTUMN
        else:
            season = Season.WINTER
        
        return Context(
            hour=hour,
            day_of_week=air_date.weekday(),
            month=month,
            season=season,
            # TODO add more context features if available
            #special_event=row.get('special_event'),
            #target_audience=row.get('target_audience', 'general')
        )
    
    def get_movie_features(self, movie_id: str, return_features = False) -> np.ndarray:
        """Get movie features for Curator Network and Thompson Sampling
        movie_features = [norm_revenue, norm_vote_avg, norm_popularity,
                          norm_duration, norm_movie_age] -> shape: (N,)
        """
        try:
            movie = self.movie_catalog.loc[movie_id]
        except:
            print(f"Error finding movie_id {movie_id} for movie features")
            sys.exit(1)
       
        
        # Basic movie features
        try:
            features = pd.DataFrame({
                #TODO add further movie features
                #movie.cultural_value,
                #movie.times_shown / 10.0,  # normalized showing history
                # Scaler expects on 2D array
                'norm_revenue': self.scaler_dict['revenue'].transform([[movie.revenue]])[0][0],  # normalized revenue
                'norm_vote_avg':self.scaler_dict['vote_average'].transform([[movie.vote_average]])[0][0],  # normalized voting average
                'norm_popularity': self.scaler_dict['popularity'].transform([[movie.popularity]])[0][0],  # normalized popularity
                'norm_duration': self.scaler_dict['duration'].transform([[movie.duration_min]])[0][0],  # normalized runtime
                'norm_movie_age': self.scaler_dict['movie_age'].transform([[movie.movie_age]])[0][0],  # normalized age
                }, index=[0])
        except:
            print(f"movie id: {movie}")
            print(f"movie_revenue: {movie.revenue}")

        genre_prefix = 'genre_'
        

        # Identify expected dummy columns from model
        genre_dummies = [x for x in rf_model_column_names if x.startswith(genre_prefix)]
        
        

        # Add all dummy columns with 0 initially
        for col in genre_dummies:
            features[col] = 0

        # Set matching genre dummies to 1
        try:
            movie_genre_list = [genre['name'] for genre in movie['genres']]
        except:
            print('Error processing movie: ')
            print(movie)
            sys.exit(1)

        for genre_col in genre_dummies:
            if genre_col.removeprefix(genre_prefix) in movie_genre_list:
                features[genre_col] = 1

        #Set matching original language dummy to 1
        """
        lang_prefix = 'original_language_'
        lang_dummies = [x for x in rf_model_column_names if x.startswith(lang_prefix)]

        for col in lang_dummies:
            features[col] = 0

        if 'original_language' in movie:
            lang_code = movie['original_language']
            lang_col_name = f"{lang_prefix}{lang_code}"
            if lang_col_name in features.columns:
                features[lang_col_name] = 1
        """

        
        assert not features.isna().any().any(), f"NaN values found in movie features for {movie_id}, \n {features}"

        movie_features = np.squeeze(np.array(features, dtype=np.float32))
        if return_features:
            return movie_features, features
        else:
            return movie_features
    
    def get_candidate_features(self, context: Context, air_date: str):
        X_cands = []
        movies = []
        self.available_movies = self.get_available_movies(air_date, context)

        for movie in tqdm(self.available_movies):
            context_f, _ = self.get_context_features(context)
            movie_f = self.get_movie_features(movie)
            interaction_features, _ = self.build_context_movie_interactions(context_f, movie_f)

            _, rewards = self.reward.calculate_total_reward(movie, context, air_date)
            reward_features = np.array([reward for reward in rewards.values()])
            ensemble = np.concatenate((interaction_features, reward_features)).tolist()
            movies.append(movie)
            X_cands.append(ensemble)

            #print(context_f.shape, movie_f.shape, reward_features.shape)
        return movies, np.array(X_cands)
    
    def get_candidate_features_cts_cur(self, context: Context, air_date: str):
        X_cands = []
        movies = []
        #self.available_movies = self.get_available_movies(air_date, context)
        context_f, _ = self.get_context_features(context)
        context_tensor = torch.from_numpy(context_f).unsqueeze(0)
        

        for movie in tqdm(self.available_movies, leave=True, desc = "Movies"):
            movie_f = self.get_movie_features(movie)
            movie_tensor = torch.from_numpy(movie_f).unsqueeze(0)

            with torch.no_grad():  # no grad for inference
                selection_prob = torch.Tensor.numpy(self.curator_model(context_tensor, movie_tensor))
                
            _, rewards = self.reward.calculate_total_reward(movie, context, air_date)
            reward_features = np.array([reward for reward in rewards.values()])
            ensemble = np.concatenate((selection_prob.reshape(1,), reward_features)).tolist()
            movies.append(movie)
            X_cands.append(ensemble)

            #print(context_f.shape, movie_f.shape, reward_features.shape)
        return movies, np.array(X_cands)
    
    def get_candidate_features_cts_old(self, context: Context, air_date: str):
        X_cands = []
        movies = []
        self.available_movies = self.get_available_movies(air_date, context)

        for movie in tqdm(self.available_movies):
            movie_f = self.get_movie_features(movie)

            _, rewards = self.reward.calculate_total_reward(movie, context, air_date)
            reward_features = np.array([reward for reward in rewards.values()])
            ensemble = np.concatenate((movie_f, reward_features)).tolist()
            movies.append(movie)
            X_cands.append(ensemble)

            #print(context_f.shape, movie_f.shape, reward_features.shape)
        return movies, np.array(X_cands)

    def compute_reward(self, movie_id: str, context: Context, 
                      curator_feedback: Optional[CuratorFeedback] = None,
                      IL_learning: bool = False) -> float:
        """
        Compute the full reward for showing a movie in given context
        This is your ground truth reward function
        """
        movie = self.movie_catalog[movie_id]
        
        # Base utility components
        if IL_learning:
            # For imitation learning, use real audience ratings
            audience_reward = movie.predicted_audience
        else:
            audience_reward = movie.predicted_audience * self.reward_weights['audience']
        #TODO 
        # cultural_reward = movie.cultural_value * self.reward_weights['cultural']
        
        #TODO Diversity bonus
        #diversity_bonus = self._compute_diversity_bonus(movie_id, context)
        #diversity_reward = diversity_bonus * self.reward_weights['diversity']
        
        #TODO Novelty reward
        #novelty_score = self._compute_novelty_score(movie)
        #novelty_reward = novelty_score * self.reward_weights['novelty']
        
        #TODO Urgency multiplier
        #urgency_multiplier = self._compute_urgency_multiplier(movie)
        #urgency_reward = urgency_multiplier * self.reward_weights['urgency']
        
        #TODO base_reward = (audience_reward + cultural_reward + 
        #              diversity_reward + novelty_reward + urgency_reward)
        
        base_reward = audience_reward * self.reward_weights['audience']
        
        # Curator feedback adjustment
        feedback_adjustment = 0.0
        if curator_feedback:
            if curator_feedback.accepted:
                feedback_adjustment = 0.2 #* curator_feedback.curator_score
            else:
                feedback_adjustment = -0.3  # Penalty for rejection
        
        total_reward = base_reward + feedback_adjustment
        return np.clip(total_reward, 0.0, 1.0)
    
    def step(self, context: Context, movie_id: str, 
             curator_feedback: Optional[CuratorFeedback] = None) -> Tuple[float, Dict]:
        """
        Execute action and return reward
        """
        if movie_id not in self.movie_catalog:
            raise ValueError(f"Movie {movie_id} not in movie_catalog")
        
        # Compute reward
        reward = self.compute_reward(context, movie_id, curator_feedback)
        
        # Update movie history
        movie = self.movie_catalog[movie_id]
        movie.last_shown = self.current_date
        movie.times_shown += 1
        
        # Log the decision
        self.programming_history.append((context, movie_id, reward))
        
        if curator_feedback:
            self.curator_feedback_history.append(curator_feedback)
        
        # Return info for analysis
        info = {
            'movie_title': movie.title,
            'predicted_audience': movie.predicted_audience,
            #'cultural_value': movie.cultural_value,
            #'rights_urgency': self._get_rights_urgency(movie).value,
            #'competition_context': self._get_competition_context(movie_id).value
        }
        
        return reward, info
    
    def update_competitor_schedule(self, competitor: str, 
                                 schedule: List[Tuple[datetime, str]]):
        """Update competitor programming schedule"""
        self.competitor_schedule[competitor] = schedule
      
    def update_memory(self, movie_id: int):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append(movie_id)

    # Helper methods
    def _get_rights_urgency(self, movie: Movie) -> RightsUrgency:
        """Determine rights urgency category"""
        days_left = (movie.rights_expiry - self.current_date).days
        
        if days_left < 30:
            return RightsUrgency.CRITICAL
        elif days_left < 90:
            return RightsUrgency.HIGH
        elif days_left < 365:
            return RightsUrgency.MEDIUM
        else:
            return RightsUrgency.LOW
    
    def _get_competition_context(self, movie_id: int) -> CompetitionContext:
        """Determine competition context for a movie"""
        movie = self.movie_catalog.loc[movie_id]
        
        for competitor, schedule in self.competitor_schedule.items():
            for show_date, comp_movie_title in schedule:
                # Simple title matching (you'd want fuzzy matching in practice)
                if comp_movie_title.lower() in movie.title.lower():
                    days_diff = (show_date - self.current_date).days
                    
                    if 14 <= days_diff <= 30:
                        return CompetitionContext.PRE_COMPETITIVE
                    elif 7 <= days_diff <= 14:
                        return CompetitionContext.COMPETITIVE_WINDOW
                    elif -30 <= days_diff <= -7:
                        return CompetitionContext.POST_COMPETITIVE
        
        return CompetitionContext.CLEAR
    
    def _days_since_last_shown(self, movie: Movie) -> int:
        """Calculate days since movie was last shown"""
        if movie.last_shown is None:
            return 1000  # Never shown
        return (self.current_date - movie.last_shown).days
    
    def _compute_diversity_bonus(self, movie_id: str, context: Context) -> float:
        """Compute diversity bonus based on recent programming"""
        movie = self.movie_catalog[movie_id]
        
        # Look at last 10 shows in this time slot
        recent_shows = [
            (ctx, mid) for ctx, mid, _ in self.programming_history[-20:]
            if ctx.time_slot == context.time_slot
        ][-10:]
        
        if not recent_shows:
            return 0.5  # Neutral if no history
        
        # Check diversity across multiple dimensions
        recent_genres = [self.movie_catalog[mid].genre for _, mid in recent_shows]
        recent_countries = [self.movie_catalog[mid].country for _, mid in recent_shows]
        recent_languages = [self.movie_catalog[mid].language for _, mid in recent_shows]
        
        diversity_score = 0.0
        
        # Genre diversity
        if movie.genre not in recent_genres:
            diversity_score += 0.4
        elif recent_genres.count(movie.genre) <= 2:
            diversity_score += 0.2
        
        # Country diversity  
        if movie.country not in recent_countries:
            diversity_score += 0.3
        
        # Language diversity
        if movie.language not in recent_languages:
            diversity_score += 0.3
        
        return min(1.0, diversity_score)
    
    def _compute_novelty_score(self, movie: Movie) -> float:
        """Compute novelty score based on showing history"""
        days_since_shown = self._days_since_last_shown(movie)
        
        # Novelty decreases with frequency of showing
        frequency_penalty = movie.times_shown * 0.1
        
        # Novelty increases with time since last showing
        time_bonus = min(1.0, days_since_shown / 365.0)
        
        novelty = time_bonus - frequency_penalty
        return max(0.0, min(1.0, novelty))
    
    def _compute_urgency_multiplier(self, movie: Movie) -> float:
        """Compute urgency multiplier based on rights expiration"""
        urgency = self._get_rights_urgency(movie)
        
        multipliers = {
            RightsUrgency.CRITICAL: 1.0,
            RightsUrgency.HIGH: 0.8,
            RightsUrgency.MEDIUM: 0.5,
            RightsUrgency.LOW: 0.2
        }
        
        return multipliers[urgency]
    

    def get_time_slot(self, hour: int) -> Optional[TimeSlot]:

        if 6 <= hour < 14:
            return TimeSlot.MORNING
        elif 14 <= hour < 19:
            return TimeSlot.AFTERNOON
        elif 19 <= hour < 22:
            return TimeSlot.PRIME_TIME
        elif 22 <= hour < 26:
            return TimeSlot.LATE_NIGHT
        else:
            return None  # outside defined slots
        

    def build_context_movie_interactions(
        self,
        context_feat: np.ndarray,  # shape (C,)
        movie_feat: np.ndarray,    # shape (M,)
        context_names: List[str] = context_feature_names,  # len C
        movie_names: List[str]= movie_feature_names,    # len M
        center: bool = False,
        normalize: bool = False,
        selected_pairs: Optional[List[Tuple[str, str]]] = None  # if provided, only these crosses
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Returns (interaction_vector, interaction_names), where interaction_vector corresponds to
        elementwise products between context and movie features (optionally centered/normalized),
        and names are like "ctxName__x__movieName".
        If selected_pairs is given, only those specific (context_name, movie_name) products are included.
        """
        assert len(context_feat) == len(context_names), f'{len(context_feat)}, {len(context_names)}'

        assert len(movie_feat) == len(movie_names)

        c = context_feat.astype(float).copy()
        m = movie_feat.astype(float).copy()

        if center:
            c_mean = c.mean() if c.size > 0 else 0.0
            m_mean = m.mean() if m.size > 0 else 0.0
            c -= c_mean
            m -= m_mean

        if normalize:
            c_std = c.std(ddof=0) if c.size > 0 else 1.0
            m_std = m.std(ddof=0) if m.size > 0 else 1.0
            if c_std > 0:
                c /= c_std
            if m_std > 0:
                m /= m_std

        interaction_vals = []
        interaction_names = []

        if selected_pairs is None:
            for i, cname in enumerate(context_names):
                for j, mname in enumerate(movie_names):
                    interaction_vals.append(c[i] * m[j])
                    interaction_names.append(f"{cname}__x__{mname}")
        else:
            # map names to indices for fast lookup
            ctx_idx = {name: idx for idx, name in enumerate(context_names)}
            mov_idx = {name: idx for idx, name in enumerate(movie_names)}
            for cname, mname in selected_pairs:
                if cname not in ctx_idx or mname not in mov_idx:
                    continue  # skip invalid
                i = ctx_idx[cname]
                j = mov_idx[mname]
                interaction_vals.append(c[i] * m[j])
                interaction_names.append(f"{cname}__x__{mname}")

        interaction_vector = np.array(interaction_vals, dtype=np.float32)
        return interaction_vector, interaction_names

    ## Contextual Thompson Sampling functions
    def recommend_n_films(self, context: Context, air_date: datetime):
        # movies: list of movie IDs, X_cands: Mxd numpy array
        # Thompson‐Sampling selects one
        print('Getting candidate features...')
        context_f, _ = self.get_context_features(context)
        movies, X_cands = self.get_candidate_features_cts_cur(context, air_date)
        print('Done')
        top5_idx, top5_scores, w_tilde, _ = self.cts.score_candidates(context_f, X_cands, K=5)
        recommended = [movies[i] for i in top5_idx]

        for movie, score in zip(recommended, top5_scores):
            print(f"{movie}: score = {score:.3f}")

        return recommended, top5_idx, top5_scores, w_tilde, movies, X_cands
    
    def show_top_breakdown(self, top5_idx, top5_scores, w_tilde, movies, X_cands):
        eps = 1e-9  # threshold to consider "non-zero"
        for i, idx in enumerate(top5_idx):
            movie_id = movies[idx]
            x = X_cands[idx]
            total = top5_scores[i]
            print(f"\n🎬 {movie_id}: {self.movie_catalog.loc[movie_id]['title']}  (total score = {total:.3f}) (p = {sigmoid(total)})")
            print(f"\n Actors: {self.movie_catalog.loc[movie_id]['actors']}")
            print("  Breakdown:")

            # build list of (name, xi, wi, contribution)
            contribs = [
                (name, float(xi), float(wi), float(xi * wi))
                for name, xi, wi in zip(all_reward_feature_names, x, w_tilde)
            ]
            # filter out negligible contributions
            nonzero = [t for t in contribs if abs(t[3]) > eps]
            # sort by absolute impact (change to key=lambda t: -t[3] if you want signed descending)
            nonzero.sort(key=lambda t: t[3], reverse=True)

            for name, xi, wi, contrib in nonzero:
                sign = "+" if contrib >= 0 else "-"
                print(f"    • {name:30s} {xi:6.3f} × {wi:6.3f} = {contrib:7.3f} ({sign})")

    def get_context_features_from_date_hour(self, date: str, hour: int):
        air_date = utils.date_formatting.to_datetime_format(date)
        context = self.create_context_from_date(date, hour)
        context_f, _ = self.get_context_features(context)
        return context_f, context, air_date
    
    def encode_chosen_signals(self, chosen_signals_str: str, all_signals: List = y_signal_feature_selection):
        """
        all_signals: list of feature names, e.g. ['cur','aud','comp','div','nov','rights']
        chosen_signals_str: space-separated string, e.g. "aud cur rights"
        Returns: np.ndarray of 0/1 of length len(all_signals)
        """
        chosen_set = set(chosen_signals_str.split())
        return np.array([1 if sig in chosen_set else 0 for sig in all_signals], dtype=int)


        
def safe_log1p(X):
    X = np.asarray(X, dtype=float)
    # Replace infinities with NaN so imputer can handle them upstream if needed
    X[~np.isfinite(X)] = np.nan
    # Clip negatives to zero (since data should be naturally positive)
    X = np.clip(X, 0, None)
    # log1p is now safe
    return np.log1p(X)

def make_safe_positive_pipeline(log_compress: bool = True):
    steps = []
    # 1. Impute missing / replaced inf values
    steps.append(("impute", SimpleImputer(strategy="median")))
    # 2. Optional safe log compress for heavy-tailed positives
    if log_compress:
        steps.append(("safe_log1p", FunctionTransformer(safe_log1p, validate=False)))
    # 3. Rank-based mapping to uniform [0,1], robust to outliers
    steps.append(("quantile", QuantileTransformer(output_distribution="uniform", random_state=0, copy=True)))
    return make_pipeline(*[step for _, step in steps])
