import pandas as pd
import numpy as np
from envs.env import Context, TimeSlot, Season, TVProgrammingEnvironment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import Dict, List, Tuple, Optional
from model_architectures.curator import CuratorNetwork
from model_architectures.reward_model import RewardModel
from tqdm import tqdm
from api import tmdb
import utils.preprocessing
import utils.date_formatting
from constants import BASE_CUSTOM_ID
import sys

import warnings
warnings.filterwarnings('ignore')

class HistoricalProgrammingDataset(Dataset):
    """Dataset for historical programming decisions"""
    
    def __init__(self, 
                 context_features: np.ndarray,
                 movie_features: np.ndarray,
                 targets: np.ndarray,
                 target_type: str = 'binary'):  # 'binary' for curator, 'continuous' for reward
        
        self.context_features = torch.FloatTensor(context_features)
        self.movie_features = torch.FloatTensor(movie_features)
        self.targets = torch.FloatTensor(targets)
        self.target_type = target_type
        
        if target_type == 'binary':
            self.targets = self.targets.long()
    
    def __len__(self):
        return len(self.context_features)
    
    def __getitem__(self, idx):
        return {
            'context': self.context_features[idx],
            'movie': self.movie_features[idx],
            'target': self.targets[idx]
        }

class HistoricalDataProcessor:
    """
    Processes historical programming data for training
    """
    
    def __init__(self, 
                 environment: TVProgrammingEnvironment):
        self.env = environment
        self.historical_df = self.env.historical_df
        self.gamma = 0.6
        
        #competition_historical_data = historical_df[historical_df['channel'].str.contains('competitor', case=False)]


    def prepare_training_data(self, 
                            channel_name: str = "RTS 1",
                            negative_sampling_ratio: float = 0,
                            time_split_date: str = None) -> Dict:
        """
        Prepare training data from historical programming decisions
        
        Args:
            channel_name: Name of your channel in the historical data
            negative_sampling_ratio: Ratio of negative samples to positive samples
            time_split_date: Split date for temporal validation (format: 'YYYY-MM-DD')
        
        Returns:
            Dictionary with training data for both networks
        """
        
        print("Processing historical programming data...")
        
        # Filter for your channel's programming decisions
        interest_channel_historical_df = self.historical_df[
            self.historical_df['channel'] == channel_name
        ].copy()
        
        print(f"Found {len(interest_channel_historical_df)} programming decisions for {channel_name}")
        
        # Prepare positive samples (movies that were actually shown)
        all_samples = []
        all_rewards = [] # debug
        num_successful_samples = 0
        
        # reset memory
        self.env.memory = []
        self.env.reward.memory = self.env.memory

        movies_not_found = []

        # Iterate over the interestchannel's history rows
        for _, row in tqdm(interest_channel_historical_df.iterrows(), total=len(interest_channel_historical_df), desc="Processing rows"): # Iterate through each row of the interest channel's historical decisions
                
                
                # Create context from historical data
                context = self._create_context_from_row(row)

                air_date = pd.to_datetime(row['date'])

                # Set historical data based off what is available for current date
                copy = self.historical_df.copy()
                available_historical_data = copy[utils.date_formatting.to_datetime_format(copy['date']) <= air_date]

                available_movies = self.env.get_available_movies(air_date, context)
                self.env.available_movies = available_movies # Update environment's available movies

                # Get movie features
                movie_id = row['tmdb_id']
                #print(f'row name:{movie_id}')
                
                
                
                # Check if movie exists in catalog
                try:
                    _ = self.env.movie_catalog.loc[movie_id]
                except KeyError: # If movie not found in catalog add it
                    movies_not_found.append(row.name)
                    self._add_missing_movie_to_catalog(movie_id, row)
                    
                
                self.env.update_memory(movie_id) # Store movie ID in memory
                # Set movie times shown based on historical data if available
                if pd.isna(self.env.movie_catalog.loc[movie_id, 'date_diff_1']):
                    self.env.movie_catalog.loc[movie_id, 'times_shown'] = 0
                else:
                    show_cols = ['date_diff_1', 'date_rediff_1', 'date_rediff_2', 'date_rediff_3', 'date_rediff_4']
                    for c in show_cols:
                        self.env.movie_catalog.loc[movie_id, c] = pd.to_datetime(self.env.movie_catalog.loc[movie_id, c] , errors="coerce")
                    mask = self.env.movie_catalog.loc[movie_id, show_cols].lt(air_date)
                    self.env.movie_catalog.loc[movie_id, "times_shown"] = mask.sum(axis=1)
                
                if movie_id not in self.env.movie_catalog.index:
                    print(f"Movie ID {movie_id} not found in catalog, skipping row")
                    print(row)
                    continue
                
                context_features, context_cache_key = self.env.get_context_features(context)

                movie_features = self.env.get_movie_features(movie_id)
                

                # Calculate actual reward from historical data
                actual_reward, reward_dict = self.env.reward.calculate_total_reward(movie_id, context, air_date, row, available_historical_data)
                if actual_reward < 0.05:
                    print(actual_reward)
                    print(reward_dict)
                    print(context_cache_key)            
                # debug reward_dict
                
                pos_sample = {
                    'context_features': context_features,
                    'movie_features': movie_features,
                    'movie_id': movie_id,
                    'selected': 1,  # Positive sample
                    'reward': (self.gamma * 1 + (1-self.gamma) * actual_reward), # weight the weighted reward factors with the actual selection of the curator
                    'date': air_date,
                    'context_cache_key': context_cache_key,
                    'current_memory': self.env.memory
                }
                all_samples.append(pos_sample)
                all_rewards.append(reward_dict) # debug
                num_successful_samples += 1

                # Update movie_feature for number of times shown
                self.env.movie_catalog.loc[movie_id, 'times_shown'] += 1
                
                

                # Generate negative samples (movies that could have been shown but weren't)
                neg_samples, neg_rewards = self.generate_negative_samples(pos_sample, row, air_date, context, context_cache_key, negative_sampling_ratio, available_historical_data)
                all_samples.extend(neg_samples)
                all_rewards.extend(neg_rewards) # debug

            #except Exception as e:
            #    print(f"Error processing row {row['movie_id']}: {e}")
            #    sys.exit(1)
        
        print(f"Created {num_successful_samples} positive samples")

        # Convert to arrays
        context_features = np.array([s['context_features'] for s in all_samples])
        movie_features = np.array([s['movie_features'] for s in all_samples])
        curator_targets = np.array([s['selected'] for s in all_samples])
        reward_targets =  np.array([s['reward']   for s in all_samples])
        dates = np.array([s['date'] for s in all_samples])
        context_cache_keys = np.array([s['context_cache_key'] for s in all_samples])
        current_memories = np.array([s['current_memory'] for s in all_samples])
        
        # Temporal split if specified
        if time_split_date:
            split_date = pd.to_datetime(time_split_date)
            train_mask = dates < split_date
            
            train_data = {
                'context_features': context_features[train_mask],
                'movie_features': movie_features[train_mask],
                'curator_targets': curator_targets[train_mask],
                'reward_targets': reward_targets[train_mask],
                'context_cache_keys': context_cache_keys[train_mask],
                'current_memories': current_memories[train_mask]
            }
            
            val_data = {
                'context_features': context_features[~train_mask],
                'movie_features': movie_features[~train_mask],
                'curator_targets': curator_targets[~train_mask],
                'reward_targets': reward_targets[~train_mask],
                'context_cache_keys': context_cache_keys[~train_mask],
                'current_memories': current_memories[~train_mask]
            }
            
            return {'train': train_data, 'val': val_data}
        
        else:
            return {
                'context_features': context_features,
                'movie_features': movie_features,
                'curator_targets': curator_targets,
                'reward_targets': reward_targets, # Custom reward function for IL training
                'context_cache_keys': context_cache_keys,
                'current_memories': current_memories,
            }, all_rewards, movies_not_found
    
    def _create_context_from_row(self, row) -> 'Context':
        """Create Context object from historical data row"""

        air_date = pd.to_datetime(row['date'])
        
        # Map time to time slot (you'll need to adjust based on your data)
        hour = row.get('hour')
        
        # Map month to season
        month = row.get('month', air_date.month)
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
            day_of_week=row.get('weekday'),
            month=month,
            season=season,
            # TODO add more context features if available
            #special_event=row.get('special_event'),
            #target_audience=row.get('target_audience', 'general')
        )
    
    def generate_negative_samples(self, positive_sample: Dict, row, air_date, context: Context, context_cache_key: Tuple, ratio: float, available_historical_data = None) -> List[Dict]:
        """
        row: historical_df row
        """
        negative_samples = []
            
        # Remove the actual movie that was shown
        if positive_sample['movie_id'] in self.env.available_movies:
            self.env.available_movies.remove(positive_sample['movie_id'])

        n_negatives = int(ratio)
        if len(self.env.available_movies) < n_negatives:
            n_negatives = len(self.available_movies)

        # Sample negative movies
        negative_movie_ids = np.random.choice(
            self.env.available_movies, size=n_negatives, replace=False
        )
        #context_features, _ = self.env.get_context_features(context_cache_key)
        context_features = positive_sample['context_features']
        negative_rewards = [] # debug
        for neg_movie_id in negative_movie_ids:
            movie_features = self.env.get_movie_features(neg_movie_id)
            estimated_reward, reward_dict = self.env.reward.calculate_total_reward(neg_movie_id, context, air_date, row, available_historical_data)
            if estimated_reward < 0.05:
                print(estimated_reward)
                print(reward_dict)
                print(context_cache_key)
            negative_samples.append({
                        'context_features': context_features,
                        'movie_features': movie_features,
                        'movie_id': neg_movie_id,
                        'selected': 0,  # Negative sample
                        'reward': (1- self.gamma) * estimated_reward,
                        'date': positive_sample['date'],
                        'context_cache_key': context_cache_key,
                        'current_memory': self.env.memory
                    })
            negative_rewards.append(reward_dict) # debug
        return negative_samples, negative_rewards # debug negative_rewards

    def _add_missing_movie_to_catalog(self, movie_id, historical_row):
        """
        Adds a missing movie to the catalog if not already present.

        Parameters:
        - catalog_df: pd.DataFrame, the existing catalog
        - historical_row: pd.Series or dict, representing a row from historical data
        - tmdb_data: dict or None, additional data from TMDB API (if available)

        Returns:
        - updated catalog_df with the new row added if needed
        """

        partial_new_entry = {}
        # Check if historical row already contains tmdb data
        if historical_row['missing_tmdb_id'] is False:
            partial_new_entry['catalog_id'] =  historical_row.get('tmdb_id')
            movie_features = tmdb.get_movie_features(movie_id)
            partial_new_entry |= movie_features
        else:
            missing_mask = self.env.movie_catalog['missing_tmdb'] == True
            partial_new_entry['catalog_id'] = BASE_CUSTOM_ID + str(missing_mask.sum()) # Write corresponding missing tmdb id type catalog id

        partial_new_entry |= historical_row.copy().to_dict()
        partial_new_entry['last_diff_rating'] =  historical_row.get('rt_m')
        partial_new_entry.pop('rt_m')

        
        partial_new_entry |= {
            'actors': '',
            'genres': [],
            'times_shown': 0,
        }

        row = pd.Series(partial_new_entry, name=movie_id).reindex(self.env.movie_catalog.columns) # create catalog row from partial entry

        row= pd.DataFrame(row).T
        row = utils.preprocessing.preprocess_featured_movies(row)
        self.env.movie_catalog.loc[movie_id] = row.squeeze()




class NetworkTrainer:
    """
    Handles training of both Curator Network and Reward Model
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def train_curator_network(self, 
                            training_data: Dict,
                            validation_data: Optional[Dict] = None,
                            epochs: int = 100,
                            batch_size: int = 256,
                            learning_rate: float = 0.001) -> CuratorNetwork:
        """Train the Curator Network using behavioral cloning"""
        
        print("Training Curator Network...")
        
        # Create datasets
        train_dataset = HistoricalProgrammingDataset(
            training_data['context_features'],
            training_data['movie_features'],
            training_data['curator_targets'],
            target_type='binary'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        context_dim = training_data['context_features'].shape[1]
        movie_dim = training_data['movie_features'].shape[1]
        
        model = CuratorNetwork(context_dim, movie_dim).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                context = batch['context'].to(self.device)
                movie = batch['movie'].to(self.device)
                targets = batch['target'].float().to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(context, movie)
                try:
                    loss = criterion(predictions, targets)
                except:
                    print("Predictions range:", predictions.min().item(), predictions.max().item())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation
            if validation_data is not None:
                val_acc = self._evaluate_curator(model, validation_data, batch_size)
                val_accuracies.append(val_acc)
                scheduler.step(1 - val_acc)  # Maximize accuracy
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
            else:
                scheduler.step(avg_loss)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={avg_loss:.4f}")
        
        print("Curator Network training completed!")
        return model
    
    def train_reward_model(self,
                          training_data: Dict,
                          validation_data: Optional[Dict] = None,
                          epochs: int = 100,
                          batch_size: int = 256,
                          learning_rate: float = 0.001) -> RewardModel:
        """Train the Reward Model"""
        
        print("Training Reward Model...")
        
        # Create datasets
        train_dataset = HistoricalProgrammingDataset(
            training_data['context_features'],
            training_data['movie_features'],
            training_data['reward_targets'],
            target_type='continuous'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        context_dim = training_data['context_features'].shape[1]
        movie_dim = training_data['movie_features'].shape[1]
        
        model = RewardModel(context_dim, movie_dim).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                context = batch['context'].to(self.device)
                movie = batch['movie'].to(self.device)
                targets = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(context, movie)
                loss = criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation
            if validation_data is not None:
                val_loss = self._evaluate_reward_model(model, validation_data, batch_size)
                val_losses.append(val_loss)
                scheduler.step(val_loss)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}")
            else:
                scheduler.step(avg_loss)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={avg_loss:.4f}")
        
        print("Reward Model training completed!")
        return model
    
    def _evaluate_curator(self, model, validation_data, batch_size):
        """Evaluate curator network accuracy"""
        model.eval()
        
        val_dataset = HistoricalProgrammingDataset(
            validation_data['context_features'],
            validation_data['movie_features'],
            validation_data['curator_targets'],
            target_type='binary'
        )
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                context = batch['context'].to(self.device)
                movie = batch['movie'].to(self.device)
                targets = batch['target']
                
                predictions = model(context, movie)
                predictions = (predictions > 0.5).cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets.numpy())
        
        accuracy = accuracy_score(all_targets, all_predictions)
        return accuracy
    
    def _evaluate_reward_model(self, model, validation_data, batch_size):
        """Evaluate reward model MSE"""
        model.eval()
        
        val_dataset = HistoricalProgrammingDataset(
            validation_data['context_features'],
            validation_data['movie_features'],
            validation_data['reward_targets'],
            target_type='continuous'
        )
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                context = batch['context'].to(self.device)
                movie = batch['movie'].to(self.device)
                targets = batch['target']
                
                predictions = model(context, movie)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        mse = mean_squared_error(all_targets, all_predictions)
        return mse

# Example usage function
def train_models_from_historical_data(historical_df: pd.DataFrame,
                                    movie_catalog: pd.DataFrame,
                                    environment) -> Tuple[CuratorNetwork, RewardModel]:
    """
    Complete training pipeline for both models
    """
    
    # Process historical data
    processor = HistoricalDataProcessor(historical_df, movie_catalog, environment)
    
    # Prepare training data with temporal split
    data = processor.prepare_training_data(
        channel_name="your_channel_name",
        negative_sampling_ratio=3.0,
        time_split_date="2024-01-01"  # Use last few months for validation
    )
    
    # Initialize trainer
    trainer = NetworkTrainer()
    
    # Train Curator Network
    curator_model = trainer.train_curator_network(
        training_data=data['train'],
        validation_data=data['val'],
        epochs=100,
        batch_size=256,
        learning_rate=0.001
    )
    
    # Train Reward Model
    reward_model = trainer.train_reward_model(
        training_data=data['train'],
        validation_data=data['val'],
        epochs=100,
        batch_size=256,
        learning_rate=0.001
    )
    
    return curator_model, reward_model

