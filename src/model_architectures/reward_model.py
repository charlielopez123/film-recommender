import torch
import torch.nn as nn
from typing import List

class RewardModel(nn.Module):
    """
    Reward Model - predicts expected reward (base utility) for movie-context pairs
    This is your Q-function approximation
    """
    
    def __init__(self, context_dim: int, movie_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        self.context_dim = context_dim
        self.movie_dim = movie_dim
        
        # Similar architecture to curator network but different objective
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0] // 2, hidden_dims[0] // 2)
        )
        
        self.movie_encoder = nn.Sequential(
            nn.Linear(movie_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0] // 2, hidden_dims[0] // 2)
        )
        
        # Reward prediction network
        combined_input_dim = hidden_dims[0]
        layers = []
        
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.extend([
                    nn.Linear(combined_input_dim, hidden_dims[i]),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
        
        # Output layer - expected reward [0,1]
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())  # Bound output to [0,1]
        
        self.reward_network = nn.Sequential(*layers)
    
    def forward(self, context_features, movie_features):
        context_encoded = self.context_encoder(context_features)
        movie_encoded = self.movie_encoder(movie_features)
        
        combined = torch.cat([context_encoded, movie_encoded], dim=1)
        
        # Predict expected reward
        reward = self.reward_network(combined)
        return reward.squeeze()