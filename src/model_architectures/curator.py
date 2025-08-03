import torch
import torch.nn as nn
from typing import List

class CuratorNetwork(nn.Module):
    """
    Behavioral Cloning Network - learns from expert curator decisions
    Predicts probability that curator would select this movie for this context
    """
    
    def __init__(self, context_dim: int, movie_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        self.context_dim = context_dim
        self.movie_dim = movie_dim
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0] // 2, hidden_dims[0] // 2)
        )
        
        # Movie encoder
        self.movie_encoder = nn.Sequential(
            nn.Linear(movie_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0] // 2, hidden_dims[0] // 2)
        )
        
        # Combined network
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
        
        # Output layer - probability of selection
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.combined_network = nn.Sequential(*layers)
    
    def forward(self, context_features, movie_features):
        context_encoded = self.context_encoder(context_features)
        movie_encoded = self.movie_encoder(movie_features)
        
        # Concatenate encoded features
        combined = torch.cat([context_encoded, movie_encoded], dim=1)
        
        # Get selection probability
        prob = self.combined_network(combined)
        return prob.squeeze()