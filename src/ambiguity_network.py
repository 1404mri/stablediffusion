import torch
import torch.nn as nn

class AmbiguityPredictor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        """
        Network to predict token ambiguity
        
        Args:
            input_dim: Dimension of text encoder output
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Ambiguity score prediction
            nn.Sigmoid()  # Normalized ambiguity score
        )
    
    def forward(self, text_encoding, time_embedding):
        """
        Forward pass to predict ambiguity
        
        Args:
            text_encoding: Text encoder output
            time_embedding: Time embedding
        
        Returns:
            Predicted ambiguity score
        """
        x = torch.cat([text_encoding, time_embedding], dim=-1)
        return self.network(x)