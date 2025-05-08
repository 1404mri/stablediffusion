import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer

class AmbiguityDataset(Dataset):
    def __init__(self, text_encodings, time_embeddings, ambiguity_scores):
        """
        Custom dataset for text ambiguity prediction
        
        Args:
            text_encodings: Stable Diffusion text encoder outputs
            time_embeddings: Time embeddings
            ambiguity_scores: Ground truth ambiguity scores for tokens
        """
        self.text_encodings = text_encodings
        self.time_embeddings = time_embeddings
        self.ambiguity_scores = ambiguity_scores
    
    def __len__(self):
        return len(self.text_encodings)
    
    def __getitem__(self, idx):
        return {
            'text_encoding': self.text_encodings[idx],
            'time_embedding': self.time_embeddings[idx],
            'ambiguity_score': self.ambiguity_scores[idx]
        }