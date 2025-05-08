import torch
import torch.nn as nn
from transformers import GPT2Model

class CustomLLM(nn.Module):
    def __init__(self, base_model='gpt2', embedding_dim=768):
        """
        Custom LLM for prompt generation
        
        Args:
            base_model: Base LLM architecture
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.base_model = GPT2Model.from_pretrained(base_model)
        self.prompt_generator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, ambiguity_layer_output):
        """
        Generate prompt based on ambiguity layer output
        
        Args:
            ambiguity_layer_output: Intermediate layer from ambiguity network
        
        Returns:
            Generated prompt embedding
        """
        return self.prompt_generator(ambiguity_layer_output)