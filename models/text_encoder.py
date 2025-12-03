import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', embedding_dim=512):
        super(TextEncoder, self).__init__()
        
        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Get hidden size
        self.hidden_size = self.transformer.config.hidden_size
        
        # Projection head to embedding space
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Project to embedding space
        embeddings = self.projection(cls_output)
        
        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings