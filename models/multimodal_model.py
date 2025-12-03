import torch
import torch.nn as nn
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder

class MultimodalRetrievalModel(nn.Module):
    def __init__(self, config):
        super(MultimodalRetrievalModel, self).__init__()
        
        # Initialize encoders
        self.image_encoder = ImageEncoder(
            model_name=config['model']['image_encoder'],
            embedding_dim=config['model']['embedding_dim']
        )
        
        self.text_encoder = TextEncoder(
            model_name=config['model']['text_encoder'],
            embedding_dim=config['model']['embedding_dim']
        )
        
        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(
            torch.ones([]) * config['model']['temperature']
        )
        
        # Freeze encoders if specified
        if config['model']['freeze_image_encoder']:
            for param in self.image_encoder.backbone.parameters():
                param.requires_grad = False
        
        if config['model']['freeze_text_encoder']:
            for param in self.text_encoder.transformer.parameters():
                param.requires_grad = False
    
    def forward(self, images, input_ids, attention_mask):
        # Encode images and text
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        
        return image_embeddings, text_embeddings
    
    def get_similarity(self, image_embeddings, text_embeddings):
        # Compute cosine similarity
        # image_embeddings: (batch_size, embedding_dim)
        # text_embeddings: (batch_size, embedding_dim)
        
        # Normalize
        image_embeddings = nn.functional.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = nn.functional.normalize(text_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(image_embeddings, text_embeddings.t())
        similarity = similarity / self.temperature
        
        return similarity