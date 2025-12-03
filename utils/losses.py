import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss untuk image-text retrieval (mirip CLIP)
    """
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, image_embeddings, text_embeddings):
        """
        Args:
            image_embeddings: (batch_size, embedding_dim)
            text_embeddings: (batch_size, embedding_dim)
        """
        batch_size = image_embeddings.shape[0]
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature
        
        # Labels are diagonal (each image matches its own text)
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Image-to-text loss
        loss_i2t = F.cross_entropy(logits, labels)
        
        # Text-to-image loss
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        # Total loss
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss

class TripletLoss(nn.Module):
    """
    Alternative: Triplet loss
    """
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: anchor embeddings
            positive: positive embeddings
            negative: negative embeddings
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()