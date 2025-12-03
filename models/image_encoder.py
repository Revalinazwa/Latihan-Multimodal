import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, model_name='resnet50', embedding_dim=512, pretrained=True):
        super(ImageEncoder, self).__init__()
        
        # Load pretrained model
        if model_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'resnet101':
            base_model = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove final classification layer
        if 'resnet' in model_name:
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        elif 'efficientnet' in model_name:
            self.backbone = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection head to embedding space
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.model_name = model_name
    
    def forward(self, images):
        # Extract features
        features = self.backbone(images)
        
        # Pool if needed
        if 'efficientnet' in self.model_name:
            features = self.pool(features)
        
        # Flatten
        features = features.view(features.size(0), -1)
        
        # Project to embedding space
        embeddings = self.projection(features)
        
        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings