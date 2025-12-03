import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from transformers import AutoTokenizer
from torchvision import transforms

class Flickr8kDataset(Dataset):
    def __init__(self, data_df, image_dir, tokenizer, max_length=77, transform=None):
        """
        Args:
            data_df: DataFrame with columns ['image', 'caption']
            image_dir: Directory containing images
            tokenizer: Pretrained tokenizer
            max_length: Max sequence length for text
            transform: Image transformations
        """
        self.data = data_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, row['image'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Tokenize caption
        caption = str(row['caption'])
        tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'caption': caption,
            'image_name': row['image']
        }

def get_transforms(mode='train'):
    """Get image transformations"""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

def create_dataloaders(config):
    """Create train, val, test dataloaders"""
    # Load captions
    captions_df = pd.read_csv(config['dataset']['captions_file'])
    
    # Split dataset
    from sklearn.model_selection import train_test_split
    
    # Get unique images
    unique_images = captions_df['image'].unique()
    
    train_imgs, temp_imgs = train_test_split(
        unique_images, 
        test_size=(config['dataset']['val_split'] + config['dataset']['test_split']),
        random_state=42
    )
    
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=config['dataset']['test_split']/(config['dataset']['val_split'] + config['dataset']['test_split']),
        random_state=42
    )
    
    # Create DataFrames
    train_df = captions_df[captions_df['image'].isin(train_imgs)]
    val_df = captions_df[captions_df['image'].isin(val_imgs)]
    test_df = captions_df[captions_df['image'].isin(test_imgs)]
    
    # Sampling if needed
    max_samples = config['dataset'].get('max_samples', None)
    if max_samples:
        train_df = train_df.sample(min(max_samples, len(train_df)), random_state=42)
        val_df = val_df.sample(min(max_samples//5, len(val_df)), random_state=42)
        test_df = test_df.sample(min(max_samples//5, len(test_df)), random_state=42)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    
    # Create datasets
    train_dataset = Flickr8kDataset(
        train_df, 
        config['dataset']['image_dir'],
        tokenizer,
        transform=get_transforms('train')
    )
    
    val_dataset = Flickr8kDataset(
        val_df,
        config['dataset']['image_dir'],
        tokenizer,
        transform=get_transforms('val')
    )
    
    test_dataset = Flickr8kDataset(
        test_df,
        config['dataset']['image_dir'],
        tokenizer,
        transform=get_transforms('test')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    return train_loader, val_loader, test_loader