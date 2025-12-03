import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from models.multimodal_model import MultimodalRetrievalModel
from utils.dataset import create_dataloaders
from utils.losses import ContrastiveLoss
from utils.metrics import compute_retrieval_metrics, print_metrics

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        image_embeddings, text_embeddings = model(images, input_ids, attention_mask)
        
        # Compute loss
        loss = criterion(image_embeddings, text_embeddings)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            image_embeddings, text_embeddings = model(images, input_ids, attention_mask)
            
            # Compute loss
            loss = criterion(image_embeddings, text_embeddings)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main(config_path, resume_from=None):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(config['paths']['logs_dir'])
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = MultimodalRetrievalModel(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = ContrastiveLoss(temperature=config['model']['temperature'])
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_recall = 0
    
    if resume_from:
        if os.path.exists(resume_from):
            print(f"\n{'='*60}")
            print(f"Resuming from checkpoint: {resume_from}")
            print(f"{'='*60}")
            checkpoint = torch.load(resume_from, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_recall = checkpoint.get('best_recall', 0)
            
            print(f"✓ Resumed from epoch {checkpoint['epoch']}")
            print(f"✓ Best recall so far: {best_recall:.2f}%")
        else:
            print(f"⚠ Checkpoint not found: {resume_from}")
            print("Starting training from scratch...")
    
    # Training loop
    print("\nStarting training...")
    
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Compute retrieval metrics
        val_metrics, _ = compute_retrieval_metrics(
            model, val_loader, device, config['evaluation']['recall_k']
        )
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print_metrics(val_metrics, prefix='Validation')
        
        # Tensorboard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'Metrics/{key}', value, epoch)
        
        # Save best model
        avg_recall = val_metrics['avg_recall@5']
        if avg_recall > best_recall:
            best_recall = avg_recall
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_recall': best_recall,
                'metrics': val_metrics
            }, os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth'))
            print(f"\n✓ Saved best model (Recall@5: {avg_recall:.2f}%)")
        
        # Save latest checkpoint (for resume)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'best_recall': best_recall,
            'metrics': val_metrics
        }, os.path.join(config['paths']['checkpoint_dir'], 'latest_checkpoint.pth'))
        
        # Step scheduler
        scheduler.step()
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_similarity = compute_retrieval_metrics(
        model, test_loader, device, config['evaluation']['recall_k']
    )
    
    print_metrics(test_metrics, prefix='Test')
    
    # Save results
    import json
    with open(os.path.join(config['paths']['results_dir'], 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    writer.close()
    print("\nTraining completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., checkpoints/latest_checkpoint.pth)')
    args = parser.parse_args()
    
    main(args.config, resume_from=args.resume)