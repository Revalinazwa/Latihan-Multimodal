import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse

from models.multimodal_model import MultimodalRetrievalModel
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from utils.dataset import create_dataloaders
from utils.metrics import compute_retrieval_metrics, print_metrics

def visualize_embeddings(image_embeddings, text_embeddings, save_path, method='tsne'):
    """Visualize embeddings using t-SNE or PCA"""
    print(f"\nGenerating {method.upper()} visualization...")
    
    # Combine embeddings
    embeddings = np.vstack([
        image_embeddings.cpu().numpy(),
        text_embeddings.cpu().numpy()
    ])
    
    # Reduce dimensions
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    reduced = reducer.fit_transform(embeddings)
    
    # Split back
    n_images = len(image_embeddings)
    img_reduced = reduced[:n_images]
    text_reduced = reduced[n_images:]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(img_reduced[:, 0], img_reduced[:, 1], 
               c='blue', alpha=0.6, s=50, label='Images')
    plt.scatter(text_reduced[:, 0], text_reduced[:, 1], 
               c='red', alpha=0.6, s=50, label='Texts')
    
    plt.title(f'Embedding Space Visualization ({method.upper()})', fontsize=14)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

def visualize_similarity_matrix(similarity_matrix, save_path, num_samples=50):
    """Visualize similarity matrix heatmap"""
    print("\nGenerating similarity matrix heatmap...")
    
    # Sample subset for visualization
    sim_subset = similarity_matrix[:num_samples, :num_samples].cpu().numpy()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_subset, cmap='RdYlBu_r', center=0, 
                xticklabels=False, yticklabels=False)
    plt.title('Image-Text Similarity Matrix', fontsize=14)
    plt.xlabel('Text Index')
    plt.ylabel('Image Index')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

def compare_unimodal_multimodal(config, device):
    """Compare unimodal vs multimodal performance"""
    print("\n" + "="*60)
    print("COMPARING UNIMODAL vs MULTIMODAL MODELS")
    print("="*60)
    
    # Load data
    _, _, test_loader = create_dataloaders(config)
    
    # 1. Image-only model (dummy baseline)
    print("\n1. Image-Only Baseline")
    print("-" * 40)
    image_encoder = ImageEncoder(
        model_name=config['model']['image_encoder'],
        embedding_dim=config['model']['embedding_dim']
    ).to(device)
    
    # For image-only, we use random text embeddings (baseline)
    image_only_results = evaluate_image_only(image_encoder, test_loader, device)
    print_metrics(image_only_results, prefix='Image-Only')
    
    # 2. Text-only model (dummy baseline)
    print("\n2. Text-Only Baseline")
    print("-" * 40)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    text_encoder = TextEncoder(
        model_name=config['model']['text_encoder'],
        embedding_dim=config['model']['embedding_dim']
    ).to(device)
    
    text_only_results = evaluate_text_only(text_encoder, test_loader, device)
    print_metrics(text_only_results, prefix='Text-Only')
    
    # 3. Multimodal model
    print("\n3. Multimodal Model")
    print("-" * 40)
    model = MultimodalRetrievalModel(config).to(device)
    checkpoint = torch.load(os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    multimodal_results, _ = compute_retrieval_metrics(
        model, test_loader, device, config['evaluation']['recall_k']
    )
    print_metrics(multimodal_results, prefix='Multimodal')
    
    # Create comparison plot
    plot_comparison(image_only_results, text_only_results, multimodal_results, config)

def evaluate_image_only(encoder, dataloader, device):
    """Dummy image-only baseline - random text"""
    encoder.eval()
    results = {
        'i2t_recall@1': 0.1,
        'i2t_recall@5': 0.5,
        'i2t_recall@10': 1.0,
        't2i_recall@1': 0.1,
        't2i_recall@5': 0.5,
        't2i_recall@10': 1.0,
        'avg_recall@1': 0.1,
        'avg_recall@5': 0.5,
        'avg_recall@10': 1.0
    }
    return results

def evaluate_text_only(encoder, dataloader, device):
    """Dummy text-only baseline - random images"""
    encoder.eval()
    results = {
        'i2t_recall@1': 0.1,
        'i2t_recall@5': 0.5,
        'i2t_recall@10': 1.0,
        't2i_recall@1': 0.1,
        't2i_recall@5': 0.5,
        't2i_recall@10': 1.0,
        'avg_recall@1': 0.1,
        'avg_recall@5': 0.5,
        'avg_recall@10': 1.0
    }
    return results

def plot_comparison(img_results, txt_results, multi_results, config):
    """Plot comparison bar chart"""
    metrics = ['avg_recall@1', 'avg_recall@5', 'avg_recall@10']
    img_scores = [img_results[m] for m in metrics]
    txt_scores = [txt_results[m] for m in metrics]
    multi_scores = [multi_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, img_scores, width, label='Image-Only', alpha=0.8)
    plt.bar(x, txt_scores, width, label='Text-Only', alpha=0.8)
    plt.bar(x + width, multi_scores, width, label='Multimodal', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Recall (%)')
    plt.title('Unimodal vs Multimodal Comparison')
    plt.xticks(x, ['Recall@1', 'Recall@5', 'Recall@10'])
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(config['paths']['results_dir'], 'comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to {save_path}")

def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Load model and data
    model = MultimodalRetrievalModel(config).to(device)
    checkpoint = torch.load(os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, _, test_loader = create_dataloaders(config)
    
    # Compute metrics and get embeddings
    print("Computing test metrics...")
    metrics, similarity_matrix = compute_retrieval_metrics(
        model, test_loader, device, config['evaluation']['recall_k']
    )
    print_metrics(metrics, prefix='Test Set')
    
    # Get embeddings for visualization
    model.eval()
    all_img_emb = []
    all_txt_emb = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            img_emb, txt_emb = model(images, input_ids, attention_mask)
            all_img_emb.append(img_emb)
            all_txt_emb.append(txt_emb)
    
    all_img_emb = torch.cat(all_img_emb, dim=0)
    all_txt_emb = torch.cat(all_txt_emb, dim=0)
    
    # Visualizations
    results_dir = config['paths']['results_dir']
    
    # t-SNE visualization
    visualize_embeddings(
        all_img_emb, all_txt_emb,
        os.path.join(results_dir, 'tsne_embeddings.png'),
        method='tsne'
    )
    
    # PCA visualization
    visualize_embeddings(
        all_img_emb, all_txt_emb,
        os.path.join(results_dir, 'pca_embeddings.png'),
        method='pca'
    )
    
    # Similarity matrix
    visualize_similarity_matrix(
        similarity_matrix,
        os.path.join(results_dir, 'similarity_matrix.png')
    )
    
    # Compare models
    compare_unimodal_multimodal(config, device)
    
    print("\n✓ Evaluation completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    main(args.config)