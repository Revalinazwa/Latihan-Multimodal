import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10]):
    """
    Compute Recall@K for image-to-text and text-to-image retrieval
    
    Args:
        similarity_matrix: (num_images, num_texts) similarity scores
        k_values: list of K values
    
    Returns:
        dict with recall scores
    """
    num_samples = similarity_matrix.shape[0]
    
    # Image-to-Text Retrieval
    i2t_ranks = []
    for i in range(num_samples):
        # Sort by similarity (descending)
        sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
        # Find rank of correct text (index i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        i2t_ranks.append(rank)
    
    # Text-to-Image Retrieval
    t2i_ranks = []
    for i in range(num_samples):
        sorted_indices = torch.argsort(similarity_matrix[:, i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        t2i_ranks.append(rank)
    
    # Compute Recall@K
    results = {}
    for k in k_values:
        i2t_recall = 100.0 * np.sum(np.array(i2t_ranks) < k) / num_samples
        t2i_recall = 100.0 * np.sum(np.array(t2i_ranks) < k) / num_samples
        
        results[f'i2t_recall@{k}'] = i2t_recall
        results[f't2i_recall@{k}'] = t2i_recall
        results[f'avg_recall@{k}'] = (i2t_recall + t2i_recall) / 2
    
    # Median rank
    results['i2t_median_rank'] = np.median(i2t_ranks)
    results['t2i_median_rank'] = np.median(t2i_ranks)
    
    return results

def compute_retrieval_metrics(model, dataloader, device, k_values=[1, 5, 10]):
    """
    Compute retrieval metrics on a dataset
    """
    model.eval()
    
    all_image_embeddings = []
    all_text_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get embeddings
            image_emb, text_emb = model(images, input_ids, attention_mask)
            
            all_image_embeddings.append(image_emb.cpu())
            all_text_embeddings.append(text_emb.cpu())
    
    # Concatenate all embeddings
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_image_embeddings, all_text_embeddings.t())
    
    # Compute metrics
    metrics = compute_recall_at_k(similarity_matrix, k_values)
    
    return metrics, similarity_matrix

def print_metrics(metrics, prefix=''):
    """Pretty print metrics"""
    print(f"\n{prefix} Metrics:")
    print("-" * 50)
    for key, value in metrics.items():
        if 'recall' in key:
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.2f}")
    print("-" * 50)