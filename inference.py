import os
import yaml
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import argparse

from models.multimodal_model import MultimodalRetrievalModel
from utils.dataset import get_transforms

def retrieve_images_from_text(model, query_text, image_paths, image_embeddings, 
                              tokenizer, device, top_k=5):
    """Retrieve top-K images given text query"""
    model.eval()
    
    # Tokenize query
    tokens = tokenizer(
        query_text,
        padding='max_length',
        truncation=True,
        max_length=77,
        return_tensors='pt'
    ).to(device)
    
    # Get text embedding
    with torch.no_grad():
        text_embedding = model.text_encoder(
            tokens['input_ids'],
            tokens['attention_mask']
        )
    
    # Compute similarities
    similarities = torch.matmul(text_embedding, image_embeddings.t()).squeeze(0)
    
    # Get top-K
    top_k_indices = similarities.argsort(descending=True)[:top_k]
    top_k_scores = similarities[top_k_indices]
    
    results = []
    for idx, score in zip(top_k_indices, top_k_scores):
        results.append({
            'image_path': image_paths[idx],
            'score': score.item()
        })
    
    return results

def retrieve_texts_from_image(model, image_path, captions, text_embeddings,
                              transform, device, top_k=5):
    """Retrieve top-K captions given image query"""
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Get image embedding
    with torch.no_grad():
        image_embedding = model.image_encoder(image)
    
    # Compute similarities
    similarities = torch.matmul(image_embedding, text_embeddings.t()).squeeze(0)
    
    # Get top-K
    top_k_indices = similarities.argsort(descending=True)[:top_k]
    top_k_scores = similarities[top_k_indices]
    
    results = []
    for idx, score in zip(top_k_indices, top_k_scores):
        results.append({
            'caption': captions[idx],
            'score': score.item()
        })
    
    return results

def visualize_retrieval(query, results, query_type='text', save_path=None):
    """Visualize retrieval results"""
    if query_type == 'text':
        # Text-to-Image retrieval
        num_results = len(results)
        fig, axes = plt.subplots(1, num_results, figsize=(4*num_results, 4))
        
        if num_results == 1:
            axes = [axes]
        
        for idx, (ax, result) in enumerate(zip(axes, results)):
            img = Image.open(result['image_path'])
            ax.imshow(img)
            ax.set_title(f"Rank {idx+1}\nScore: {result['score']:.3f}", fontsize=10)
            ax.axis('off')
        
        fig.suptitle(f'Query: "{query}"', fontsize=12, y=0.98)
        
    else:
        # Image-to-Text retrieval
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Show query image
        img = Image.open(query)
        axes[0].imshow(img)
        axes[0].set_title('Query Image')
        axes[0].axis('off')
        
        # Show top captions
        caption_text = "Top Retrieved Captions:\n\n"
        for idx, result in enumerate(results):
            caption_text += f"{idx+1}. {result['caption']}\n"
            caption_text += f"   Score: {result['score']:.3f}\n\n"
        
        axes[1].text(0.1, 0.5, caption_text, fontsize=10, 
                    verticalalignment='center', wrap=True)
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def precompute_embeddings(model, dataloader, device):
    """Precompute all embeddings for fast retrieval"""
    model.eval()
    
    all_image_embeddings = []
    all_text_embeddings = []
    all_image_paths = []
    all_captions = []
    
    print("Precomputing embeddings...")
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            img_emb, txt_emb = model(images, input_ids, attention_mask)
            
            all_image_embeddings.append(img_emb)
            all_text_embeddings.append(txt_emb)
            all_image_paths.extend(batch['image_name'])
            all_captions.extend(batch['caption'])
    
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    return all_image_embeddings, all_text_embeddings, all_image_paths, all_captions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--query', type=str, required=True, help='Text query or image path')
    parser.add_argument('--mode', type=str, default='text2image', 
                       choices=['text2image', 'image2text'])
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MultimodalRetrievalModel(config).to(device)
    checkpoint = torch.load(os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data for embedding database
    from utils.dataset import create_dataloaders
    _, _, test_loader = create_dataloaders(config)
    
    # Precompute embeddings
    img_emb, txt_emb, img_paths, captions = precompute_embeddings(model, test_loader, device)
    
    # Perform retrieval
    if args.mode == 'text2image':
        tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
        results = retrieve_images_from_text(
            model, args.query, img_paths, img_emb, tokenizer, device, args.top_k
        )
        
        print(f"\nTop {args.top_k} images for query: '{args.query}'")
        print("-" * 60)
        for i, result in enumerate(results):
            print(f"{i+1}. {result['image_path']} (score: {result['score']:.4f})")
        
        visualize_retrieval(args.query, results, 'text', 
                          os.path.join(config['paths']['results_dir'], 'retrieval_text2img.png'))
    
    else:
        transform = get_transforms('test')
        results = retrieve_texts_from_image(
            model, args.query, captions, txt_emb, transform, device, args.top_k
        )
        
        print(f"\nTop {args.top_k} captions for image: {args.query}")
        print("-" * 60)
        for i, result in enumerate(results):
            print(f"{i+1}. {result['caption']} (score: {result['score']:.4f})")
        
        visualize_retrieval(args.query, results, 'image',
                          os.path.join(config['paths']['results_dir'], 'retrieval_img2text.png'))

if __name__ == '__main__':
    main()