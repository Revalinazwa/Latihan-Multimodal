# Multimodal Image-Text Retrieval System

Sistem end-to-end untuk Image-Text Retrieval menggunakan shared embedding space (mirip CLIP).

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Flickr8k Dataset

Download dari: [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

### 3. Training

```bash
python train.py --config configs/config.yaml
```

### 4. Evaluation

```bash
python evaluate.py --config configs/config.yaml
```

### 5. Inference

**Text-to-Image Retrieval:**
```bash
python inference.py --query "a dog playing in the park" --mode text2image --top_k 5
```

**Image-to-Text Retrieval:**
```bash
python inference.py --query "path/to/image.jpg" --mode image2text --top_k 5
```

## Arsitektur: Shared Embedding Space (CLIP-like)

### Key Components

1. **Image Encoder**: ResNet50 pretrained + projection head
2. **Text Encoder**: DistilBERT + projection head
3. **Loss Function**: Contrastive Loss (symmetric cross-entropy)
4. **Embedding Dimension**: 512-d shared space

## Evaluation Metrics

### Retrieval Metrics
- **Recall@K**: Percentage correct dalam top-K results
- **Median Rank**: Posisi median item yang benar

### Comparisons
1. Image-only baseline
2. Text-only baseline  
3. Multimodal model

## Expected Results

Dengan Flickr8k (5000 samples, 30 epochs):

| Model | R@1 | R@5 | R@10 |
|-------|-----|-----|------|
| Image-only | ~0.1% | ~0.5% | ~1% |
| Text-only | ~0.1% | ~0.5% | ~1% |
| **Multimodal** | **30-40%** | **60-70%** | **75-85%** |

## Hyperparameter Tuning

Edit `configs/config.yaml`:

```yaml
training:
  batch_size: 64        # ↓ jika OOM
  learning_rate: 0.0001 # Tuning range: 1e-5 to 5e-4
  num_epochs: 10        # 20-50 epochs
  
model:
  embedding_dim: 512    # 256, 512, 1024
  temperature: 0.07     # 0.05 - 0.1
```

## Assignment
- [x] Minimal 2 modality (Image + Text)
- [x] Image-Text Retrieval task
- [x] Dataset: Flickr8k
- [x] Preprocessing: ResNet50 + DistilBERT
- [x] Architecture: Shared Embedding Space
- [x] Loss: Contrastive Loss
- [x] Hyperparameter tuning
- [x] Evaluation: Unimodal vs Multimodal
- [x] Metrics: Recall@K
- [x] Visualizations: t-SNE, PCA, similarity matrix
