# ICCVW 2025

## Visual Emotion Recognition and Multimodal Fusion

This repository contains code for training visual emotion recognition models and performing multimodal fusion at the level of probabilistic emotional predictions.

### Training Visual Models (Face-based)

To train face-based emotion recognition models:
- Use `main.py`
- Configure model and training parameters in `config.toml`
- Define hyperparameter search space in `search_params.toml`

### Training Multimodal Model (Probability Fusion)

To train the multimodal fusion model:
- Use `main_multimodal.py`
- Configure model and training parameters in `config_multimodal.toml`
- Define hyperparameter search space in `search_params.toml`

### Preprocessing & Prediction Scripts

- `get_faces.ipynb` – extract facial regions from video frames.
- `get_C-EXPR_DB_pred.ipynb` – generate compound emotion predictions.
- `clip_video_features.py` – extract CLIP embeddings and obtain scene/text-based emotion predictions.