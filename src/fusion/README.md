# ICCVW 2025
## Multimodal Emotion Recognition

This repository contains code for training audio-based emotion recognition models

### Training Multimodal Models 

To train text-based emotion recognition models:
- Use any file starts from `train.py`
- Configure paths, model and training parameters in `config.yaml`

### Preprocessing & Prediction Scripts
- `extract_audio_from_videos_ffmpeg.py` –  extracts audio from video files using `config.yaml`
- `train_multimodal.py` –  train multimodal models from `multimodal_models.py` using `config.yaml`
- `test_multimodal.py` –  extract embeddings and predictions from audio-based model. Configure inference parameters in `config.yaml`.
