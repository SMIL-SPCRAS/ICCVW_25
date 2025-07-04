# ICCVW 2025
## Audio-based Emotion Recognition

This repository contains code for training audio-based emotion recognition models

### Training audio-based Models 

To train text-based emotion recognition models:
- Use any file starts from `train.py`
- Configure paths, model and training parameters in `config.yaml`

### Preprocessing & Prediction Scripts
- `extract_audio_from_videos_ffmpeg.py` –  extracts audio from video files using `config.yaml`
- `train.py` –  train simple WavLM-based models from `models.py` using `config.yaml`
- `train_multihead_whisper.py` –  train whisper-based model from `multihead_models.py` using `config.yaml`
- `train_multihead.py` –  train WavLM-based model from `multihead_models.py` using `config.yaml`
- `train_vae.py` –  train VAE-based from `VAE_models.py` using `config.yaml`
- `train_wrapper.py` –  wrapper for execution several train scripts
- `test.py` –  extract embeddings and predictions from audio-based model. Configure inference parameters in `config.yaml`.
