# ICCVW 2025
## Text Emotion Recognition

This repository contains code for training text emotion recognition models

### Training Text Models 

To train text-based emotion recognition models:
- Use `train.py`
- Configure model and training parameters in `configs/config.yaml`

### Preprocessing & Prediction Scripts
- `video_dataset_process.py` – using selected csv extract audios from video dataset. Then transcribe text and write it in selected csv. Arguments there are: 

  - --folders / -f: Required. One or more root folders containing emotion-based subdirectories (e.g., Train_AFEW, Val_AFEW) where videos are stored.

  - --model / -m: Required. ASR (automatic speech recognition) model to use. Either a HuggingFace model ID (e.g., openai/whisper-base) or 'parakeet' to use NVIDIA NeMo's Parakeet model.

  - --csvs / -c: Optional. One or more paths to CSV files listing video names and metadata. If not provided, the script defaults to processing all *.csv files in the current directory.

  - --device / -d: Optional. Device ID for inference: 0, 1, etc. for specific CUDA GPU or -1 to use CPU (default)

- `inference.py` – extract text embeddings and obtain text-based emotion predictions. Configure inference parameters in `configs/config.yaml`.
