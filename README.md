# ICCVW 2025

# 🧠 9th_ABAW Compound Expression (CE) Recognition Challenge

This repository contains a modular pipeline for multimodal emotion recognition using **audio**, **video**, **text**, and **scene** modalities. It is structured to support isolated training and evaluation within each modality, as well as ensemble and fusion-based methods.

## ⚙️ Modalities
- [🧠 LLMs promting](./LLMs/)
- [🎵 Audio modality](./src/audio/)
- [🔀 Feature-level fusion](./src/fusion/)
- [📄 Text modality](./src/text/)
- [🤖 Scene modality](./src/text_llm/)
- [🎥 Video modality and probability-level fusion](./src/video/)

## 📁 Project Structure
```
LLMs # LLMs promting and LLMs results
src
├── audio/ # Audio-based emotion recognition, data loading and preprocessing.
├── common/ # Shared utilities and trainer logic across modalities.
├── fusion/ # Feature-level fusion and data preparation.
├── text/ # Text-based emotion recognition, data loading and preprocessing.
├── text_llm/ # LLM-based emotion recognition or Scene modality, data loading and preprocessing.
├── video/ # Video-based emotion recognition, data loading and preprocessing, as well as probability-level fusion.
```

## 🚀 Usage
Each module is (mostly) self-contained. To train or evaluate, navigate to the corresponding directory and run the appropriate script (e.g., `train.py`, `inference.py`, or `notebooks/` for LLM-based modelling).

Example (from `text/` modality):
```bash
cd text
python train.py
```