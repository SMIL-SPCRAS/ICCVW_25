import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor


import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor


class WavLMEmotionClassifier(nn.Module):
    """
    Fine-tuned WavLM-based emotion classifier with improved architecture.
    """
    def __init__(self, pretrained_model_name: str, num_emotions: int):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        # Partially unfreeze encoder: only top layers
        for name, param in self.encoder.named_parameters():
            if any(name.startswith(f"encoder.layers.{i}") for i in range(9, 12)) or "projector" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.pooling_strategy = "cls"

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )

    def forward(self, waveform: torch.Tensor) -> dict:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        outputs = self.encoder(waveform, output_hidden_states=False, return_dict=True)

        if self.pooling_strategy == "mean":
            pooled = outputs.last_hidden_state.mean(dim=1)
        elif self.pooling_strategy == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        logits = self.classifier(pooled)
        return {"emo": logits}

