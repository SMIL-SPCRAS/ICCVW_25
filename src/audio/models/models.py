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
    

class WavLMEmotionClassifierV2(nn.Module):
    """
    Fine-tuned WavLM-based emotion classifier with improved architecture.
    """
    def __init__(self, pretrained_model_name: str, num_emotions: int):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        # Partially unfreeze encoder: only top layers
        for name, param in self.encoder.named_parameters():
            if any(name.startswith(f"encoder.layers.{i}") for i in range(6, 12)) or "projector" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.pooling_strategy = "mean"

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
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


class SelfAttentionClassifier(nn.Module):
    def __init__(self, input_dim, num_emotions):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True, bidirectional=True)
        self.layernorm = nn.LayerNorm(input_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        gru_output, _ = self.gru(attn_output)
        pooled = gru_output.mean(dim=1)
        normed = self.layernorm(pooled)
        return self.classifier(normed)

class WavLMEmotionClassifierV3(nn.Module):
    """
    Improved WavLM-based emotion classifier with attention and recurrent layer.
    """
    def __init__(self, pretrained_model_name: str, num_emotions: int):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        for name, param in self.encoder.named_parameters():
            if any(name.startswith(f"encoder.layers.{i}") for i in range(6, 12)) or "projector" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.pooling_strategy = "full_sequence"

        self.classifier = SelfAttentionClassifier(self.encoder.config.hidden_size, num_emotions)

    def forward(self, waveform: torch.Tensor) -> dict:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        outputs = self.encoder(waveform, output_hidden_states=False, return_dict=True)

        if self.pooling_strategy == "full_sequence":
            features = outputs.last_hidden_state
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        logits = self.classifier(features)
        return {"emo": logits}
    

class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)  # (B, T, 1)
        weights = torch.softmax(weights, dim=1)  # (B, T, 1)
        return (x * weights).sum(dim=1)  # (B, D)


class ResidualBiLSTMBlock(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(input_dim * 2, input_dim)  # reduce to input_dim
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.projection(out)
        out = self.norm(out + x)  # Residual connection
        return out


class WavLMEmotionClassifierV4(nn.Module):
    def __init__(self, pretrained_model_name: str, num_emotions: int):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        for name, param in self.encoder.named_parameters():
            if any(name.startswith(f"encoder.layers.{i}") for i in range(6, 12)) or "projector" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.sequence_model = ResidualBiLSTMBlock(self.encoder.config.hidden_size)
        self.pooling = AttentionPooling(self.encoder.config.hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )

    def forward(self, waveform: torch.Tensor) -> dict:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        outputs = self.encoder(waveform, output_hidden_states=False, return_dict=True)
        features = outputs.last_hidden_state  # (B, T, D)

        features = self.sequence_model(features)  # residual BiLSTM
        pooled = self.pooling(features)  # attention pooling

        logits = self.classifier(pooled)
        return {"emo": logits}