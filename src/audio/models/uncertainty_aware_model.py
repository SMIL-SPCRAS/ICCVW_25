import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor

class TemporalRelationalModule(nn.Module):
    def __init__(self, input_dim, num_segments=3):
        super().__init__()
        self.num_segments = num_segments
        self.segment_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
            for _ in range(num_segments)
        ])
        self.relational_fc = nn.Sequential(
            nn.Linear(input_dim * num_segments, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        B, T, D = x.shape
        segment_len = T // self.num_segments
        segments = [x[:, i * segment_len:(i + 1) * segment_len, :] for i in range(self.num_segments)]

        segment_outputs = []
        for i, segment in enumerate(segments):
            attn_out, _ = self.segment_attn[i](segment, segment, segment)
            pooled = attn_out.mean(dim=1)
            segment_outputs.append(pooled)

        relational_vector = torch.cat(segment_outputs, dim=-1)
        return self.relational_fc(relational_vector)

class EmotionUncertaintyHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.mean_head = nn.Linear(input_dim, num_classes)
        self.logvar_head = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        mu = self.mean_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar

class CompositeEmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_emotions):
        super().__init__()
        self.temporal_module = TemporalRelationalModule(input_dim)
        self.uncertainty_head = EmotionUncertaintyHead(256, num_emotions)

    def forward(self, x):
        features = self.temporal_module(x)
        mu, logvar = self.uncertainty_head(features)
        probs = F.softmax(mu, dim=-1)
        return {"mu": mu, "logvar": logvar, "emo": probs}

class WavLMEmotionClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_emotions: int):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        for name, param in self.encoder.named_parameters():
            if any(name.startswith(f"encoder.layers.{i}") for i in range(6, 12)) or "projector" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.classifier = CompositeEmotionClassifier(self.encoder.config.hidden_size, num_emotions)

    def forward(self, waveform: torch.Tensor) -> dict:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        outputs = self.encoder(waveform, output_hidden_states=False, return_dict=True)
        features = outputs.last_hidden_state
        return self.classifier(features)
