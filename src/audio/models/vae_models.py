import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor


class TemporalRelationalModuleV1(nn.Module):
    def __init__(self, input_dim: int, num_segments: int = 3) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class EmotionUncertaintyHeadV1(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.mean_head = nn.Linear(input_dim, num_classes)
        self.logvar_head = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mean_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar
    

class EmotionUncertaintyHeadV2(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim),
            nn.Dropout(0.3),
            nn.Linear(input_dim, num_classes)
        )

        self.logvar_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim),
            nn.Dropout(0.3),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mean_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar
    

class EmotionUncertaintyHeadV3(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim),
            nn.Dropout(0.3),
            nn.Linear(input_dim, num_classes)
        )

        self.logvar_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim),
            nn.Dropout(0.3),
            nn.Linear(input_dim, num_classes)
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mean_head(x)
        logvar = self.logvar_head(x)
        probs = F.softmax(mu / self.temperature, dim=-1)
        return mu, logvar, probs

    def _init_weights(self) -> None:
        for module in [self.mean_head, self.logvar_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)


class EmotionUncertaintyHeadV4(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 logvar_clamp_range: tuple[float, float] = (-3.0, 3.0), temperature: float = 1.5) -> None:
        super().__init__()
        self.mu_layer = nn.Linear(input_dim, output_dim)
        self.logvar_layer = nn.Linear(input_dim, output_dim)
        self.temperature = temperature

        # Clamp range to keep logvar in stable zone
        self.logvar_clamp_range = logvar_clamp_range

        # Safe initialization: small weights, zero bias
        nn.init.xavier_uniform_(self.mu_layer.weight, gain=0.5)
        nn.init.constant_(self.mu_layer.bias, 0.0)

        nn.init.xavier_uniform_(self.logvar_layer.weight, gain=0.1)
        nn.init.constant_(self.logvar_layer.bias, -1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu_layer(x)  # [B, C]
        logvar = self.logvar_layer(x).clamp(*self.logvar_clamp_range)  # [B, C]
        probs = F.softmax(mu / self.temperature, dim=-1)

        return mu, logvar, probs


class WavLMEmotionClassifierV1(nn.Module):
    def __init__(self, pretrained_model_name: str, num_emotions: int) -> None:
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        for name, param in self.encoder.named_parameters():
            if any(name.startswith(f"encoder.layers.{i}") for i in range(6, 12)) or "projector" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.temporal_module = TemporalRelationalModuleV1(self.encoder.config.hidden_size)
        self.uncertainty_head = EmotionUncertaintyHeadV1(256, num_emotions)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        outputs = self.encoder(waveform, output_hidden_states=False, return_dict=True)
        features = outputs.last_hidden_state
        features = self.temporal_module(features)
        mu, logvar = self.uncertainty_head(features)
        probs = F.softmax(mu, dim=-1)
        return {"mu": mu, "logvar": logvar, "emo": probs}
    
    
class WavLMEmotionClassifierV2(nn.Module):
    def __init__(self, pretrained_model_name: str, num_emotions: int) -> None:
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        for name, param in self.encoder.named_parameters():
            if any(name.startswith(f"encoder.layers.{i}") for i in range(6, 12)) or "projector" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.temporal_module = TemporalRelationalModuleV1(self.encoder.config.hidden_size)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)  # Global context across time
        self.global_fc = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

        self.uncertainty_head = EmotionUncertaintyHeadV1(256, num_emotions)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        outputs = self.encoder(waveform, output_hidden_states=False, return_dict=True)
        features = outputs.last_hidden_state  # features: [B, T, D]
        temporal_feat = self.temporal_module(features)         # [B, 256]
        global_feat = self.global_pooling(features.transpose(1, 2)).squeeze(-1)  # [B, D] → [B, input_dim]
        global_feat = self.global_fc(global_feat)       # [B, 256]

        fused = torch.cat([temporal_feat, global_feat], dim=-1)  # [B, 512]
        features = self.combined_fc(fused)             # [B, 256]

        mu, logvar = self.uncertainty_head(features)
        probs = F.softmax(mu, dim=-1)
        return {"mu": mu, "logvar": logvar, "emo": probs}
    

class WavLMEmotionClassifierV3(nn.Module):
    def __init__(self, pretrained_model_name: str, num_emotions: int) -> None:
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        for name, param in self.encoder.named_parameters():
            if any(name.startswith(f"encoder.layers.{i}") for i in range(6, 12)) or "projector" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.temporal_module = TemporalRelationalModuleV1(self.encoder.config.hidden_size)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)  # Global context across time
        self.global_fc = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

        self.uncertainty_head = EmotionUncertaintyHeadV2(256, num_emotions)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        outputs = self.encoder(waveform, output_hidden_states=False, return_dict=True)
        features = outputs.last_hidden_state # features: [B, T, D]
        temporal_feat = self.temporal_module(features)         # [B, 256]
        global_feat = self.global_pooling(features.transpose(1, 2)).squeeze(-1)  # [B, D] → [B, input_dim]
        global_feat = self.global_fc(global_feat)       # [B, 256]

        fused = torch.cat([temporal_feat, global_feat], dim=-1)  # [B, 512]
        features = self.combined_fc(fused)             # [B, 256]

        mu, logvar = self.uncertainty_head(features)
        probs = F.softmax(mu, dim=-1)
        return {"mu": mu, "logvar": logvar, "emo": probs}
    

class WavLMEmotionClassifierV4(nn.Module):
    def __init__(self, pretrained_model_name: str, num_emotions: int) -> None:
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        for name, param in self.encoder.named_parameters():
            if any(name.startswith(f"encoder.layers.{i}") for i in range(6, 12)) or "projector" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.temporal_module = TemporalRelationalModuleV1(self.encoder.config.hidden_size)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)  # Global context across time
        self.global_fc = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

        self.uncertainty_head = EmotionUncertaintyHeadV3(256, num_emotions, temperature=1.5)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        outputs = self.encoder(waveform, output_hidden_states=False, return_dict=True)
        features = outputs.last_hidden_state # features: [B, T, D]
        temporal_feat = self.temporal_module(features)         # [B, 256]
        global_feat = self.global_pooling(features.transpose(1, 2)).squeeze(-1)  # [B, D] → [B, input_dim]
        global_feat = self.global_fc(global_feat)       # [B, 256]

        fused = torch.cat([temporal_feat, global_feat], dim=-1)  # [B, 512]
        features = self.combined_fc(fused)             # [B, 256]

        mu, logvar, probs = self.uncertainty_head(features)
        return {"mu": mu, "logvar": logvar, "emo": probs}
    

class WavLMEmotionClassifierV5(nn.Module):
    def __init__(self, pretrained_model_name: str, num_emotions: int) -> None:
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        for name, param in self.encoder.named_parameters():
            if any(name.startswith(f"encoder.layers.{i}") for i in range(6, 12)) or "projector" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.temporal_module = TemporalRelationalModuleV1(self.encoder.config.hidden_size)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)  # Global context across time
        self.global_fc = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

        self.uncertainty_head = EmotionUncertaintyHeadV4(256, num_emotions, temperature=1.5)

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        with torch.no_grad():
            outputs = self.encoder(waveform, output_hidden_states=False, return_dict=True)
            features = outputs.last_hidden_state  # [B, T, D]
            temporal_feat = self.temporal_module(features)  # [B, 256]
            global_feat = self.global_pooling(features.transpose(1, 2)).squeeze(-1)  # [B, D] → [B, input_dim]
            global_feat = self.global_fc(global_feat)  # [B, 256]

            fused = torch.cat([temporal_feat, global_feat], dim=-1)  # [B, 512]
            combined_features = self.combined_fc(fused)  # [B, 256]

        return combined_features

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        outputs = self.encoder(waveform, output_hidden_states=False, return_dict=True)
        features = outputs.last_hidden_state # features: [B, T, D]
        temporal_feat = self.temporal_module(features)         # [B, 256]
        global_feat = self.global_pooling(features.transpose(1, 2)).squeeze(-1)  # [B, D] → [B, input_dim]
        global_feat = self.global_fc(global_feat)       # [B, 256]

        fused = torch.cat([temporal_feat, global_feat], dim=-1)  # [B, 512]
        features = self.combined_fc(fused)             # [B, 256]

        mu, logvar, probs = self.uncertainty_head(features)
        return {"mu": mu, "logvar": logvar, "emo": probs}


class EmotionUncertaintyHeadV5(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 logvar_clamp_range: tuple[float, float] = (-3.0, 3.0), temperature: float = 1.5) -> None:
        super().__init__()
        self.mu_layer = nn.Linear(input_dim, output_dim)
        self.logvar_layer = nn.Linear(input_dim, output_dim)
        self.temperature = temperature
        self.logvar_clamp_range = logvar_clamp_range

        nn.init.xavier_uniform_(self.mu_layer.weight, gain=0.5)
        nn.init.constant_(self.mu_layer.bias, 0.0)

        nn.init.xavier_uniform_(self.logvar_layer.weight, gain=0.1)
        nn.init.constant_(self.logvar_layer.bias, -1.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu_layer(x)  # [B, C]
        logvar = self.logvar_layer(x).clamp(*self.logvar_clamp_range)  # [B, C]
        z = self.reparameterize(mu, logvar)  # [B, C]
        probs = F.softmax(z / self.temperature, dim=-1)  # [B, C]

        return mu, logvar, z, probs
    

class VAEWavLMEmotionClassifierV6(nn.Module):
    def __init__(self, pretrained_model_name: str, num_emotions: int) -> None:
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        for name, param in self.encoder.named_parameters():
            if any(name.startswith(f"encoder.layers.{i}") for i in range(6, 12)) or "projector" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.temporal_module = TemporalRelationalModuleV1(self.encoder.config.hidden_size)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)  # Global context across time
        self.global_fc = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

        self.uncertainty_head = EmotionUncertaintyHeadV5(256, num_emotions, temperature=1.5)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        outputs = self.encoder(waveform, output_hidden_states=False, return_dict=True)
        features = outputs.last_hidden_state # features: [B, T, D]
        temporal_feat = self.temporal_module(features)         # [B, 256]
        global_feat = self.global_pooling(features.transpose(1, 2)).squeeze(-1)  # [B, D] → [B, input_dim]
        global_feat = self.global_fc(global_feat)       # [B, 256]

        fused = torch.cat([temporal_feat, global_feat], dim=-1)  # [B, 512]
        features = self.combined_fc(fused)             # [B, 256]

        mu, logvar, z, probs = self.uncertainty_head(features)
        return {"mu": mu, "logvar": logvar, "z": z, "emo": probs}
