import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel
from transformers import WhisperModel
from transformers import WhisperConfig, WhisperModel

class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int) -> None:
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
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(input_dim * 2, input_dim)  # reduce to input_dim
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.projection(out)
        out = self.norm(out + x)  # Residual connection
        return out
    

class MultiHeadWavLMEmotionClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_emotions: int, num_heads: int = 3) -> None:
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        # Фризим слои, аналогично V4
        for name, param in self.encoder.named_parameters():
            if any(name.startswith(f"encoder.layers.{i}") for i in range(6, 12)) \
               or "projector" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.sequence_model = ResidualBiLSTMBlock(self.encoder.config.hidden_size)
        self.pooling = AttentionPooling(self.encoder.config.hidden_size)

        # Множество голов
        self.heads = nn.ModuleList([
            nn.Sequential(
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
            for _ in range(num_heads)
        ])

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        outputs = self.encoder(waveform, output_hidden_states=False, return_dict=True)
        features = outputs.last_hidden_state  # (B, T, D)
        features = self.sequence_model(features)
        pooled = self.pooling(features)

        logits_list = [head(pooled) for head in self.heads]  # ← создаешь список логитов от каждой головы

        outputs = {
            "emo_heads": logits_list,
            "emo": logits_list[0]
        }
        
        return outputs
    

class MultiHeadWhisperEmotionClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_emotions: int, 
                 num_heads: int = 3, max_position: int = 200) -> None:
        super().__init__()
        whisper_model = AutoModel.from_pretrained(pretrained_model_name)

        # Cut embed_positions
        old_weights = whisper_model.encoder.embed_positions.weight.data
        new_embed = nn.Embedding(max_position, whisper_model.config.d_model)
        new_embed.weight.data.copy_(old_weights[:max_position])
        whisper_model.encoder.embed_positions = new_embed

        
        # 3. update config
        whisper_model.config.max_source_positions = max_position

        self.encoder = whisper_model.encoder

        self.sequence_model = ResidualBiLSTMBlock(whisper_model.config.d_model)
        self.pooling = AttentionPooling(whisper_model.config.d_model)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(whisper_model.config.d_model, 512),
                nn.ReLU(),
                nn.LayerNorm(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(0.3),
                nn.Linear(256, num_emotions)
            )
            for _ in range(num_heads)
        ])

    def forward(self, input_features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(input_features).last_hidden_state  # (B, T, D)
        hidden = self.sequence_model(hidden)
        pooled = self.pooling(hidden)

        logits_list = [head(pooled) for head in self.heads]
        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)  # head-wise mean

        return {
            "emo_heads": logits_list,
            "emo": avg_logits
        }

