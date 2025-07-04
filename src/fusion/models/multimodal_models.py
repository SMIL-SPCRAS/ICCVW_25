import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEmotionHead(nn.Module):
    def __init__(self, input_dim: int, num_emotions: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_emotions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class EmotionCompositionNet(nn.Module):
    def __init__(self, num_modalities: int, num_emotions: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = num_modalities * num_emotions
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_emotions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class EmotionFusionModelV1(nn.Module):
    def __init__(self, modality_dims: dict[str, int], num_emotions: int = 8) -> None:
        super().__init__()
        self.modalities = modality_dims.keys()
        self.heads = nn.ModuleDict({
            mod: ModalityEmotionHead(input_dim=dim, num_emotions=num_emotions)
            for mod, dim in modality_dims.items()
        })
        
        self.composer = EmotionCompositionNet(num_modalities=len(modality_dims), num_emotions=num_emotions)

    def extract_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = []
        for mod in self.modalities:
            if mod in features:
                logits.append(self.heads[mod](features[mod]))
            else:
                batch_size = next(iter(features.values())).shape[0]
                device = next(iter(features.values())).device
                logits.append(torch.zeros(batch_size, self.heads[mod].linear.out_features, device=device))
        
        fused_input = torch.cat(logits, dim=-1)
        return fused_input 

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = []
        for mod in self.modalities:
            if mod in features:
                logits.append(self.heads[mod](features[mod]))
            else:
                batch_size = next(iter(features.values())).shape[0]
                device = next(iter(features.values())).device
                logits.append(torch.zeros(batch_size, self.heads[mod].linear.out_features, device=device))
        fused_input = torch.cat(logits, dim=-1)
        output = self.composer(fused_input)
        return {'emo': output}


# Example usage:
# model = EmotionFusionModel({"audio": 512, "text": 768, "video": 1024})
# out = model({"audio": x1, "text": x2, "video": x3})
# loss_fn = nn.BCEWithLogitsLoss()  # multi-label target shape: [B, 8] (e.g., [0,0,0,1,0,1,0,0])


class ModalityAttentionPooling(nn.Module):
    def __init__(self, modality_dims: dict[str, int], hidden_dim: int = 128) -> None:
        super().__init__()
        self.modalities = list(modality_dims.keys())
        self.projections = nn.ModuleDict({
            mod: nn.Linear(dim, hidden_dim) for mod, dim in modality_dims.items()
        })

        self.attn_vector = nn.Parameter(F.normalize(torch.randn(hidden_dim), dim=0))
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        inputs: dict[modality] -> tensor of shape [B, D]
        returns: tensor of shape [B, hidden_dim]
        """
        projected = []
        attn_logits = []

        for mod in self.modalities:
            if mod not in inputs:
                continue
            x = self.projections[mod](inputs[mod])
            projected.append(x)
            score = torch.matmul(x, self.attn_vector)
            attn_logits.append(score)

        attn_logits_tensor = torch.stack(attn_logits, dim=1)
        if mask is not None:
            attn_logits_tensor = attn_logits_tensor.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits_tensor, dim=1)

        fused = torch.stack(projected, dim=1)
        weighted = (attn_weights.unsqueeze(-1) * fused).sum(dim=1)
        return weighted


class EmotionFusionModelV2(nn.Module):
    def __init__(self, modality_dims: dict[str, int], num_emotions: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.modalities = list(modality_dims.keys())
        self.heads = nn.ModuleDict({
            mod: ModalityEmotionHead(input_dim=dim, num_emotions=num_emotions)
            for mod, dim in modality_dims.items()
        })
        self.attn_pooling = ModalityAttentionPooling(
            modality_dims={mod: num_emotions for mod in modality_dims},
            hidden_dim=hidden_dim
        )
        self.classifier = nn.Linear(hidden_dim, num_emotions)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        probs = {}
        mask_list = []
        batch_size = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device

        for mod in self.modalities:
            if mod in features:
                probs[mod] = self.heads[mod](features[mod])
                mask_list.append(torch.ones(batch_size, device=device))
            else:
                probs[mod] = torch.zeros(batch_size, self.heads[mod].linear.out_features, device=device)
                mask_list.append(torch.zeros(batch_size, device=device))

        mask = torch.stack(mask_list, dim=1)
        pooled = self.attn_pooling(probs, mask=mask)
        return {'emo': self.classifier(pooled)}  # [B, 8]


class EmotionTokenEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear(x)
        return self.dropout(self.norm(x))
    

class EmotionFusionModelV3(nn.Module):
    def __init__(self, modality_dims: dict[str, int], num_emotions: int = 8, hidden_dim: int = 128, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.modalities = list(modality_dims.keys())
        self.token_encoders = nn.ModuleDict({
            mod: EmotionTokenEncoder(dim, hidden_dim) for mod, dim in modality_dims.items()
        })
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_emotions)
        )

    def extract_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device

        tokens = []
        key_padding_mask = []

        for mod in self.modalities:
            mod_tensor = features[mod]  # [B, D]
            is_missing = (mod_tensor == 0).all(dim=-1)  # [B]
            key_padding_mask.append(is_missing)

            token = self.token_encoders[mod](mod_tensor)
            tokens.append(token.unsqueeze(1))

        token_seq = torch.cat(tokens, dim=1)  # [B, M, H]
        mask_tensor = torch.stack(key_padding_mask, dim=1)  # [B, M]

        query = self.query_token.expand(batch_size, 1, -1)  # [B, 1, H]
        x = torch.cat([query, token_seq], dim=1)  # [B, 1+M, H]

        padding_mask = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            mask_tensor
        ], dim=1)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        return x[:, 0]  # [B, hidden_dim] – fused emotion feature

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device

        tokens = []
        key_padding_mask = []

        for mod in self.modalities:
            mod_tensor = features[mod]  # [B, D]
            is_missing = (mod_tensor == 0).all(dim=-1)  # [B]
            key_padding_mask.append(is_missing)

            token = self.token_encoders[mod](mod_tensor)
            tokens.append(token.unsqueeze(1))

        token_seq = torch.cat(tokens, dim=1)  # [B, M, H]
        mask_tensor = torch.stack(key_padding_mask, dim=1)  # [B, M]
        
        query = self.query_token.expand(batch_size, -1, -1)  # [B, 1, H]
        x = torch.cat([query, token_seq], dim=1)  # [B, 1+M, H]

        # Добавим False для query токена (он всегда есть)
        padding_mask = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool, device=device), mask_tensor], dim=1)

        x = self.transformer(x, src_key_padding_mask=padding_mask)
        emo_logits = self.classifier(x[:, 0])
        return {'emo': emo_logits}
    

class EmotionFusionModelV4(nn.Module):
    def __init__(self, modality_dims: dict[str, int], num_emotions: int = 8, hidden_dim: int = 128, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.scene_key = None

        for key in modality_dims.keys():
            if str(key).lower().endswith('scene'):
                self.scene_key = str(key)

        self.modalities = [m for m in modality_dims.keys() if m != self.scene_key]
        self.token_encoders = nn.ModuleDict({
            mod: EmotionTokenEncoder(dim, hidden_dim) for mod, dim in modality_dims.items() if mod != self.scene_key
        })
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.scene_proj = nn.Linear(modality_dims[self.scene_key], hidden_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, num_emotions)
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device

        tokens = []
        key_padding_mask = []

        for mod in self.modalities:
            mod_tensor = features[mod]  # [B, D]
            is_missing = (mod_tensor == 0).all(dim=-1)  # [B]
            key_padding_mask.append(is_missing)

            token = self.token_encoders[mod](mod_tensor)
            tokens.append(token.unsqueeze(1))

        token_seq = torch.cat(tokens, dim=1)  # [B, M, H]
        mask_tensor = torch.stack(key_padding_mask, dim=1)  # [B, M]

        query = self.query_token.expand(batch_size, 1, -1)  # [B, 1, H]
        x = torch.cat([query, token_seq], dim=1)  # [B, 1+M, H]

        padding_mask = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            mask_tensor
        ], dim=1)

        fused = self.transformer(x, src_key_padding_mask=padding_mask)[:, 0]  # [B, H]

        scene_feat = features[self.scene_key]  # [B, scene_dim]
        scene_proj = self.scene_proj(scene_feat)  # [B, H]

        joint = torch.cat([fused, scene_proj], dim=-1)  # [B, 2H]
        emo_logits = self.classifier(joint)
        return {'emo': emo_logits}

    def extract_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device

        tokens = []
        key_padding_mask = []

        for mod in self.modalities:
            mod_tensor = features[mod]
            is_missing = (mod_tensor == 0).all(dim=-1)
            key_padding_mask.append(is_missing)

            token = self.token_encoders[mod](mod_tensor)
            tokens.append(token.unsqueeze(1))

        token_seq = torch.cat(tokens, dim=1)
        mask_tensor = torch.stack(key_padding_mask, dim=1)

        query = self.query_token.expand(batch_size, 1, -1)
        x = torch.cat([query, token_seq], dim=1)
        padding_mask = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            mask_tensor
        ], dim=1)

        fused = self.transformer(x, src_key_padding_mask=padding_mask)[:, 0]
        scene_feat = features[self.scene_key]
        scene_proj = self.scene_proj(scene_feat)
        return torch.cat([fused, scene_proj], dim=-1)  # unified feature vector
