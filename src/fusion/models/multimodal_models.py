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
    
