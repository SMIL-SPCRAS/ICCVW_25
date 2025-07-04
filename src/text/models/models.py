import torch
import torch.nn as nn
from transformers import AutoModel, CLIPModel

class JinaMultiLabelClassifier(nn.Module):
    def __init__(self, jina_model_name: str, embed_dim: int, num_labels: int, lora_task: str = "classification"):
        super().__init__()
        self.jina = AutoModel.from_pretrained(jina_model_name, trust_remote_code=True)
        task_id = self.jina._adaptation_map.get(lora_task, None)
        if task_id is None:
            raise ValueError(f"Task '{lora_task}' not in adaptation_map")
        self.task_id = task_id
        # get embed_dim from config
        self.embed_dim = embed_dim
        self.classifier = nn.Linear(embed_dim, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        adapter_mask = torch.full((batch_size,), self.task_id, dtype=torch.int32, device=input_ids.device)
        outputs = self.jina(input_ids=input_ids, attention_mask=attention_mask, adapter_mask=adapter_mask)
        token_embeddings = outputs[0]  # last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        lengths = torch.clamp(mask.sum(dim=1), min=1e-9)
        emb = summed / lengths
        logits = self.classifier(emb)
        return logits


class ClipTextClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.model.config.projection_dim, num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_ids=None, attention_mask=None):
        text_features = self.model.get_text_features(input_ids=input_ids)
        return self.proj(text_features)


class CustomRobertaForEmotion(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids=None, attention_mask=None):
        # Get last hidden state
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        # Mean-pool
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        lengths = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / lengths
        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits

def freeze_all_except_last_k_and_head(
    model,
    k: int = 3,
    lora_keywords: tuple[str] = ("lora",),  # For LoRA parameter detection
):
    """
    Unified function to freeze all parameters in a transformer-based model
    except the last `k` encoder layers and the classifier head.
    
    Parameters:
    - model: model wrapper (e.g., HuggingFace classifier, Jina, or CLIP wrapper)
    - k (int): number of final layers to keep unfrozen
    - lora_keywords (tuple): substrings used to identify LoRA adapters (optional)
    """

    # CLIPTextClassifier
    if hasattr(model, "model") and hasattr(model, "proj"):
        clip_model = model.model
        encoder_layers = getattr(clip_model.text_model.encoder, "layers", None)
        classifier_params = model.proj.parameters()
        model_type = "CLIP"

    # Jina
    elif hasattr(model, "jina") and hasattr(model, "classifier"):
        jina_model = model.jina
        classifier_params = model.classifier.parameters()
        total_layers = getattr(jina_model.config, "num_hidden_layers", None)
        encoder_layers = None

        if hasattr(jina_model, "base_model") and hasattr(jina_model.base_model, "encoder"):
            encoder_layers = getattr(jina_model.base_model.encoder, "layers", None)
        elif hasattr(jina_model, "encoder"):
            encoder_layers = getattr(jina_model.encoder, "layers", None)

        if encoder_layers is None:
            raise ValueError("Could not find encoder layers in Jina model")

        model_type = "Jina"

    # HuggingFace-style transformer model
    elif hasattr(model, "base_model") and hasattr(model, "classifier"):
        base_model = model.base_model
        encoder_layers = getattr(base_model.encoder, "layer", None)
        classifier_params = model.classifier.parameters()
        model_type = "HuggingFace"

    else:
        raise TypeError("Unsupported model type or structure")

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last k transformer layers
    if encoder_layers is not None:
        total_layers = len(encoder_layers)
        start_idx = max(0, total_layers - k)
        for layer in encoder_layers[start_idx:]:
            for p in layer.parameters():
                p.requires_grad = True
    else:
        print("Could not locate encoder layers to unfreeze.")

    # Unfreeze classifier head
    for p in classifier_params:
        p.requires_grad = True

    # Unfreeze LoRA adapters if model supports them
    if model_type == "Jina":
        for name, p in jina_model.named_parameters():
            if any(kw.lower() in name.lower() for kw in lora_keywords):
                p.requires_grad = True

    # Print stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[{model_type}] Trainable params: {trainable}/{total} "
          f"(unfrozen last {k} layers + classifier head{' + LoRA' if model_type == 'Jina' else ''})")
