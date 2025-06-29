import torch
from transformers import (
    AutoTokenizer,
)
from models.models import *
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer
from transformers import AutoModel
from utils.utils import to_single_label, fix_seeds, compute_class_weights
from utils.factories import create_text_dataloaders
from data.get_train_eval_dfs import build_train_eval_dfs
from trainers.trainer import Trainer
import yaml
from omegaconf import OmegaConf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    with open("./configs/config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)
    # Assign config values
    model_name = cfg["model"]["name"]
    layers_freeze = int(cfg["model"]["layers_freeze"])
    soft = cfg["training"]["soft"]
    classification_mode = cfg["training"]["classification_mode"]
    model_class_size = int(cfg["model"]["num_classes"])
    epochs = int(cfg["training"]["epochs"])
    seed = int(cfg["training"]["seed"])
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"]["wd"])
    # LOAD & PROCESS CSV‐BASED DATASETS

    print("-"*100)
    print("Load and clean")
    
    if "clip" in model_name:
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    torch.autograd.set_detect_anomaly(True)
    fix_seeds(seed) # fixate seed
    if "jina" in model_name:
        # get embed_dim from config
        tmp = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        embed_dim = tmp.config.hidden_size
        del tmp
        model = JinaMultiLabelClassifier(
            jina_model_name="jinaai/jina-embeddings-v3",
            embed_dim=embed_dim,
            num_labels=model_class_size,
            lora_task="classification"
        ).to(device)
    elif "clip" in model_name:
        model = ClipTextClassifier(
            model_name=model_name,
            num_labels=model_class_size,
        ).to(device)
    else:
        model = CustomRobertaForEmotion(
            model_name,
            num_labels = model_class_size, 
        ).to(device)

    freeze_all_except_last_k_and_head(model, layers_freeze)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(len(trainable_params))
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=wd) 

    dataloaders = create_text_dataloaders(cfg, tokenizer=tokenizer)

    train_dataloader = dataloaders.get("train")
    val_dataloader = dataloaders.get("eval")
    batch = next(iter(train_dataloader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    logits = model(input_ids, attention_mask)
    print("Logits shape:", logits.shape)  
    
    class_weights = compute_class_weights(train_dataloader, model_class_size, device)
    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        print(f"Epoch: {e}/{epochs}")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=cfg,
            tokenizer=tokenizer,
            train_loader=train_dataloader,
            val_loader=val_dataloader
        )
        
        trainer.train()


if __name__ == "__main__":
    main()
