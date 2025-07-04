import os, json, time, random, gc, copy, datetime as dt
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score, f1_score

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from transformers import CLIPTokenizer, CLIPTextModel, get_cosine_schedule_with_warmup

import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner

EMOTION_LABELS = ["Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise","Other"]

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CsvEmotionDataset(Dataset):
    def __init__(self, csv_files, tokenizer, max_len=128):
        self.df = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
        self.texts  = self.df["text"].tolist()
        self.labels = torch.tensor(self.df[EMOTION_LABELS].values.argmax(1), dtype=torch.long)
        self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.texts[i], self.labels[i]

def collate(batch, tok, max_len):
    txt, lab = zip(*batch)
    enc = tok(list(txt), padding=True, truncation=True,
              max_length=max_len, return_tensors="pt")
    enc["labels"] = torch.stack(lab)
    return enc

def compute_class_weights(labels, num_classes):
    labels = labels if isinstance(labels, np.ndarray) else np.array(labels)
    present = np.unique(labels)
    partial = compute_class_weight("balanced", classes=present, y=labels)
    full = np.zeros(num_classes, dtype=np.float32)
    for c, w in zip(present, partial): full[c] = w
    return torch.tensor(full, dtype=torch.float32)

def make_loader(ds, tok, max_len, bs, shuffle, weighted=False):
    sampler = None
    if weighted:
        cnt = Counter(ds.labels.tolist()); tot = sum(cnt.values())
        w_cls = {c: tot/(len(cnt)*v) for c,v in cnt.items()}
        w_smpl = [w_cls[int(y)] for y in ds.labels]
        sampler = WeightedRandomSampler(w_smpl, len(w_smpl), replacement=True)
        shuffle = False
    return DataLoader(ds, bs, shuffle=shuffle, sampler=sampler,
                      collate_fn=lambda b: collate(b, tok, max_len),
                      num_workers=0, pin_memory=True)

def step(model, batch, device, loss_fn=None, fp16=True):
    batch = {k: v.to(device, non_blocking=True) for k,v in batch.items()}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = loss_fn(logits, batch["labels"]) if loss_fn else None
    return type("O", (), {"logits": logits, "loss": loss})

def epoch_train(model, loader, optimizer, scheduler, device, epoch, loss_fn):
    model.train()
    tot_loss, preds, targs = 0., [], []
    for batch in tqdm(loader, desc=f"Train E{epoch}", leave=False):
        optimizer.zero_grad(set_to_none=True)
        out = step(model, batch, device, loss_fn)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        bs = batch["labels"].size(0)
        tot_loss += out.loss.item() * bs
        preds.extend(out.logits.argmax(1).detach().cpu())
        targs.extend(batch["labels"].cpu())
    loss = tot_loss / len(loader.dataset)
    uar = recall_score(targs, preds, average="macro", zero_division=0)
    return loss, uar

@torch.inference_mode()
def epoch_eval(model, loader, device, epoch, loss_fn):
    model.eval()
    tot_loss, preds, targs = 0., [], []
    for batch in tqdm(loader, desc=f"Dev E{epoch}", leave=False):
        out = step(model, batch, device, loss_fn, fp16=False)
        bs = batch["labels"].size(0)
        tot_loss += out.loss.item() * bs
        preds.extend(out.logits.argmax(1).cpu())
        targs.extend(batch["labels"].cpu())
    loss = tot_loss / len(loader.dataset)
    uar = recall_score(targs, preds, average="macro", zero_division=0)
    war = recall_score(targs, preds, average="weighted", zero_division=0)
    mf1 = f1_score(targs, preds, average="macro", zero_division=0)
    wf1 = f1_score(targs, preds, average="weighted", zero_division=0)
    return loss, uar, war, mf1, wf1

class EmbeddingClassifier(nn.Module):
    def __init__(self, base_model, embedding_dim=512, num_classes=8, freeze_prefixes=None):
        super().__init__()
        self.base = base_model
        if freeze_prefixes:
            for name, param in self.base.named_parameters():
                if any(p in name for p in freeze_prefixes):
                    param.requires_grad = False
        self.classifier = nn.Linear(embedding_dim, num_classes)
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = out.pooler_output  # размерности (batch, hidden_size)
        return self.classifier(cls)

def train_run(cfg):
    fix_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    tok = CLIPTokenizer.from_pretrained(cfg["model_name"])
    ds_tr = CsvEmotionDataset(cfg["train_csv"], tok, cfg["max_len"])
    ds_va = CsvEmotionDataset(cfg["val_csv"], tok, cfg["max_len"])
    ld_tr = make_loader(ds_tr, tok, cfg["max_len"], cfg["bs"], shuffle=True, weighted=False)
    ld_va = make_loader(ds_va, tok, cfg["max_len"], cfg["bs"], shuffle=False)
    base = CLIPTextModel.from_pretrained(cfg["model_name"])
    model = EmbeddingClassifier(base, embedding_dim=base.config.hidden_size,
                                num_classes=len(EMOTION_LABELS),
                                freeze_prefixes=cfg.get("freeze_prefixes") or []).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=compute_class_weights(ds_tr.labels.numpy(), len(EMOTION_LABELS)).to(device)) if cfg.get("use_class_weights", False) else nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), cfg["lr"], weight_decay=cfg["wd"]) if cfg["optimizer"]=="AdamW" else torch.optim.Adam(model.parameters(), cfg["lr"], weight_decay=cfg["wd"])
    tot_steps = cfg["epochs"] * len(ld_tr)
    sched = get_cosine_schedule_with_warmup(opt, int(tot_steps*cfg["warmup"]), tot_steps)
    writer = SummaryWriter(cfg["logdir"])
    best_uar, patience = 0.0, 0
    for ep in range(1, cfg["epochs"]+1):
        t0 = time.time()
        tr_loss, tr_uar = epoch_train(model, ld_tr, opt, sched, device, ep, loss_fn)
        va_loss, va_uar, va_war, va_mf1, va_wf1 = epoch_eval(model, ld_va, device, ep, loss_fn)
        writer.add_scalars("loss", {"train":tr_loss,"val":va_loss}, ep)
        writer.add_scalars("UAR_WAR", {"UAR":va_uar, "WAR":va_war}, ep)
        writer.add_scalars("F1", {"macro":va_mf1,"weighted":va_wf1}, ep)
        print(f"[{ep}/{cfg['epochs']}] train_loss={tr_loss:.4f} val_UAR={va_uar:.4f} macroF1={va_mf1:.3f} {time.time()-t0:.1f}s")
        if va_uar > best_uar:
            best_uar, patience = va_uar, 0
            if best_uar >= cfg.get("min_uar_to_save",0.0):
                torch.save(model.state_dict(), cfg["logdir"]/f"best_{best_uar:.4f}.pth")
        else:
            patience +=1
            if patience>=cfg["patience"]:
                print("Early stopping"); break
    writer.close()
    return best_uar, {"WAR":va_war,"macroF1":va_mf1,"weightedF1":va_wf1}

cfg_base = dict(
    train_csv=[
        "AFEW/Qwen2.5-VL-32B-Instruct/train_segment_with_text_2.csv",
        "AffWild2/Qwen2.5-VL-32B-Instruct/train_segment_with_text_2.csv",
    ],
    val_csv=[
        "AFEW/Qwen2.5-VL-32B-Instruct/dev_segment_with_text_2.csv",
        "AffWild2/Qwen2.5-VL-32B-Instruct/dev_segment_with_text_2.csv",
    ],
    model_name="openai/clip-vit-base-patch32",
    max_len=144,
    bs=32,
    epochs=75,
    lr=1e-5,
    wd=1e-4,
    warmup=0.1,
    optimizer="Adam",
    # freeze_prefixes=["encoder.layer.0", "encoder.layer.1"],  # пример частичной заморозки
    freeze_prefixes=[],
    patience=10,
    min_uar_to_save=0.4,
    seed=42,
    use_class_weights=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    logdir=Path(
        "S:/Alex/WSM_diagnoses/ABAW/logs_clip/" + dt.datetime.now().strftime("%H%M%S")
    ),
)


def objective(trial):
    print(f"📊 Trials already in DB: {len(study.trials)}")

    cfg_t = copy.deepcopy(cfg_base)
    cfg_t.update(
        lr=trial.suggest_float("lr", 1e-5, 1e-4, log=True),
        wd=trial.suggest_float("wd", 1e-6, 1e-2, log=True),
        warmup=trial.suggest_float("warmup", 0.05, 0.15),
        bs=trial.suggest_categorical("bs", [8, 16, 24, 32, 48]),
        freeze_prefixes=trial.suggest_categorical(
            "freeze_prefixes",
            (
                None,
                # "roberta.encoder.layers.20,roberta.encoder.layers.21,roberta.encoder.layers.22,roberta.encoder.layers.23",
                # "roberta.encoder.layers.22,roberta.encoder.layers.23",
            ),
        ),
        max_len=trial.suggest_categorical("max_len", [77]),
        use_class_weights=trial.suggest_categorical("use_class_weights", [True]),
        optimizer=trial.suggest_categorical("optimizer", ["Adam"]),
        logdir=cfg_base["logdir"] / f"trial{trial.number:03d}",
    )

    # Нормализуем freeze_prefixes для безопасности
    val = cfg_t["freeze_prefixes"]
    if val is None or val == []:
        cfg_t["freeze_prefixes"] = []
    elif isinstance(val, str):
        cfg_t["freeze_prefixes"] = val.split(",")
    else:
        cfg_t["freeze_prefixes"] = list(val)

    # Создаём папку лога, если не существует
    cfg_t["logdir"].mkdir(parents=True, exist_ok=True)

    best_uar, extra = train_run(cfg_t)

    for k, v in extra.items():
        trial.set_user_attr(k, v)

    log_str = [f"Trial {trial.number} finished:"]
    log_str.append(f"  UAR: {best_uar:.4f}")
    log_str.append("  Params:")
    for key, value in trial.params.items():
        log_str.append(f"    {key}: {value}")
    log_str.append("  Extra metrics:")
    for key, value in extra.items():
        log_str.append(f"    {key}: {value}")

    log_path = cfg_t["logdir"] / "trial_log.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(log_str))

    print("\n".join(log_str))

    return best_uar



study = optuna.create_study(
    study_name = "emotion_clip",
    direction  = "maximize",
    storage    = "sqlite:///optuna_hpo_clip_32.db",   # файл появится рядом с ноутбуком
    load_if_exists = True,                    # продолжит, если уже есть
    sampler = TPESampler(seed=cfg_base["seed"]),
    pruner  = MedianPruner(n_warmup_steps=1)  # рубим trial после 1-й вал-эпохи
)

# study.optimize(
#     objective,
#     n_trials = 50,        #
#     n_jobs   = 1,         # параллельные процессы; =1, если одна GPU
#     gc_after_trial = True, # чистим CUDA память,
#     show_progress_bar=True
# )


RERUN_BEST = False  # или False — если хочешь обычную оптимизацию

if RERUN_BEST:
    print("🚀 Повторный запуск лучшего trial из Optuna...")
    study = optuna.load_study(
        study_name="emotion_clip",
        storage="sqlite:///optuna_hpo_clip_32.db"
    )
    best_trial = study.best_trial

    cfg_best = copy.deepcopy(cfg_base)
    cfg_best.update(best_trial.params)

    # Нормализуем freeze_prefixes
    if cfg_best["freeze_prefixes"] in (None, []):
        cfg_best["freeze_prefixes"] = []
    elif isinstance(cfg_best["freeze_prefixes"], str):
        cfg_best["freeze_prefixes"] = cfg_best["freeze_prefixes"].split(",")

    cfg_best["max_len"] = min(cfg_best.get("max_len", 77), 77)

    cfg_best["logdir"] = Path("logs_rerun_best") / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg_best["logdir"].mkdir(parents=True, exist_ok=True)

    best_uar, extra = train_run(cfg_best)

    print(f"\n✅ Повторный запуск завершён: UAR={best_uar:.4f}")
    for k, v in extra.items():
        print(f"  {k}: {v:.4f}")
else:
    study.optimize(
        objective,
        n_trials=50,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=True
    )