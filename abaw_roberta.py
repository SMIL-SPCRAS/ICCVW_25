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

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner

EMOTION_LABELS = [
    "Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise","Other"
    # "Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise"
]

def fix_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class CsvEmotionDataset(Dataset):
    def __init__(self, csv_files, tokenizer, max_len=128):
        self.df = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
        self.texts  = self.df["text"].tolist()
        self.labels = torch.tensor(self.df[EMOTION_LABELS].values.argmax(1), dtype=torch.long)
        self.tok = tokenizer; self.max_len = max_len
    def __len__(self):  return len(self.labels)
    def __getitem__(self, i): return self.texts[i], self.labels[i]

def collate(batch, tok, max_len):
    txt, lab = zip(*batch)
    enc = tok(list(txt), padding=True, truncation=True,
              max_length=max_len, return_tensors="pt")
    enc["labels"] = torch.stack(lab)
    return enc


def compute_class_weights(labels, num_classes):
    """
    Вычисляет веса классов для использования в CrossEntropyLoss.
    :param labels: torch.Tensor или numpy массив с метками
    :param num_classes: общее число классов
    :return: torch.Tensor весов длины num_classes
    """

    labels = labels if isinstance(labels, np.ndarray) else np.array(labels)
    present_classes = np.unique(labels)

    weights_partial = compute_class_weight(
        class_weight="balanced", classes=present_classes, y=labels
    )

    full_weights = np.zeros(num_classes, dtype=np.float32)
    for cls, w in zip(present_classes, weights_partial):
        full_weights[cls] = w

    return torch.tensor(full_weights, dtype=torch.float32)


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
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**batch)
        logits, labels = out.logits, batch["labels"]
        loss = loss_fn(logits, labels) if loss_fn else out.loss
    out.loss = loss
    return out


def epoch_train(model, loader, optimizer, scheduler, device, epoch, loss_fn):
    """Обучающая эпоха: возвращает (loss, UAR)."""
    model.train()
    tot_loss, preds, targets = 0., [], []

    for batch in tqdm(loader, desc=f"Train E{epoch}", leave=False):
        optimizer.zero_grad(set_to_none=True)

        out = step(model, batch, device, loss_fn=loss_fn)
        out.loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()

        bs = batch["labels"].size(0)
        tot_loss += out.loss.item() * bs
        preds.extend(out.logits.argmax(1).detach().cpu())
        targets.extend(batch["labels"].cpu())

    loss = tot_loss / len(loader.dataset)
    uar  = recall_score(targets, preds, average="macro", zero_division=0)
    return loss, uar


@torch.inference_mode()
def epoch_eval(model, loader, device, epoch, loss_fn):
    """
    Валидационная эпоха.
    Возвращает: loss, UAR, WAR, macro-F1, weighted-F1
    """
    model.eval()
    tot_loss, preds, targets = 0., [], []

    for batch in tqdm(loader, desc=f"Dev E{epoch}", leave=False):
        out = step(model, batch, device, loss_fn=loss_fn, fp16=False)
        bs = batch["labels"].size(0)
        tot_loss += out.loss.item() * bs
        preds.extend(out.logits.argmax(1).cpu())
        targets.extend(batch["labels"].cpu())

    loss = tot_loss / len(loader.dataset)
    uar  = recall_score(targets, preds, average="macro",    zero_division=0)
    war  = recall_score(targets, preds, average="weighted", zero_division=0)
    mf1  = f1_score     (targets, preds, average="macro",   zero_division=0)
    wf1  = f1_score     (targets, preds, average="weighted",zero_division=0)
    return loss, uar, war, mf1, wf1


def train_run(cfg):
    fix_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    ds_tr = CsvEmotionDataset(cfg["train_csv"], tok, cfg["max_len"])
    ld_tr = make_loader(ds_tr, tok, cfg["max_len"], cfg["bs"], True, weighted=False)
    
    ds_va_afew = CsvEmotionDataset(cfg["val_csv_afew"], tok, cfg["max_len"])
    ds_va_affw = CsvEmotionDataset(cfg["val_csv_affw"], tok, cfg["max_len"])

    ld_va_afew = make_loader(ds_va_afew, tok, cfg["max_len"], cfg["bs"], False)
    ld_va_affw = make_loader(ds_va_affw, tok, cfg["max_len"], cfg["bs"], False)
    
    # ds_va = CsvEmotionDataset(cfg["val_csv"],   tok, cfg["max_len"])
    # ld_va = make_loader(ds_va, tok, cfg["max_len"], cfg["bs"], False)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(EMOTION_LABELS), ignore_mismatched_sizes=True
    ).to(device, dtype=torch.bfloat16)

    # === Class Weights ===
    if cfg.get("use_class_weights", False):
        print("🎯 Используем веса классов для CrossEntropyLoss")
        class_weights = compute_class_weights(ds_tr.labels.numpy(), len(EMOTION_LABELS)).to(
            device
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    if cfg["freeze_layers"] > 0:
        if hasattr(model, "distilbert"):                    # DistilBERT / DistilRoBERTa
            layers = model.distilbert.transformer.layer
        elif hasattr(model, "roberta"):                     # полная RoBERTa
            layers = model.roberta.encoder.layer
        elif hasattr(model, "bert"):                        # BERT-семейство
            layers = model.bert.encoder.layer
        else:
            layers = []

        k = min(cfg["freeze_layers"], len(layers))
        for layer in layers[:k]:
            for p in layer.parameters():
                p.requires_grad = False

    if cfg["optimizer"] == "AdamW":
        opt = AdamW(model.parameters(), cfg["lr"], weight_decay=cfg["wd"])
    elif cfg["optimizer"] == "Adam":
        opt = torch.optim.Adam(model.parameters(), cfg["lr"], weight_decay=cfg["wd"])
    elif cfg["optimizer"] == "SGD":
        opt = torch.optim.SGD(model.parameters(), cfg["lr"], momentum=0.9, weight_decay=cfg["wd"])
    else:
        raise ValueError("Unknown optimizer")

    tot_steps = cfg["epochs"]*len(ld_tr)
    sched = get_cosine_schedule_with_warmup(opt, int(tot_steps*cfg["warmup"]), tot_steps)
    writer = SummaryWriter(cfg["logdir"])

    best_uar, patience = 0.0, 0
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # ---- train ----
        tr_loss, tr_uar = epoch_train(model, ld_tr, opt, sched, device, ep, loss_fn)

        # ---- val ----
        # va_loss, va_uar, va_war, va_mf1, va_wf1 = epoch_eval(
        #     model, ld_va, device, ep, loss_fn
        # )
        _, uar_afew, war_afew, mf1_afew, wf1_afew = epoch_eval(model, ld_va_afew, device, ep, loss_fn)
        _, uar_affw, war_affw, mf1_affw, wf1_affw = epoch_eval(model, ld_va_affw, device, ep, loss_fn)
                

        # ---- усреднённые метрики ----
        va_uar = (uar_afew + uar_affw) / 2
        va_war = (war_afew + war_affw) / 2
        va_mf1 = (mf1_afew + mf1_affw) / 2
        va_wf1 = (wf1_afew + wf1_affw) / 2
        
        print(f"[{ep}/{cfg['epochs']}] val_UAR: {va_uar:.4f} (AFEW: {uar_afew:.4f}, AffWild2: {uar_affw:.4f})")
         
        # ---- TensorBoard ----
        writer.add_scalars("UAR_vs_WAR", {"UAR": va_uar, "WAR": va_war}, ep)
        writer.add_scalars("F1", {"macro": va_mf1, "weighted": va_wf1}, ep)

        # ---- console ----
        print(f"E{ep:02d} train_loss={tr_loss:.4f} "
              f"UAR={va_uar:.4f} WAR={va_war:.4f} "
              f"macroF1={va_mf1:.4f} weightedF1={va_wf1:.4f} "
              f"{time.time()-t0:.1f}s")

        # ---- early-stopping ----
        if va_uar > best_uar:
            best_uar, patience = va_uar, 0
            if best_uar >= cfg.get("min_uar_to_save", 0.0):
                torch.save(model.state_dict(), cfg["logdir"] / f"best_{best_uar:.4f}.pth")
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("Early stopping.")
                break

    writer.close()
    return best_uar, {"WAR": va_war, "macroF1": va_mf1, "weightedF1": va_wf1}

cfg_base = dict(
    train_csv=[
        "AFEW/Qwen2.5-VL-32B-Instruct/train_segment_with_text_1.csv",
        "AffWild2/Qwen2.5-VL-32B-Instruct/train_segment_with_text_1.csv",
    ],
    # val_csv=[
    #     "AFEW/Qwen2.5-VL-32B-Instruct/dev_segment_with_text_1.csv",
    #     "AffWild2/Qwen2.5-VL-32B-Instruct/dev_segment_with_text_1.csv",
    # ],
    val_csv_afew = ["AFEW/Qwen2.5-VL-32B-Instruct/dev_segment_with_text_1.csv"],
    val_csv_affw = ["AffWild2/Qwen2.5-VL-32B-Instruct/dev_segment_with_text_1.csv"],
    model_name="j-hartmann/emotion-english-distilroberta-base",
    max_len=128,
    bs=32,
    epochs=75,
    lr=1e-5,
    wd=1e-4,
    warmup=0.1,
    optimizer="AdamW",
    freeze_layers=0,  # 0–5 для DistilRoBERTa
    patience=10,
    min_uar_to_save=0.38,
    seed=42,
    use_class_weights=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    logdir=Path(
        "S:/Alex/WSM_diagnoses/ABAW/logs_roberta/" + dt.datetime.now().strftime("%H%M%S")
    ),
)

def objective(trial):

    print(f"📊 Trials already in DB: {len(study.trials)}")

    # ---- 1. сформировать конфиг для trial’а ----
    # cfg_t = cfg_base.copy()
    cfg_t = copy.deepcopy(cfg_base)
    
    cfg_t.update(
        lr=trial.suggest_float("lr", 1e-5, 1e-4, log=True),
        wd = trial.suggest_float("wd", 1e-6, 1e-2, log=True),
        warmup=trial.suggest_float("warmup", 0.05, 0.2),
        bs=trial.suggest_categorical("bs", [8, 16, 24, 32, 48]),
        freeze_layers=trial.suggest_int("freeze", 0, 5),
        max_len=trial.suggest_categorical("max_len", [112, 144, 192]),
        use_weights = trial.suggest_categorical("use_class_weights", [True]),
        optimizer=trial.suggest_categorical("optimizer", ["Adam"]),
        logdir=cfg_base["logdir"] / f"trial{trial.number:03d}",
    )

    # ---- 2. обучение ----
    best_uar, extra = train_run(cfg_t)   # train_run теперь отдаёт (uar, dict)

    # ---- 3. логируем побочные метрики, пригодится для анализа ----
    for k, v in extra.items():
        trial.set_user_attr(k, v)

    # ---- ЛОГИРОВАНИЕ В ФАЙЛ И КОНСОЛЬ ----
    log_str = [f"Trial {trial.number} finished:"]
    log_str.append(f"  UAR: {best_uar:.4f}")
    log_str.append("  Params:")
    for key, value in trial.params.items():
        log_str.append(f"    {key}: {value}")
    log_str.append("  Extra metrics:")
    for key, value in extra.items():
        log_str.append(f"    {key}: {value}")

    # путь до trial-лог-файла (внутри logdir/trialXXX)
    log_path = cfg_t["logdir"] / "trial_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        f.write("\n".join(log_str))

    # (необязательно) также вывести в консоль
    print("\n".join(log_str))

    return best_uar                       # Optuna будет МАКСИМИЗИРОВАТЬ


study = optuna.create_study(
    study_name = "emotion_roberta",
    direction  = "maximize",
    storage    = "sqlite:///optuna_hpo_roberta_32.db",   # файл появится рядом с ноутбуком
    load_if_exists = True,                    # продолжит, если уже есть
    sampler = TPESampler(seed=cfg_base["seed"]),
    pruner  = MedianPruner(n_warmup_steps=3)  # рубим trial после 1-й вал-эпохи
)


RERUN_BEST = True  # или False — если хочешь обычную оптимизацию

if RERUN_BEST:
    print("🚀 Повторный запуск лучшего trial из Optuna...")
    study = optuna.load_study(
        study_name="emotion_roberta",
        storage="sqlite:///optuna_hpo_roberta_32.db"
    )
    best_trial = study.best_trial

    cfg_best = copy.deepcopy(cfg_base)
    cfg_best.update(best_trial.params)

    # Нормализуем freeze_prefixes
    # if cfg_best["freeze_prefixes"] in (None, []):
    #     cfg_best["freeze_prefixes"] = []
    # elif isinstance(cfg_best["freeze_prefixes"], str):
    #     cfg_best["freeze_prefixes"] = cfg_best["freeze_prefixes"].split(",")

    cfg_best["max_len"] = min(cfg_best.get("max_len", 144), 512)

    cfg_best["logdir"] = Path("logs_rerun_best") / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg_best["logdir"].mkdir(parents=True, exist_ok=True)

    best_uar, extra = train_run(cfg_best)

    print(f"\n✅ Повторный запуск завершён: UAR={best_uar:.4f}")
    for k, v in extra.items():
        print(f"  {k}: {v:.4f}")
else:  
    study.optimize(
        objective,
        n_trials = 100,        #
        n_jobs   = 1,         # параллельные процессы; =1, если одна GPU
        gc_after_trial = True, # чистим CUDA память,
        show_progress_bar=True
    )