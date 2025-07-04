import os
import pickle
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.clean_whisper import clean
from utils.global_variables import EMOTIONS_EXT
from models.models import JinaMultiLabelClassifier, CustomRobertaForEmotion, ClipTextClassifier
from data.dataloaders import InferenceDataset

from omegaconf import OmegaConf

def load_model(model_dir):
    with open(os.path.join(model_dir, "best_model.pkl"), "rb") as f:
        meta = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model_name = meta["model_name"]
    num_labels = meta["num_labels"]

    if meta.get("is_jina"):
        model = JinaMultiLabelClassifier(
            jina_model_name=meta["jina_model_name"],
            embed_dim=meta["embed_dim"],
            num_labels=num_labels,
            lora_task=meta["lora_task"]
        )
    elif "clip" in model_name:
        model = ClipTextClassifier(model_name, num_labels=num_labels)
    else:
        model = CustomRobertaForEmotion(model_name, num_labels=num_labels)

    model.load_state_dict(meta["state_dict"])
    model.eval().cuda()
    return model, tokenizer, meta


def load_and_prepare_data(csv_path, tokenizer, meta, batch_size):
    df = pd.read_csv(csv_path)
    df["text_clean"] = df["text"].astype(str).apply(clean)

    max_len = 1024 if meta.get("is_jina") else 77 if "clip" in meta["model_name"] else None
    dataset = InferenceDataset(df["text_clean"].tolist(), tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size)
    return df, loader


def get_pooled_embeddings(model, input_ids, attn):
    if isinstance(model, JinaMultiLabelClassifier):
        outputs = model.jina(input_ids=input_ids, attention_mask=attn)
        hidden = outputs[0]
    elif isinstance(model, ClipTextClassifier):
        outputs = model.model.text_model(input_ids=input_ids, attention_mask=attn)
        hidden = outputs[0]
    else:
        outputs = model.base_model(input_ids=input_ids, attention_mask=attn, return_dict=True)
        return outputs.last_hidden_state[:, 0, :]

    mask = attn.unsqueeze(-1).expand(hidden.size()).float()
    summed = torch.sum(hidden * mask, dim=1)
    lengths = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / lengths


def run_inference(model, loader):
    all_probs, all_features, all_texts = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].cuda()
            attn = batch["attention_mask"].cuda()

            logits = model(input_ids, attn)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pooled = get_pooled_embeddings(model, input_ids, attn)

            all_probs.append(probs)
            all_features.append(pooled.cpu().numpy())
            all_texts.extend(batch["text"])

    return (
        np.vstack(all_probs),
        np.vstack(all_features),
        all_texts
    )


def create_output_df(df, probs, preds, num_labels):
    prob_df = pd.DataFrame(probs, columns=[f"{c}_prob" for c in EMOTIONS_EXT[:num_labels]])
    
    if isinstance(preds[0], (np.integer, int, np.int32, np.int64)):
        pred_df = pd.DataFrame(0, index=np.arange(len(preds)), columns=[f"{c}_bin" for c in EMOTIONS_EXT[:num_labels]])
        for i, class_idx in enumerate(preds):
            pred_df.iloc[i, class_idx] = 1
    else:
        pred_df = pd.DataFrame(preds, columns=[f"{c}_bin" for c in EMOTIONS_EXT[:num_labels]])

    return pd.concat([df, prob_df, pred_df], axis=1)


def save_outputs(result_df, output_path, texts, probs, features):
    result_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    records = []
    for i, text in enumerate(texts):
        if text != "nan":
            records.append({
                "video_name": result_df["video_name"].iloc[i] if "video_name" in result_df.columns else f"row_{i}",
                "predictions": probs[i].tolist(),
                "features": features[i].tolist(),
                "text": text
            })

    pkl_path = output_path.replace(".csv", ".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)
    print(f"Pickle saved to {pkl_path}")


def run_pipeline(config):
    model, tokenizer, meta = load_model(config.inference.model_dir)
    df, loader = load_and_prepare_data(config.inference.csv_path, tokenizer, meta, config.inference.batch_size)
    probs, features, texts = run_inference(model, loader)
    preds = np.argmax(probs, axis=1)
    result_df = create_output_df(df, probs, preds, meta["num_labels"])
    save_outputs(result_df, config.inference.output, texts, probs, features)


def main():
    with open("./configs/config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()