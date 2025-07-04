import pandas as pd
from utils.clean_whisper import clean
from utils.extract_labels import extract_labels

def load_and_process_csv(path):
    df = pd.read_csv(path)
    df = df[df["text"].notna()].dropna().reset_index(drop=True)
    texts = [clean(str(t)) for t in df["text"].tolist()]
    labels = extract_labels(df)
    return texts, labels

def prepare_dataframe(texts, labels, ds_labels=None):
    data = {"text": texts, "labels": labels}
    if ds_labels is not None:
        data["ds_labels"] = ds_labels
    return pd.DataFrame(data)

def filter_empty_labels(df):
    return df[df["labels"].apply(lambda x: not all(v == 0 for v in x))]

def build_train_eval_dfs(config):
    train_texts, train_labels = [], []
    for entry in config["datasets"]["train"]:
        texts, labels = load_and_process_csv(entry["csv"])
        train_texts.extend(texts)
        train_labels.extend(labels)

    df_train = prepare_dataframe(train_texts, train_labels)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train = filter_empty_labels(df_train)

    eval_texts, eval_labels, eval_ds_labels = [], [], []
    for entry in config["datasets"]["eval"]:
        texts, labels = load_and_process_csv(entry["csv"])
        ds_label = entry.get("ds_label", "unknown")
        eval_texts.extend(texts)
        eval_labels.extend(labels)
        eval_ds_labels.extend([ds_label] * len(texts))

    df_eval = prepare_dataframe(eval_texts, eval_labels, ds_labels=eval_ds_labels)
    df_eval = df_eval.sample(frac=1, random_state=42).reset_index(drop=True)
    df_eval = filter_empty_labels(df_eval)

    return df_train, df_eval