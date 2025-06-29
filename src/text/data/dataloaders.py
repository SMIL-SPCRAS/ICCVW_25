import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokenized_texts, labels, ds_labels=None):
        self.tokenized_texts = tokenized_texts
        self.labels = labels
        self.ds_labels = ds_labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenized_texts['input_ids'][idx],
            'attention_mask': self.tokenized_texts['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx]),
            'ds_labels': self.ds_labels[idx] if self.ds_labels else []
        }
    
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenized = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    def __len__(self):
        return len(self.tokenized["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized["input_ids"][idx],
            "attention_mask": self.tokenized["attention_mask"][idx],
            "text": self.texts[idx]  # for later use
        }