import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from typing import List, Dict, Any


class AudioEmotionDataset(Dataset):
    """
    Dataset for audio-based multi-label emotion classification.
    Loads and processes waveform data and soft label annotations from CSV.
    """
    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        db: str,
        emotion_labels: List[str],
        sample_rate: int = 16000,
        max_length: int = 4,
    ):
        self.df = pd.read_csv(csv_path).dropna(subset=emotion_labels, how='all')
        self.audio_dir = audio_dir
        self.db = db
        self.emotion_labels = emotion_labels
        self.sample_rate = sample_rate
        self.num_samples = int(max_length * sample_rate)

    def __len__(self) -> int:
        return len(self.df)

    def _load_waveform(self, wav_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        waveform = waveform.mean(dim=0, keepdim=True)  # Mono

        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        elif waveform.shape[1] < self.num_samples:
            pad_len = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        
        return waveform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row['audio_name'])
        waveform = self._load_waveform(wav_path)
        emo = torch.tensor(row[self.emotion_labels].values.astype('float32'))

        return waveform, {"emo": emo}, {"db": self.db}
