import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

from transformers import AutoProcessor

from audio.utils.vad import SileroVAD


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
        emotion_labels: list[str],
        sample_rate: int = 16000,
        max_length: int = 4,
        processor_name: str = None
    ) -> None:
        self.df = pd.read_csv(csv_path)
        if "test" not in csv_path:
            self.df = self.df.dropna(subset=emotion_labels, how='all')
        
        self.audio_dir = audio_dir
        self.db = db
        self.emotion_labels = emotion_labels
        self.sample_rate = sample_rate
        self.num_samples = int(max_length * sample_rate)
        self.vad = SileroVAD(threshold=0.1)
        self.processor = AutoProcessor.from_pretrained(processor_name) if processor_name else None

        if self.processor:
            self.cache_dir = os.path.join("/media/maxim/Databases/9th_ABAW/features", processor_name, db, os.path.basename(csv_path).split('_')[0])
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)
    
    def _get_cache_path(self, audio_name: str) -> str:
        return os.path.join(self.cache_dir, f"{audio_name}.pt")

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, any]]:
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row['audio_name'])
        waveform = self._load_waveform(wav_path)
        emo = torch.tensor(row[self.emotion_labels].values.astype('float32'))

        has_speech = self.vad(waveform, self.sample_rate)

        if self.processor:
            cache_path = self._get_cache_path(row['audio_name'])

            if os.path.exists(cache_path):
                input_tensor = torch.load(cache_path)
            else:
                features = self.processor(
                    waveform.squeeze(0).cpu().numpy(), 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt", 
                    padding="longest",
                )

                if "input_features" in features:
                    input_tensor = features["input_features"].squeeze(0)
                elif "input_values" in features:
                    input_tensor = features["input_values"].squeeze(0)
                else:
                    raise KeyError("Keys 'input_features' or 'input_values' not found.")
                
                torch.save(input_tensor, cache_path)
        else:
            input_tensor = waveform

        return input_tensor, {"emo": emo}, {"db": self.db, "has_speech": has_speech, "audio_name": row['audio_name']}

