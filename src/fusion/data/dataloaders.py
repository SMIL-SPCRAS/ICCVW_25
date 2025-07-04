import os
import pickle
from collections import defaultdict

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


MODALITY_FEATURES_KEYS = {
    "audio": "features", # 256 audio=audio_8
    "audio_8": "features", # 256
    "audio_25": "features", # 256
    "clip": "image_features", # 256
    "new_scene": "features", # 768
    "scene": "features", # 1024
    "text": "features", # 512
    "video": "features", # 128
    "video_static": "static_features" # 128
}

MODALITY_FILENAME_KEYS = {
    "audio": "metas", # metas -> audio_name ['000149120___Angry_0000.wav'] audio=audio_8
    "audio_8": "metas", # metas -> audio_name ['000149120___Angry_0000.wav']
    "audio_25": "metas", # metas -> audio_name ['000149120___Angry_0000.wav']
    "clip": "video_name", # 000149120___Angry_0000
    "new_scene": "video_name", # 000149120___Angry_0000.avi
    "scene": "video_name", # 000149120___Angry_0000.avi
    "text": "video_name", # 000149120___Angry_0000.avi
    "video": "video_name", # 004713680___Angry_0000.avi
    "video_static": "video_name" # 004713680___Angry_0000.avi
}

MODALITY_PREDICTS_KEYS = {
    "audio": "predictions", # predictions -> emo [0.11277925968170166, 0.11209339648485184, 0.11213059723377228, 0.11190540343523026, 0.11249196529388428, 0.11250203102827072, 0.11201754212379456, 0.2140798419713974] audio=audio_8
    "audio_8": "predictions", # predictions -> emo [0.11277925968170166, 0.11209339648485184, 0.11213059723377228, 0.11190540343523026, 0.11249196529388428, 0.11250203102827072, 0.11201754212379456, 0.2140798419713974]
    "audio_25": "predictions", # predictions -> emo [0.11277925968170166, 0.11209339648485184, 0.11213059723377228, 0.11190540343523026, 0.11249196529388428, 0.11250203102827072, 0.11201754212379456, 0.2140798419713974]
    "clip": "basic_predictions", # numpy.ndarray [8]
    "new_scene": "probs", # list [8]
    "scene": "probs", # list [8]
    "text": "predictions", # list [8]
    "video": "dynamic_predictions", # numpy.ndarray [8]
    "video_static": "static_predictions" # numpy.ndarray [8]
}


class MultimodalEmotionDataset(Dataset):
    """
    Dataset that fuses multiple modality features for emotion classification.
    """

    def __init__(
        self,
        csv_path: str,
        features_dir: str,
        db: str,
        subset: str,
        emotion_labels: list[str],
        modalities: list[str],
    ) -> None:
        self.df = pd.read_csv(csv_path)
        if "test" not in csv_path:
            self.df = self.df.dropna(subset=emotion_labels, how='all')
        
        self.features_dir = features_dir
        self.db = db
        self.subset = subset
        self.emotion_labels = emotion_labels
        self.modalities = modalities

        self.samples = []
        self.class_counts = None
        self.load_all_modalities()

    def _convert_features_predicts(self, feature: any) -> torch.Tensor:
        """
        Normalize feature into a torch.Tensor regardless of its original type.
        """
        if isinstance(feature, torch.Tensor):
            return feature.float()
        elif isinstance(feature, np.ndarray):
            return torch.from_numpy(feature).float()
        elif isinstance(feature, list):
            return torch.tensor(feature, dtype=torch.float)
        if isinstance(feature, dict):
            return torch.tensor(feature['emo'], dtype=torch.float)
        else:
            raise TypeError(f"Unsupported feature type: {type(feature)}")
        
    def load_all_modalities(self) -> None:
        """
        Load and fuse samples from multiple modalities.
        """
        indexed = defaultdict(dict)
        for modality in self.modalities:
            fname = f"{self.subset}_{self.db}_feats.pkl"
            path = os.path.join(self.features_dir, f"{modality}_predicts", fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file for modality '{modality}': {path}")
            
            with open(path, 'rb') as f:
                data = pickle.load(f)

            if 'audio' in modality:
                for i, sample in enumerate(data):
                    sample_name = sample['metas']['audio_name'].split('.')[0]
                    targets = self._convert_features_predicts(sample['targets'])
                    indexed[sample_name][modality] = {
                        'features': self._convert_features_predicts(sample[MODALITY_FEATURES_KEYS[modality]]),
                        'predicts': self._convert_features_predicts(sample[MODALITY_PREDICTS_KEYS[modality]]),
                        'targets': targets,
                        'meta': {
                            'db': self.db,
                            'has_speech': sample['metas']['has_speech'],
                            'file_name': sample_name,
                        }
                    }
            else:
                for i, sample in enumerate(data):
                    sample_name = sample[MODALITY_FILENAME_KEYS[modality]].split('.')[0]
                    indexed[sample_name][modality] = {
                        'features': self._convert_features_predicts(sample[MODALITY_FEATURES_KEYS[modality]]),
                        'predicts': self._convert_features_predicts(sample[MODALITY_PREDICTS_KEYS[modality]]),
                        'meta': {
                            'db': self.db,
                            'has_speech': None,
                            'file_name': sample_name,
                        }
                    }

        for _, row in self.df.iterrows():
            sample_name = row['audio_name'].split('.')[0]

            if sample_name not in indexed:
                continue

            mods_data = indexed[sample_name]
            feature_dict = {m: d['features'] for m, d in mods_data.items() if 'features' in d}
            predicts_dict = {m: d['predicts'] for m, d in mods_data.items() if 'predicts' in d}

            targets = torch.tensor(row[self.emotion_labels].values.astype('float32'))
            has_speech = mods_data['audio']['meta']['has_speech'] if 'audio' in mods_data else None
            meta = {
                'db': self.db, 
                'has_speech': has_speech,
                'file_name': sample_name,
            }

            if self.class_counts is None:
                self.class_counts = targets.clone()
            else:
                self.class_counts += targets

            self.samples.append({
                'features': feature_dict,
                'predicts': predicts_dict,
                'targets': targets,
                'meta': meta
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, any]]:
        sample = self.samples[idx]
        input_tensor = sample['features']
        return input_tensor, {"emo": sample['targets']}, sample['meta']