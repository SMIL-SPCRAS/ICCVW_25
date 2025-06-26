# src/audio/utils/silero_vad.py

import torch


class SileroVAD:
    """Voice Activity Detector using Silero pre-trained model from torch.hub."""
    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold
        self.model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
        self.get_speech_timestamps = utils[0]

    def compute_ratio(self, waveform: torch.Tensor, sample_rate: int) -> float:
        """Computes ratio of speech-active frames to total duration."""
        timestamps = self.get_speech_timestamps(audio=waveform, model=self.model, sampling_rate=sample_rate)
        total_speech = sum(ts["end"] - ts["start"] for ts in timestamps)
        total_length = waveform.shape[-1] / sample_rate
        return total_speech / total_length if total_length > 0 else 0.0

    def has_speech(self, waveform: torch.Tensor, sample_rate: int) -> bool:
        """Checks whether audio contains speech based on threshold."""
        return self.compute_ratio(waveform, sample_rate) > self.threshold

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> bool:
        """Shortcut for self.has_speech."""
        return self.has_speech(waveform, sample_rate)
