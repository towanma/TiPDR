"""Audio processing utilities for Tibetan speech."""

import numpy as np
import torch
import torchaudio
import librosa
from typing import Optional, Tuple


class AudioProcessor:
    """Audio preprocessing for Tibetan speech data."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1280,
        hop_length: int = 320,
        win_length: int = 1280,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = 8000.0
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            norm="slaney",
            mel_scale="slaney"
        )
        
    def load_audio(
        self, 
        path: str, 
        target_sr: Optional[int] = None
    ) -> Tuple[torch.Tensor, int]:
        """Load audio file and resample if needed."""
        waveform, sr = torchaudio.load(path)
        
        if target_sr is None:
            target_sr = self.sample_rate
            
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        return waveform, target_sr
    
    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        return waveform / (waveform.abs().max() + 1e-8)
    
    def extract_mel_spectrogram(
        self, 
        waveform: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """Extract mel spectrogram from waveform."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        mel = self.mel_transform(waveform)
        
        # Log mel spectrogram
        mel = torch.log(mel.clamp(min=1e-5))
        
        if normalize:
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)
            
        return mel.squeeze(0)
    
    def extract_mel_spectrogram_numpy(
        self,
        waveform: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """Extract mel spectrogram using librosa (numpy version)."""
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )
        
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        
        if normalize:
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)
            
        return mel
    
    def trim_silence(
        self, 
        waveform: torch.Tensor, 
        top_db: int = 20
    ) -> torch.Tensor:
        """Trim leading and trailing silence."""
        waveform_np = waveform.numpy().squeeze()
        trimmed, _ = librosa.effects.trim(waveform_np, top_db=top_db)
        return torch.from_numpy(trimmed).unsqueeze(0)
    
    def pad_or_truncate(
        self, 
        waveform: torch.Tensor, 
        target_length: int
    ) -> torch.Tensor:
        """Pad or truncate waveform to target length."""
        current_length = waveform.shape[-1]
        
        if current_length > target_length:
            return waveform[..., :target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            return torch.nn.functional.pad(waveform, (0, padding))
        return waveform
    
    def get_frame_count(self, audio_length: int) -> int:
        """Calculate number of frames for given audio length."""
        return (audio_length - self.win_length) // self.hop_length + 1
