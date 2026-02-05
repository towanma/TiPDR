"""PyTorch Dataset for Tibetan speech data."""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TibetanSpeechDataset(Dataset):
    """Dataset for prosody-disentangled Tibetan speech model.
    
    Returns:
        - mel: Mel spectrogram [n_mels, T]
        - f0: F0 contour [T]
        - f0_norm: Normalized F0 [T]
        - energy: Energy contour [T]
        - waveform: Raw waveform for HuBERT input
        - speaker_id: Speaker index
        - dialect_id: Dialect index (0=Lhasa, 1=Kham, 2=Amdo)
        - has_tone: Whether dialect has lexical tone
    """
    
    DIALECTS = ["lhasa", "kham", "amdo"]
    
    def __init__(
        self,
        metadata_path: str,
        preprocessed_dir: str,
        file_list: Optional[List[str]] = None,
        max_frames: int = 800,
        min_frames: int = 50,
        augment: bool = False,
        statistics: Optional[Dict] = None
    ):
        """
        Args:
            metadata_path: Path to metadata JSON
            preprocessed_dir: Directory containing preprocessed .npz files
            file_list: List of files to include (for train/val/test split)
            max_frames: Maximum number of mel frames
            min_frames: Minimum number of mel frames
            augment: Whether to apply data augmentation
            statistics: Global statistics for normalization
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.augment = augment
        
        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        self.statistics = statistics or metadata.get("statistics", {})
        
        # Build file list
        all_files = metadata["files"]
        if file_list is not None:
            self.files = [(f, all_files[f]) for f in file_list if f in all_files]
        else:
            self.files = list(all_files.items())
        
        # Filter by frame count
        self.files = [
            (f, info) for f, info in self.files
            if self.min_frames <= info.get("num_frames", self.max_frames) <= self.max_frames
        ]
        
        # Build speaker mapping
        speakers = sorted(set(info["speaker_id"] for _, info in self.files))
        self.speaker_to_idx = {s: i for i, s in enumerate(speakers)}
        self.idx_to_speaker = {i: s for s, i in self.speaker_to_idx.items()}
        
        # Dialect mapping
        self.dialect_to_idx = {d: i for i, d in enumerate(self.DIALECTS)}
        
        print(f"Dataset: {len(self.files)} files, "
              f"{len(self.speaker_to_idx)} speakers")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rel_path, info = self.files[idx]
        
        # Load preprocessed features
        feature_path = self.preprocessed_dir / rel_path.replace(".wav", ".npz")
        features = np.load(feature_path)
        
        mel = features["mel"]
        f0 = features["f0"]
        f0_norm = features["f0_norm"]
        energy = features["energy"]
        waveform = features["waveform"]
        voiced = features["voiced"]
        
        # Normalize mel using global statistics
        if self.statistics:
            mel_mean = self.statistics.get("mel_mean", 0.0)
            mel_std = self.statistics.get("mel_std", 1.0)
            mel = (mel - mel_mean) / (mel_std + 1e-8)
        
        # Data augmentation
        if self.augment:
            mel, f0, f0_norm, energy, voiced = self._augment(
                mel, f0, f0_norm, energy, voiced
            )
        
        # Random crop if too long
        T = mel.shape[1]
        if T > self.max_frames:
            start = random.randint(0, T - self.max_frames)
            mel = mel[:, start:start + self.max_frames]
            f0 = f0[start:start + self.max_frames]
            f0_norm = f0_norm[start:start + self.max_frames]
            energy = energy[start:start + self.max_frames]
            voiced = voiced[start:start + self.max_frames]
            
            # Also crop waveform
            hop_length = 320  # From config
            wav_start = start * hop_length
            wav_end = wav_start + self.max_frames * hop_length
            waveform = waveform[wav_start:wav_end]
        
        # Get speaker and dialect info
        speaker_idx = self.speaker_to_idx[info["speaker_id"]]
        dialect_idx = self.dialect_to_idx[info["dialect"]]
        has_tone = info.get("has_tone", dialect_idx != 2)  # Amdo has no tone
        
        return {
            "mel": torch.from_numpy(mel).float(),
            "f0": torch.from_numpy(f0).float(),
            "f0_norm": torch.from_numpy(f0_norm).float(),
            "energy": torch.from_numpy(energy).float(),
            "voiced": torch.from_numpy(voiced).bool(),
            "waveform": torch.from_numpy(waveform).float(),
            "speaker_id": torch.tensor(speaker_idx, dtype=torch.long),
            "dialect_id": torch.tensor(dialect_idx, dtype=torch.long),
            "has_tone": torch.tensor(has_tone, dtype=torch.bool),
            "file_path": rel_path
        }
    
    def _augment(
        self,
        mel: np.ndarray,
        f0: np.ndarray,
        f0_norm: np.ndarray,
        energy: np.ndarray,
        voiced: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Apply data augmentation."""
        
        # Pitch shift (for tonal dialects, be careful)
        if random.random() < 0.3:
            shift = random.uniform(-2, 2)  # semitones
            factor = 2 ** (shift / 12)
            f0 = f0 * factor
            # Re-normalize
            if voiced.any():
                voiced_f0 = f0[voiced]
                log_f0 = np.log(voiced_f0 + 1e-8)
                log_mean = log_f0.mean()
                log_std = log_f0.std() + 1e-8
                f0_norm = np.zeros_like(f0)
                f0_norm[voiced] = (np.log(f0[voiced] + 1e-8) - log_mean) / log_std
        
        # Energy perturbation
        if random.random() < 0.3:
            energy_scale = random.uniform(0.8, 1.2)
            energy = energy * energy_scale
        
        # Time masking (SpecAugment style)
        if random.random() < 0.3:
            T = mel.shape[1]
            mask_len = random.randint(1, min(20, T // 4))
            mask_start = random.randint(0, T - mask_len)
            mel[:, mask_start:mask_start + mask_len] = 0
        
        # Frequency masking
        if random.random() < 0.3:
            F = mel.shape[0]
            mask_len = random.randint(1, min(10, F // 4))
            mask_start = random.randint(0, F - mask_len)
            mel[mask_start:mask_start + mask_len, :] = 0
        
        return mel, f0, f0_norm, energy, voiced
    
    @property
    def num_speakers(self) -> int:
        return len(self.speaker_to_idx)
    
    @property
    def num_dialects(self) -> int:
        return len(self.DIALECTS)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader with dynamic padding."""
    
    # Get all keys except file_path
    keys = [k for k in batch[0].keys() if k != "file_path"]
    
    collated = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        
        if key in ["speaker_id", "dialect_id", "has_tone"]:
            # Stack scalars
            collated[key] = torch.stack(values)
        elif key == "waveform":
            # Pad waveforms
            max_len = max(v.shape[0] for v in values)
            padded = torch.zeros(len(values), max_len)
            lengths = []
            for i, v in enumerate(values):
                padded[i, :v.shape[0]] = v
                lengths.append(v.shape[0])
            collated[key] = padded
            collated["waveform_lengths"] = torch.tensor(lengths)
        elif key == "mel":
            # Pad mel spectrograms [B, n_mels, T]
            max_T = max(v.shape[1] for v in values)
            n_mels = values[0].shape[0]
            padded = torch.zeros(len(values), n_mels, max_T)
            lengths = []
            for i, v in enumerate(values):
                padded[i, :, :v.shape[1]] = v
                lengths.append(v.shape[1])
            collated[key] = padded
            collated["mel_lengths"] = torch.tensor(lengths)
        else:
            # Pad 1D sequences (f0, energy, etc.)
            max_len = max(v.shape[0] for v in values)
            if values[0].dtype == torch.bool:
                padded = torch.zeros(len(values), max_len, dtype=torch.bool)
            else:
                padded = torch.zeros(len(values), max_len)
            for i, v in enumerate(values):
                padded[i, :v.shape[0]] = v
            collated[key] = padded
    
    # Keep file paths as list
    collated["file_path"] = [item["file_path"] for item in batch]
    
    return collated
