"""Data preprocessing pipeline for Tibetan speech data."""

import os
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
import librosa
import soundfile as sf

from utils.audio import AudioProcessor
from utils.f0_extractor import F0Extractor


class DataPreprocessor:
    """Preprocess raw audio data for training.
    
    Extracts and caches:
    - Mel spectrograms
    - F0 contours
    - Energy features
    - Speaker statistics for normalization
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 320,
        n_mels: int = 80,
        f0_method: str = "dio",
        output_dir: str = "preprocessed"
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f0_method = f0_method
        self.output_dir = Path(output_dir)
        
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.f0_extractor = F0Extractor(
            sample_rate=sample_rate,
            hop_length=hop_length,
            method=f0_method
        )
        
    def preprocess_file(
        self, 
        audio_path: str, 
        output_path: Optional[str] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """Preprocess a single audio file."""
        try:
            # Load audio using soundfile (more reliable)
            waveform, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)
            
            # Resample if needed
            if sr != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
            
            # Normalize
            waveform = waveform.astype(np.float32)
            waveform = waveform / (np.abs(waveform).max() + 1e-8)
            
            # Extract mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_fft=1280,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=0,
                fmax=8000
            )
            mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
            
            # Extract prosody features
            prosody = self.f0_extractor.extract_prosody_features(waveform)
            
            features = {
                "mel": mel.astype(np.float32),
                "f0": prosody["f0"].astype(np.float32),
                "f0_norm": prosody["f0_norm"].astype(np.float32),
                "f0_interp": prosody["f0_interp"].astype(np.float32),
                "voiced": prosody["voiced"],
                "energy": prosody["energy"].astype(np.float32),
                "waveform": waveform
            }
            
            # Save if output path specified
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.savez_compressed(output_path, **features)
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def preprocess_dataset(
        self,
        data_root: str,
        metadata_path: Optional[str] = None,
        num_workers: int = 1,  # Use single worker to avoid file handle issues
        save_features: bool = True
    ) -> Dict[str, Dict]:
        """Preprocess entire dataset."""
        data_root = Path(data_root)
        
        # Collect all audio files
        audio_files = []
        for dialect in ["lhasa", "kham", "amdo"]:
            dialect_dir = data_root / dialect
            if not dialect_dir.exists():
                print(f"Warning: {dialect_dir} does not exist")
                continue
            
            for audio_file in dialect_dir.rglob("*.wav"):
                rel_path = str(audio_file.relative_to(data_root))
                audio_files.append((str(audio_file), rel_path))
        
        print(f"Found {len(audio_files)} audio files")
        
        if len(audio_files) == 0:
            print("No audio files found! Check your data_root path.")
            print(f"Looking in: {data_root}")
            print(f"Expected structure: {data_root}/lhasa/speaker_xxx/*.wav")
            return {"files": {}, "statistics": {}}
        
        # Process files
        metadata = {}
        stats = {
            "mel_mean": [],
            "mel_std": [],
            "f0_mean": [],
            "f0_std": [],
            "energy_mean": [],
            "energy_std": []
        }
        
        if save_features:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process sequentially to avoid file handle issues
        for audio_path, rel_path in tqdm(audio_files, desc="Preprocessing"):
            output_path = None
            if save_features:
                output_path = str(self.output_dir / rel_path.replace(".wav", ".npz"))
            
            features = self.preprocess_file(audio_path, output_path)
            
            if features is None:
                continue
            
            # Extract path components
            parts = rel_path.split(os.sep)
            dialect = parts[0]
            speaker_id = parts[1] if len(parts) > 1 else "unknown"
            
            # Store metadata
            metadata[rel_path] = {
                "speaker_id": speaker_id,
                "dialect": dialect,
                "has_tone": dialect in ["lhasa", "kham"],
                "duration": len(features["waveform"]) / self.sample_rate,
                "num_frames": features["mel"].shape[1]
            }
            
            # Collect statistics
            stats["mel_mean"].append(features["mel"].mean())
            stats["mel_std"].append(features["mel"].std())
            
            voiced_f0 = features["f0"][features["voiced"]]
            if len(voiced_f0) > 0:
                stats["f0_mean"].append(voiced_f0.mean())
                stats["f0_std"].append(voiced_f0.std())
            
            stats["energy_mean"].append(features["energy"].mean())
            stats["energy_std"].append(features["energy"].std())
            
            # Clean up memory periodically
            del features
            gc.collect()
        
        # Compute global statistics
        global_stats = {
            "mel_mean": float(np.mean(stats["mel_mean"])) if stats["mel_mean"] else 0.0,
            "mel_std": float(np.mean(stats["mel_std"])) if stats["mel_std"] else 1.0,
            "f0_mean": float(np.mean(stats["f0_mean"])) if stats["f0_mean"] else 0.0,
            "f0_std": float(np.mean(stats["f0_std"])) if stats["f0_std"] else 1.0,
            "energy_mean": float(np.mean(stats["energy_mean"])) if stats["energy_mean"] else 0.0,
            "energy_std": float(np.mean(stats["energy_std"])) if stats["energy_std"] else 1.0
        }
        
        # Save metadata
        full_metadata = {
            "files": metadata,
            "statistics": global_stats,
            "config": {
                "sample_rate": self.sample_rate,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "f0_method": self.f0_method
            }
        }
        
        if metadata_path:
            os.makedirs(os.path.dirname(metadata_path) if os.path.dirname(metadata_path) else ".", exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(full_metadata, f, indent=2, ensure_ascii=False)
            print(f"Metadata saved to {metadata_path}")
        
        return full_metadata
    
    def split_dataset(
        self,
        metadata: Dict,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        by_speaker: bool = True
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split dataset into train/val/test sets."""
        files = list(metadata.get("files", {}).keys())
        
        if len(files) == 0:
            return [], [], []
        
        if by_speaker:
            # Group by speaker
            speaker_files = {}
            for f in files:
                spk = metadata["files"][f]["speaker_id"]
                if spk not in speaker_files:
                    speaker_files[spk] = []
                speaker_files[spk].append(f)
            
            # Split speakers
            speakers = list(speaker_files.keys())
            np.random.shuffle(speakers)
            
            n_train = int(len(speakers) * train_ratio)
            n_val = int(len(speakers) * val_ratio)
            
            train_speakers = speakers[:n_train]
            val_speakers = speakers[n_train:n_train + n_val]
            test_speakers = speakers[n_train + n_val:]
            
            train_files = [f for spk in train_speakers for f in speaker_files[spk]]
            val_files = [f for spk in val_speakers for f in speaker_files[spk]]
            test_files = [f for spk in test_speakers for f in speaker_files[spk]]
        else:
            # Random split
            np.random.shuffle(files)
            
            n_train = int(len(files) * train_ratio)
            n_val = int(len(files) * val_ratio)
            
            train_files = files[:n_train]
            val_files = files[n_train:n_train + n_val]
            test_files = files[n_train + n_val:]
        
        return train_files, val_files, test_files
