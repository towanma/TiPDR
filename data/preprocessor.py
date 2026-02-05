"""Data preprocessing pipeline for Tibetan speech data."""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import torch

from ..utils.audio import AudioProcessor
from ..utils.f0_extractor import F0Extractor


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
    ) -> Dict[str, np.ndarray]:
        """Preprocess a single audio file."""
        # Load audio
        waveform, _ = self.audio_processor.load_audio(audio_path)
        waveform = waveform.squeeze(0)
        waveform = self.audio_processor.normalize_audio(waveform)
        waveform_np = waveform.numpy()
        
        # Extract mel spectrogram
        mel = self.audio_processor.extract_mel_spectrogram_numpy(
            waveform_np, 
            normalize=False  # We'll do global normalization later
        )
        
        # Extract prosody features
        prosody = self.f0_extractor.extract_prosody_features(waveform_np)
        
        features = {
            "mel": mel,
            "f0": prosody["f0"],
            "f0_norm": prosody["f0_norm"],
            "f0_interp": prosody["f0_interp"],
            "voiced": prosody["voiced"],
            "energy": prosody["energy"],
            "waveform": waveform_np
        }
        
        # Save if output path specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez_compressed(output_path, **features)
        
        return features
    
    def preprocess_dataset(
        self,
        data_root: str,
        metadata_path: Optional[str] = None,
        num_workers: int = 4,
        save_features: bool = True
    ) -> Dict[str, Dict]:
        """Preprocess entire dataset.
        
        Args:
            data_root: Root directory containing dialect subdirectories
            metadata_path: Path to save/load metadata
            num_workers: Number of parallel workers
            save_features: Whether to save preprocessed features to disk
            
        Returns:
            metadata: Dictionary with file info and statistics
        """
        data_root = Path(data_root)
        
        # Collect all audio files
        audio_files = []
        for dialect in ["lhasa", "kham", "amdo"]:
            dialect_dir = data_root / dialect
            if not dialect_dir.exists():
                continue
            
            for audio_file in dialect_dir.rglob("*.wav"):
                rel_path = str(audio_file.relative_to(data_root))
                audio_files.append((str(audio_file), rel_path))
        
        print(f"Found {len(audio_files)} audio files")
        
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
        
        # Process with progress bar
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            
            for audio_path, rel_path in audio_files:
                output_path = None
                if save_features:
                    output_path = str(
                        self.output_dir / rel_path.replace(".wav", ".npz")
                    )
                
                future = executor.submit(
                    self.preprocess_file,
                    audio_path,
                    output_path
                )
                futures[future] = rel_path
            
            for future in tqdm(
                as_completed(futures), 
                total=len(futures),
                desc="Preprocessing"
            ):
                rel_path = futures[future]
                try:
                    features = future.result()
                    
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
                    
                except Exception as e:
                    print(f"Error processing {rel_path}: {e}")
        
        # Compute global statistics
        global_stats = {
            "mel_mean": float(np.mean(stats["mel_mean"])),
            "mel_std": float(np.mean(stats["mel_std"])),
            "f0_mean": float(np.mean(stats["f0_mean"])) if stats["f0_mean"] else 0.0,
            "f0_std": float(np.mean(stats["f0_std"])) if stats["f0_std"] else 1.0,
            "energy_mean": float(np.mean(stats["energy_mean"])),
            "energy_std": float(np.mean(stats["energy_std"]))
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
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(full_metadata, f, indent=2, ensure_ascii=False)
        
        return full_metadata
    
    def compute_speaker_statistics(
        self,
        data_root: str,
        metadata: Dict
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-speaker statistics for normalization."""
        speaker_stats = {}
        
        for rel_path, info in metadata["files"].items():
            speaker_id = info["speaker_id"]
            
            # Load preprocessed features
            feature_path = self.output_dir / rel_path.replace(".wav", ".npz")
            if not feature_path.exists():
                continue
            
            features = np.load(feature_path)
            
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {
                    "f0_values": [],
                    "energy_values": []
                }
            
            voiced_f0 = features["f0"][features["voiced"]]
            speaker_stats[speaker_id]["f0_values"].extend(voiced_f0.tolist())
            speaker_stats[speaker_id]["energy_values"].extend(
                features["energy"].tolist()
            )
        
        # Compute statistics
        for speaker_id in speaker_stats:
            f0_vals = np.array(speaker_stats[speaker_id]["f0_values"])
            energy_vals = np.array(speaker_stats[speaker_id]["energy_values"])
            
            speaker_stats[speaker_id] = {
                "f0_mean": float(f0_vals.mean()) if len(f0_vals) > 0 else 0.0,
                "f0_std": float(f0_vals.std()) if len(f0_vals) > 0 else 1.0,
                "energy_mean": float(energy_vals.mean()),
                "energy_std": float(energy_vals.std())
            }
        
        return speaker_stats
    
    def split_dataset(
        self,
        metadata: Dict,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        by_speaker: bool = True
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split dataset into train/val/test sets.
        
        Args:
            metadata: Dataset metadata
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            by_speaker: If True, split by speaker to avoid speaker leakage
        """
        files = list(metadata["files"].keys())
        
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
