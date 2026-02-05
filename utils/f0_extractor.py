"""F0 (fundamental frequency) extraction for prosody encoding."""

import numpy as np
import torch
from typing import Optional, Tuple, Literal
import warnings


class F0Extractor:
    """Extract F0 contour from audio for prosody encoding.
    
    Supports multiple extraction methods:
    - DIO: Fast but less accurate
    - Harvest: More accurate but slower
    - CREPE: Neural network based, most accurate
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 320,
        f0_min: float = 50.0,
        f0_max: float = 800.0,
        method: Literal["dio", "harvest", "crepe"] = "dio"
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.method = method
        self.frame_period = hop_length / sample_rate * 1000  # ms
        
        # Try to import pyworld for DIO/Harvest
        try:
            import pyworld as pw
            self.pyworld = pw
        except ImportError:
            self.pyworld = None
            if method in ["dio", "harvest"]:
                warnings.warn(
                    "pyworld not installed. Install with: pip install pyworld"
                )
        
        # Try to import crepe
        try:
            import crepe
            self.crepe = crepe
        except ImportError:
            self.crepe = None
            if method == "crepe":
                warnings.warn(
                    "crepe not installed. Install with: pip install crepe"
                )
    
    def extract(
        self, 
        waveform: np.ndarray,
        return_voiced: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract F0 from waveform.
        
        Args:
            waveform: Audio waveform (1D numpy array)
            return_voiced: Whether to return voiced/unvoiced flags
            
        Returns:
            f0: F0 contour (Hz), unvoiced frames are 0
            voiced: Optional boolean array indicating voiced frames
        """
        waveform = waveform.astype(np.float64)
        
        if self.method == "dio":
            f0, voiced = self._extract_dio(waveform)
        elif self.method == "harvest":
            f0, voiced = self._extract_harvest(waveform)
        elif self.method == "crepe":
            f0, voiced = self._extract_crepe(waveform)
        else:
            raise ValueError(f"Unknown F0 extraction method: {self.method}")
        
        if return_voiced:
            return f0, voiced
        return f0, None
    
    def _extract_dio(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 using DIO algorithm."""
        if self.pyworld is None:
            raise RuntimeError("pyworld is required for DIO extraction")
            
        f0, t = self.pyworld.dio(
            waveform,
            self.sample_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=self.frame_period
        )
        
        # Refine F0 with StoneMask
        f0 = self.pyworld.stonemask(waveform, f0, t, self.sample_rate)
        
        voiced = f0 > 0
        return f0, voiced
    
    def _extract_harvest(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 using Harvest algorithm."""
        if self.pyworld is None:
            raise RuntimeError("pyworld is required for Harvest extraction")
            
        f0, t = self.pyworld.harvest(
            waveform,
            self.sample_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=self.frame_period
        )
        
        voiced = f0 > 0
        return f0, voiced
    
    def _extract_crepe(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 using CREPE neural network."""
        if self.crepe is None:
            raise RuntimeError("crepe is required for CREPE extraction")
            
        # CREPE expects float32
        waveform = waveform.astype(np.float32)
        
        time, frequency, confidence, _ = self.crepe.predict(
            waveform,
            self.sample_rate,
            step_size=self.frame_period,
            viterbi=True,
            model_capacity="full"
        )
        
        # Filter by confidence and frequency range
        voiced = (confidence > 0.5) & (frequency >= self.f0_min) & (frequency <= self.f0_max)
        f0 = np.where(voiced, frequency, 0.0)
        
        return f0, voiced
    
    def normalize_f0(
        self, 
        f0: np.ndarray, 
        method: Literal["log", "zscore", "minmax"] = "log"
    ) -> np.ndarray:
        """Normalize F0 values.
        
        Args:
            f0: Raw F0 values (Hz)
            method: Normalization method
                - log: Log scale (good for pitch perception)
                - zscore: Z-score normalization
                - minmax: Min-max scaling to [0, 1]
        """
        # Handle unvoiced (zero) values
        voiced_mask = f0 > 0
        f0_voiced = f0[voiced_mask]
        
        if len(f0_voiced) == 0:
            return np.zeros_like(f0)
        
        if method == "log":
            # Log normalize, keeping unvoiced as special value
            f0_norm = np.zeros_like(f0)
            f0_norm[voiced_mask] = np.log(f0_voiced + 1e-8)
            # Normalize log values
            log_mean = f0_norm[voiced_mask].mean()
            log_std = f0_norm[voiced_mask].std() + 1e-8
            f0_norm[voiced_mask] = (f0_norm[voiced_mask] - log_mean) / log_std
            
        elif method == "zscore":
            f0_norm = np.zeros_like(f0)
            mean = f0_voiced.mean()
            std = f0_voiced.std() + 1e-8
            f0_norm[voiced_mask] = (f0_voiced - mean) / std
            
        elif method == "minmax":
            f0_norm = np.zeros_like(f0)
            min_val = f0_voiced.min()
            max_val = f0_voiced.max()
            range_val = max_val - min_val + 1e-8
            f0_norm[voiced_mask] = (f0_voiced - min_val) / range_val
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return f0_norm
    
    def interpolate_f0(self, f0: np.ndarray) -> np.ndarray:
        """Interpolate unvoiced regions in F0 contour."""
        voiced_mask = f0 > 0
        
        if not voiced_mask.any():
            return f0
        
        # Get indices
        indices = np.arange(len(f0))
        voiced_indices = indices[voiced_mask]
        voiced_values = f0[voiced_mask]
        
        # Interpolate
        f0_interp = np.interp(indices, voiced_indices, voiced_values)
        
        return f0_interp
    
    def extract_prosody_features(
        self, 
        waveform: np.ndarray
    ) -> dict:
        """Extract comprehensive prosody features.
        
        Returns dict with:
            - f0: Raw F0 contour
            - f0_norm: Normalized F0
            - f0_interp: Interpolated F0
            - voiced: Voiced/unvoiced flags
            - energy: Frame energy
            - duration: Speaking rate proxy
        """
        f0, voiced = self.extract(waveform, return_voiced=True)
        
        # Energy (RMS)
        frame_length = int(self.sample_rate * self.frame_period / 1000)
        num_frames = len(f0)
        energy = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * self.hop_length
            end = start + frame_length
            if end <= len(waveform):
                frame = waveform[start:end]
                energy[i] = np.sqrt(np.mean(frame ** 2))
        
        return {
            "f0": f0,
            "f0_norm": self.normalize_f0(f0, method="log"),
            "f0_interp": self.interpolate_f0(f0),
            "voiced": voiced,
            "energy": energy
        }
    
    def to_tensor(self, f0: np.ndarray) -> torch.Tensor:
        """Convert F0 array to torch tensor."""
        return torch.from_numpy(f0).float()
