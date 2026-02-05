#!/usr/bin/env python
"""Extract discrete content units for ASR or analysis.

The content encoder produces VQ indices that can be used as
dialect-invariant discrete units for ASR training.

Usage:
    python scripts/extract_content_units.py \
        --checkpoint checkpoints/best_model.pt \
        --audio_dir /path/to/audios \
        --output units.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ProsodyDisentangledModel, create_model
from utils.audio import AudioProcessor


def main():
    parser = argparse.ArgumentParser(description="Extract content units")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = create_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    audio_processor = AudioProcessor(sample_rate=16000)
    
    # Find all audio files
    audio_dir = Path(args.audio_dir)
    audio_files = list(audio_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files")
    
    results = []
    
    with torch.no_grad():
        for audio_path in tqdm(audio_files, desc="Extracting units"):
            try:
                # Load audio
                waveform, sr = audio_processor.load_audio(str(audio_path))
                waveform = waveform.to(device)
                
                # Extract content units
                if model.use_mel_content_encoder:
                    mel = audio_processor.extract_mel_spectrogram(waveform)
                    mel = mel.unsqueeze(0).to(device)
                    _, _, indices = model.content_encoder(mel)
                else:
                    waveform = waveform.squeeze(0).unsqueeze(0)  # [1, T]
                    _, _, indices = model.content_encoder(waveform)
                
                # Convert to list
                units = indices.squeeze(0).cpu().tolist()
                
                results.append({
                    "file": str(audio_path.relative_to(audio_dir)),
                    "units": units,
                    "num_frames": len(units)
                })
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
    
    # Save results
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\nSaved {len(results)} files to {args.output}")
    
    # Statistics
    all_units = [u for r in results for u in r["units"]]
    unique_units = set(all_units)
    print(f"Total units: {len(all_units)}")
    print(f"Unique units: {len(unique_units)}")
    print(f"Codebook utilization: {len(unique_units) / model.vq_num_embeddings * 100:.1f}%")


if __name__ == "__main__":
    main()
