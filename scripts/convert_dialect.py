#!/usr/bin/env python
"""Dialect conversion script.

Convert speech from one dialect's prosody pattern to another.
Example: Amdo content + Lhasa prosody = Lhasa-accented Amdo speech

Usage:
    python scripts/convert_dialect.py \
        --checkpoint checkpoints/best_model.pt \
        --source_audio amdo_speech.wav \
        --target_prosody_audio lhasa_speech.wav \
        --output converted.wav
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ProsodyDisentangledModel, create_model
from utils.audio import AudioProcessor
from utils.f0_extractor import F0Extractor


def main():
    parser = argparse.ArgumentParser(description="Convert dialect prosody")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--source_audio", type=str, required=True,
                       help="Source audio (content source)")
    parser.add_argument("--target_prosody_audio", type=str, required=True,
                       help="Target audio for prosody")
    parser.add_argument("--speaker_audio", type=str, default=None,
                       help="Speaker reference audio (optional, defaults to source)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output audio path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    args = parser.parse_args()
    
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = create_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {args.checkpoint}")
    
    # Initialize processors
    audio_processor = AudioProcessor(sample_rate=16000)
    f0_extractor = F0Extractor(sample_rate=16000)
    
    # Load source audio (for content)
    source_wav, _ = audio_processor.load_audio(args.source_audio)
    source_wav = source_wav.squeeze(0)
    
    # Load target audio (for prosody)
    target_wav, _ = audio_processor.load_audio(args.target_prosody_audio)
    target_wav = target_wav.squeeze(0)
    
    # Load speaker reference (optional)
    if args.speaker_audio:
        speaker_wav, _ = audio_processor.load_audio(args.speaker_audio)
        speaker_wav = speaker_wav.squeeze(0)
    else:
        speaker_wav = source_wav
    
    # Extract features
    source_mel = audio_processor.extract_mel_spectrogram(source_wav.unsqueeze(0))
    target_mel = audio_processor.extract_mel_spectrogram(target_wav.unsqueeze(0))
    speaker_mel = audio_processor.extract_mel_spectrogram(speaker_wav.unsqueeze(0))
    
    # Extract F0 for target prosody
    target_prosody = f0_extractor.extract_prosody_features(target_wav.numpy())
    target_f0 = f0_extractor.to_tensor(target_prosody["f0_norm"])
    target_energy = f0_extractor.to_tensor(target_prosody["energy"])
    
    print(f"Source mel shape: {source_mel.shape}")
    print(f"Target F0 shape: {target_f0.shape}")
    
    # Prepare inputs
    with torch.no_grad():
        # Move to device
        source_mel = source_mel.unsqueeze(0).to(device)  # [1, n_mels, T]
        target_f0 = target_f0.unsqueeze(0).to(device)    # [1, T]
        target_energy = target_energy.unsqueeze(0).to(device)
        speaker_mel = speaker_mel.unsqueeze(0).to(device)
        
        # Encode content from source
        content, _ = model.encode_content(mel=source_mel)
        
        # Encode prosody from target
        prosody = model.encode_prosody(target_f0, target_energy)
        
        # Encode speaker
        speaker_emb = model.encode_speaker(speaker_mel)
        
        # Align sequences
        content, prosody = model._align_sequences(content, prosody)
        
        # Decode
        converted_mel = model.decoder.inference(content, prosody, speaker_emb)
        
        print(f"Converted mel shape: {converted_mel.shape}")
    
    # Convert mel to audio (requires vocoder)
    print("\nNote: To convert mel spectrogram to audio, you need a vocoder.")
    print("Options:")
    print("  1. Use HiFi-GAN: https://github.com/jik876/hifi-gan")
    print("  2. Use Griffin-Lim (lower quality)")
    
    # Save mel spectrogram for now
    converted_mel_np = converted_mel.cpu().squeeze(0).numpy()
    np.save(args.output.replace(".wav", "_mel.npy"), converted_mel_np)
    print(f"\nSaved converted mel spectrogram to {args.output.replace('.wav', '_mel.npy')}")
    
    # Optional: Griffin-Lim reconstruction
    try:
        import librosa
        
        # Denormalize mel (approximate)
        mel_denorm = converted_mel_np * 4.0  # Rough scaling
        
        # Inverse mel
        mel_linear = librosa.db_to_power(mel_denorm)
        
        # Griffin-Lim
        audio = librosa.feature.inverse.mel_to_audio(
            mel_linear,
            sr=16000,
            hop_length=320,
            n_fft=1280
        )
        
        # Save
        import soundfile as sf
        sf.write(args.output, audio, 16000)
        print(f"Saved Griffin-Lim audio to {args.output}")
        print("(Note: Quality may be low. Use HiFi-GAN for better results)")
        
    except Exception as e:
        print(f"Griffin-Lim reconstruction failed: {e}")
        print("Please use a neural vocoder for audio synthesis.")


if __name__ == "__main__":
    main()
