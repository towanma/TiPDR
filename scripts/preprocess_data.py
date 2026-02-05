#!/usr/bin/env python
"""Preprocess raw Tibetan speech data for training.

Usage:
    python scripts/preprocess_data.py \
        --data_root /path/to/raw/data \
        --output_dir /path/to/output \
        --num_workers 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessor import DataPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess Tibetan speech data")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory containing dialect subdirectories")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for preprocessed data")
    parser.add_argument("--sample_rate", type=int, default=16000,
                       help="Target sample rate")
    parser.add_argument("--hop_length", type=int, default=320,
                       help="Hop length for feature extraction")
    parser.add_argument("--n_mels", type=int, default=80,
                       help="Number of mel bins")
    parser.add_argument("--f0_method", type=str, default="dio",
                       choices=["dio", "harvest", "crepe"],
                       help="F0 extraction method")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel workers")
    args = parser.parse_args()
    
    print(f"Preprocessing data from {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f0_method=args.f0_method,
        output_dir=args.output_dir + "/preprocessed"
    )
    
    # Preprocess dataset
    metadata = preprocessor.preprocess_dataset(
        data_root=args.data_root,
        metadata_path=args.output_dir + "/metadata.json",
        num_workers=args.num_workers,
        save_features=True
    )
    
    print(f"\nPreprocessing complete!")
    print(f"Total files: {len(metadata['files'])}")
    print(f"Statistics: {metadata['statistics']}")
    
    # Split dataset
    train_files, val_files, test_files = preprocessor.split_dataset(
        metadata,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        by_speaker=True
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_files)}")
    print(f"  Val: {len(val_files)}")
    print(f"  Test: {len(test_files)}")
    
    # Save splits
    import json
    with open(args.output_dir + "/splits.json", "w") as f:
        json.dump({
            "train": train_files,
            "val": val_files,
            "test": test_files
        }, f, indent=2)
    
    print(f"\nSplits saved to {args.output_dir}/splits.json")


if __name__ == "__main__":
    main()
