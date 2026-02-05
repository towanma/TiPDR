"""Training script for Prosody-Disentangled Tibetan Speech Model.

Implements multi-stage training:
1. Stage 1: Reconstruction only
2. Stage 2: Add adversarial training
3. Stage 3: Full training with MI minimization

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --resume checkpoints/latest.pt
"""

import os
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import ProsodyDisentangledModel, create_model
from data import TibetanSpeechDataset, collate_fn
from losses import TotalLoss


class Trainer:
    """Trainer for the prosody-disentangled model."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        resume_path: Optional[str] = None
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        training_config = config.get("training", {})
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 1e-6)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=training_config.get("warmup_steps", 10000),
            T_mult=2
        )
        
        # Loss function
        loss_weights = training_config.get("loss_weights", {})
        self.criterion = TotalLoss(
            reconstruction_weight=loss_weights.get("reconstruction", 1.0),
            vq_weight=loss_weights.get("vq_commitment", 0.25),
            adversarial_weight=loss_weights.get("content_adversarial", 0.1),
            mi_weight=loss_weights.get("mi_minimization", 0.05)
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Multi-stage training config
        self.stages = training_config.get("stages", [
            {"epochs": 50, "freeze": ["discriminator"], 
             "loss_weights": {"reconstruction": 1.0, "vq_commitment": 0.25}},
            {"epochs": 100, "freeze": [], 
             "loss_weights": {"reconstruction": 1.0, "vq_commitment": 0.25, 
                             "content_adversarial": 0.1}},
            {"epochs": 50, "freeze": [], 
             "loss_weights": {"reconstruction": 1.0, "vq_commitment": 0.25,
                             "content_adversarial": 0.1, "mi_minimization": 0.05}}
        ])
        self.current_stage = 0
        
        # Paths
        paths = config.get("paths", {})
        self.checkpoint_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
        self.log_dir = Path(paths.get("log_dir", "logs"))
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(
            self.log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Gradient clipping
        self.gradient_clip = training_config.get("gradient_clip", 1.0)
        
        # Resume if specified
        if resume_path:
            self.load_checkpoint(resume_path)
    
    def get_current_stage(self) -> Dict:
        """Get current training stage configuration."""
        total_epochs = 0
        for i, stage in enumerate(self.stages):
            total_epochs += stage["epochs"]
            if self.current_epoch < total_epochs:
                return stage
        return self.stages[-1]
    
    def apply_stage_config(self, stage: Dict):
        """Apply stage-specific configuration."""
        # Update loss weights
        loss_weights = stage.get("loss_weights", {})
        self.criterion.weights.update(loss_weights)
        
        # Freeze/unfreeze components
        freeze_list = stage.get("freeze", [])
        
        # Unfreeze all first
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Then freeze specified
        if "discriminator" in freeze_list:
            for param in self.model.content_discriminator.parameters():
                param.requires_grad = False
            for param in self.model.prosody_discriminator.parameters():
                param.requires_grad = False
            for param in self.model.speaker_discriminator.parameters():
                param.requires_grad = False
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        stage = self.get_current_stage()
        self.apply_stage_config(stage)
        
        epoch_losses = {}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                waveform=batch.get("waveform"),
                mel=batch.get("mel"),
                f0=batch.get("f0_norm"),
                energy=batch.get("energy"),
                voiced=batch.get("voiced"),
                dialect_id=batch.get("dialect_id"),
                speaker_id=batch.get("speaker_id"),
                mel_lengths=batch.get("mel_lengths"),
                waveform_lengths=batch.get("waveform_lengths")
            )
            
            # Compute loss
            total_loss, losses = self.criterion(
                pred_mel=outputs["pred_mel"],
                target_mel=batch["mel"],
                content=outputs["content_aligned"],
                prosody=outputs["prosody_aligned"],
                speaker_emb=outputs["speaker_emb"],
                vq_loss=outputs["vq_loss"],
                vq_indices=outputs["vq_indices"],
                num_embeddings=self.model.vq_num_embeddings,
                content_disc_logits=outputs.get("content_disc_logits"),
                dialect_labels=batch.get("dialect_id"),
                prosody_disc_logits=outputs.get("prosody_disc_logits"),
                speaker_disc_logits=outputs.get("speaker_disc_logits"),
                speaker_labels=batch.get("speaker_id"),
                speaker_clf_logits=outputs.get("speaker_logits"),
                lengths=batch.get("mel_lengths"),
                mi_estimator=self.model.mi_estimator if self.model.use_mi_estimator else None
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "recon": f"{losses['reconstruction_total'].item():.4f}"
            })
            
            # Log to tensorboard
            if self.global_step % 100 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train/{k}", v.item(), self.global_step)
                self.writer.add_scalar(
                    "train/learning_rate", 
                    self.optimizer.param_groups[0]["lr"],
                    self.global_step
                )
            
            self.global_step += 1
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        val_losses = {}
        num_batches = len(self.val_loader)
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(
                waveform=batch.get("waveform"),
                mel=batch.get("mel"),
                f0=batch.get("f0_norm"),
                energy=batch.get("energy"),
                voiced=batch.get("voiced"),
                dialect_id=batch.get("dialect_id"),
                speaker_id=batch.get("speaker_id"),
                mel_lengths=batch.get("mel_lengths")
            )
            
            _, losses = self.criterion(
                pred_mel=outputs["pred_mel"],
                target_mel=batch["mel"],
                content=outputs["content_aligned"],
                prosody=outputs["prosody_aligned"],
                speaker_emb=outputs["speaker_emb"],
                vq_loss=outputs["vq_loss"],
                vq_indices=outputs["vq_indices"],
                num_embeddings=self.model.vq_num_embeddings,
                content_disc_logits=outputs.get("content_disc_logits"),
                dialect_labels=batch.get("dialect_id"),
                lengths=batch.get("mel_lengths")
            )
            
            for k, v in losses.items():
                if k not in val_losses:
                    val_losses[k] = 0.0
                val_losses[k] += v.item()
        
        for k in val_losses:
            val_losses[k] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        print(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
    
    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Training stages: {len(self.stages)}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Get current stage
            stage = self.get_current_stage()
            print(f"\nEpoch {epoch} - Stage: {self.stages.index(stage) + 1}")
            print(f"Loss weights: {stage.get('loss_weights', {})}")
            
            # Train
            train_losses = self.train_epoch()
            
            # Log training losses
            print(f"Train - Total: {train_losses['total_loss']:.4f}, "
                  f"Recon: {train_losses['reconstruction_total']:.4f}")
            
            # Validate (only if val_loader has data)
            eval_config = self.config.get("evaluation", {})
            if (epoch + 1) % eval_config.get("eval_every", 1) == 0 and len(self.val_loader) > 0:
                val_losses = self.validate()
                
                if val_losses and "total_loss" in val_losses:
                    print(f"Val - Total: {val_losses['total_loss']:.4f}, "
                          f"Recon: {val_losses['reconstruction_total']:.4f}")
                    
                    # Log to tensorboard
                    for k, v in val_losses.items():
                        self.writer.add_scalar(f"val/{k}", v, self.current_epoch)
                    
                    # Save best model
                    if val_losses["total_loss"] < self.best_val_loss:
                        self.best_val_loss = val_losses["total_loss"]
                        self.save_checkpoint("best_model.pt")
            elif len(self.val_loader) == 0:
                # No validation data, save based on train loss
                if train_losses["total_loss"] < self.best_val_loss:
                    self.best_val_loss = train_losses["total_loss"]
                    self.save_checkpoint("best_model.pt")
            
            # Save periodic checkpoint
            if (epoch + 1) % eval_config.get("save_every", 5) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Always save latest
            self.save_checkpoint("latest.pt")
        
        print("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Prosody-Disentangled Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    data_config = config.get("data", {})
    
    # Load metadata and splits
    metadata_path = Path(data_config.get("metadata_path", "data/metadata.json"))
    preprocessed_dir = Path(data_config.get("preprocessed_dir", "data/preprocessed"))
    splits_path = Path(data_config.get("splits_path", "data/splits.json"))
    
    if not metadata_path.exists():
        print(f"Error: Metadata not found at {metadata_path}")
        print("Please run preprocessing first:")
        print("  python scripts/preprocess_data.py --data_root <path> --output_dir <path>")
        return
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Load splits
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        train_files = splits.get("train", [])
        val_files = splits.get("val", [])
        print(f"Loaded splits: train={len(train_files)}, val={len(val_files)}")
    else:
        # Fallback: random split
        print(f"Warning: splits.json not found, using random 90/10 split")
        all_files = list(metadata.get("files", {}).keys())
        n_train = int(len(all_files) * 0.9)
        train_files = all_files[:n_train]
        val_files = all_files[n_train:]
    
    # Create datasets
    train_dataset = TibetanSpeechDataset(
        metadata_path=str(metadata_path),
        preprocessed_dir=str(preprocessed_dir),
        file_list=train_files,
        augment=True,
        statistics=metadata.get("statistics", {})
    )
    
    val_dataset = TibetanSpeechDataset(
        metadata_path=str(metadata_path),
        preprocessed_dir=str(preprocessed_dir),
        file_list=val_files,
        augment=False,
        statistics=metadata.get("statistics", {})
    )
    
    # Create dataloaders
    training_config = config.get("training", {})
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get("batch_size", 16),
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get("batch_size", 16),
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = create_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        resume_path=args.resume
    )
    
    # Train
    trainer.train(training_config.get("num_epochs", 200))


if __name__ == "__main__":
    main()
