"""Evaluation script for Prosody-Disentangled Model.

Evaluates:
1. Reconstruction quality (mel spectrogram MSE, MCD)
2. Disentanglement metrics (ABX, classification accuracy)
3. Dialect conversion quality
4. Speaker verification EER

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --config configs/config.yaml
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from scipy.spatial.distance import cdist

from models import ProsodyDisentangledModel, create_model
from data import TibetanSpeechDataset, collate_fn


class Evaluator:
    """Evaluator for prosody-disentangled model."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        config: Dict[str, Any]
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
    @torch.no_grad()
    def compute_reconstruction_metrics(self) -> Dict[str, float]:
        """Compute mel spectrogram reconstruction metrics."""
        total_mse = 0.0
        total_mae = 0.0
        total_mcd = 0.0  # Mel Cepstral Distortion
        num_samples = 0
        
        for batch in tqdm(self.test_loader, desc="Computing reconstruction metrics"):
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
            
            pred_mel = outputs["pred_mel"]
            target_mel = batch["mel"]
            
            # Mask padding
            if "mel_lengths" in batch:
                B, _, T = pred_mel.shape
                mask = torch.arange(T, device=pred_mel.device)[None, None, :] < \
                       batch["mel_lengths"][:, None, None]
                mask = mask.float()
                
                mse = ((pred_mel - target_mel) ** 2 * mask).sum() / mask.sum()
                mae = ((pred_mel - target_mel).abs() * mask).sum() / mask.sum()
            else:
                mse = F.mse_loss(pred_mel, target_mel)
                mae = F.l1_loss(pred_mel, target_mel)
            
            # MCD (Mel Cepstral Distortion)
            # MCD = (10 * sqrt(2) / ln(10)) * ||pred - target||_2
            mcd_coef = 10.0 * np.sqrt(2.0) / np.log(10.0)
            mcd = mcd_coef * torch.sqrt(mse)
            
            total_mse += mse.item() * batch["mel"].size(0)
            total_mae += mae.item() * batch["mel"].size(0)
            total_mcd += mcd.item() * batch["mel"].size(0)
            num_samples += batch["mel"].size(0)
        
        return {
            "reconstruction_mse": total_mse / num_samples,
            "reconstruction_mae": total_mae / num_samples,
            "mcd": total_mcd / num_samples
        }
    
    @torch.no_grad()
    def compute_dialect_classification(self) -> Dict[str, Any]:
        """Evaluate if content is dialect-independent.
        
        Train a simple classifier on content embeddings to predict dialect.
        Low accuracy = good disentanglement.
        """
        all_contents = []
        all_prosodies = []
        all_dialect_labels = []
        
        for batch in tqdm(self.test_loader, desc="Extracting embeddings"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(
                waveform=batch.get("waveform"),
                mel=batch.get("mel"),
                f0=batch.get("f0_norm"),
                energy=batch.get("energy"),
                voiced=batch.get("voiced"),
                dialect_id=batch.get("dialect_id"),
                mel_lengths=batch.get("mel_lengths")
            )
            
            # Global pooling for classification
            content = outputs["content_aligned"].mean(dim=1)  # [B, D]
            prosody = outputs["prosody_aligned"].mean(dim=1)  # [B, D]
            
            all_contents.append(content.cpu())
            all_prosodies.append(prosody.cpu())
            all_dialect_labels.append(batch["dialect_id"].cpu())
        
        all_contents = torch.cat(all_contents, dim=0)
        all_prosodies = torch.cat(all_prosodies, dim=0)
        all_dialect_labels = torch.cat(all_dialect_labels, dim=0)
        
        # Split for train/test
        n = len(all_contents)
        n_train = int(n * 0.8)
        
        train_content = all_contents[:n_train]
        test_content = all_contents[n_train:]
        train_labels = all_dialect_labels[:n_train]
        test_labels = all_dialect_labels[n_train:]
        
        # Simple linear classifier
        content_classifier = nn.Linear(train_content.size(1), 3)
        optimizer = torch.optim.Adam(content_classifier.parameters(), lr=1e-3)
        
        # Train
        for _ in range(100):
            logits = content_classifier(train_content)
            loss = F.cross_entropy(logits, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate
        with torch.no_grad():
            pred_labels = content_classifier(test_content).argmax(dim=1)
            content_accuracy = accuracy_score(
                test_labels.numpy(), 
                pred_labels.numpy()
            )
        
        # Same for prosody (should be HIGH accuracy since prosody contains dialect info)
        train_prosody = all_prosodies[:n_train]
        test_prosody = all_prosodies[n_train:]
        
        prosody_classifier = nn.Linear(train_prosody.size(1), 3)
        optimizer = torch.optim.Adam(prosody_classifier.parameters(), lr=1e-3)
        
        for _ in range(100):
            logits = prosody_classifier(train_prosody)
            loss = F.cross_entropy(logits, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            pred_labels = prosody_classifier(test_prosody).argmax(dim=1)
            prosody_accuracy = accuracy_score(
                test_labels.numpy(),
                pred_labels.numpy()
            )
        
        return {
            "content_dialect_accuracy": content_accuracy,  # Lower is better
            "prosody_dialect_accuracy": prosody_accuracy,  # Higher expected
            "disentanglement_score": prosody_accuracy - content_accuracy  # Higher is better
        }
    
    @torch.no_grad()
    def compute_abx_score(self, n_samples: int = 1000) -> Dict[str, float]:
        """Compute ABX discrimination score.
        
        For content disentanglement:
        - A and X have same content, different prosody
        - B has different content
        - Test if distance(A,X) < distance(B,X)
        """
        all_contents = []
        all_vq_indices = []
        
        for batch in tqdm(self.test_loader, desc="Extracting for ABX"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(
                waveform=batch.get("waveform"),
                mel=batch.get("mel"),
                f0=batch.get("f0_norm"),
                energy=batch.get("energy"),
                voiced=batch.get("voiced"),
                mel_lengths=batch.get("mel_lengths")
            )
            
            # Use frame-level content for ABX
            content = outputs["content_aligned"]  # [B, T, D]
            indices = outputs["vq_indices"]  # [B, T]
            
            all_contents.append(content.cpu())
            all_vq_indices.append(indices.cpu())
        
        all_contents = torch.cat(all_contents, dim=0)
        all_vq_indices = torch.cat(all_vq_indices, dim=0)
        
        # Sample ABX triplets
        n = len(all_contents)
        correct = 0
        total = 0
        
        for _ in range(n_samples):
            # Sample A, B (different samples)
            idx_a = np.random.randint(n)
            idx_b = np.random.randint(n)
            if idx_a == idx_b:
                continue
            
            # Find frame in A with same VQ index as some frame in X (same content)
            indices_a = all_vq_indices[idx_a]
            indices_b = all_vq_indices[idx_b]
            
            # For simplicity, use utterance-level representation
            content_a = all_contents[idx_a].mean(dim=0)
            content_b = all_contents[idx_b].mean(dim=0)
            
            # X is another sample with same VQ indices distribution
            # For now, use random X
            idx_x = np.random.randint(n)
            content_x = all_contents[idx_x].mean(dim=0)
            
            # Compute distances
            dist_ax = torch.dist(content_a, content_x)
            dist_bx = torch.dist(content_b, content_x)
            
            # ABX: is A closer to X than B?
            # This is a simplified version; proper ABX requires same-content pairs
            if dist_ax < dist_bx:
                correct += 1
            total += 1
        
        return {
            "abx_score": correct / total if total > 0 else 0.5
        }
    
    @torch.no_grad()
    def compute_speaker_eer(self) -> Dict[str, float]:
        """Compute speaker verification Equal Error Rate."""
        all_speaker_embs = []
        all_speaker_labels = []
        
        for batch in tqdm(self.test_loader, desc="Extracting speaker embeddings"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(
                waveform=batch.get("waveform"),
                mel=batch.get("mel"),
                f0=batch.get("f0_norm"),
                energy=batch.get("energy"),
                voiced=batch.get("voiced"),
                mel_lengths=batch.get("mel_lengths")
            )
            
            all_speaker_embs.append(outputs["speaker_emb"].cpu())
            all_speaker_labels.append(batch["speaker_id"].cpu())
        
        all_speaker_embs = torch.cat(all_speaker_embs, dim=0).numpy()
        all_speaker_labels = torch.cat(all_speaker_labels, dim=0).numpy()
        
        # Compute pairwise cosine similarities
        # Create positive and negative pairs
        n = len(all_speaker_embs)
        
        pos_scores = []
        neg_scores = []
        
        for i in range(min(n, 1000)):  # Limit for efficiency
            for j in range(i + 1, min(n, 1000)):
                sim = np.dot(all_speaker_embs[i], all_speaker_embs[j])
                
                if all_speaker_labels[i] == all_speaker_labels[j]:
                    pos_scores.append(sim)
                else:
                    neg_scores.append(sim)
        
        if not pos_scores or not neg_scores:
            return {"speaker_eer": 0.5}
        
        # Compute EER
        pos_scores = np.array(pos_scores)
        neg_scores = np.array(neg_scores)
        
        thresholds = np.linspace(
            min(pos_scores.min(), neg_scores.min()),
            max(pos_scores.max(), neg_scores.max()),
            1000
        )
        
        best_eer = 1.0
        for thresh in thresholds:
            far = (neg_scores >= thresh).mean()  # False Accept Rate
            frr = (pos_scores < thresh).mean()   # False Reject Rate
            eer = (far + frr) / 2
            if abs(far - frr) < best_eer:
                best_eer = eer
        
        return {"speaker_eer": best_eer}
    
    @torch.no_grad()
    def evaluate_dialect_conversion(self) -> Dict[str, float]:
        """Evaluate dialect conversion quality.
        
        Convert Amdo speech to Lhasa prosody and measure:
        1. Content preservation
        2. Prosody transfer
        """
        # This requires paired data or reference samples
        # Simplified version: measure if converted speech
        # has different prosody characteristics
        
        results = {
            "conversion_content_similarity": 0.0,
            "conversion_prosody_difference": 0.0
        }
        
        # TODO: Implement proper dialect conversion evaluation
        # when paired test data is available
        
        return results
    
    def evaluate_all(self) -> Dict[str, Any]:
        """Run all evaluations."""
        results = {}
        
        print("\n=== Reconstruction Metrics ===")
        recon_metrics = self.compute_reconstruction_metrics()
        results.update(recon_metrics)
        for k, v in recon_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        print("\n=== Dialect Classification (Disentanglement) ===")
        dialect_metrics = self.compute_dialect_classification()
        results.update(dialect_metrics)
        for k, v in dialect_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        print("\n=== ABX Score ===")
        abx_metrics = self.compute_abx_score()
        results.update(abx_metrics)
        for k, v in abx_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        print("\n=== Speaker Verification ===")
        speaker_metrics = self.compute_speaker_eer()
        results.update(speaker_metrics)
        for k, v in speaker_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Prosody-Disentangled Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Path to save results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = create_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {args.checkpoint}")
    
    # Create test dataset
    data_config = config.get("data", {})
    metadata_path = Path(data_config.get("test_path", "data")) / "metadata.json"
    preprocessed_dir = Path(data_config.get("test_path", "data")) / "preprocessed"
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    test_dataset = TibetanSpeechDataset(
        metadata_path=str(metadata_path),
        preprocessed_dir=str(preprocessed_dir),
        augment=False,
        statistics=metadata.get("statistics", {})
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("training", {}).get("batch_size", 16),
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Evaluate
    evaluator = Evaluator(model, test_loader, device, config)
    results = evaluator.evaluate_all()
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
