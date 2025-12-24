"""
ASETrainerV1_1 - Trainer for ASE v1.1 with thermodynamic learning.

Provides:
- Training loop with loss tracking
- Evaluation metrics
- Checkpointing
- Learning rate scheduling
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..neural.model import AtomicSemanticModel
from .losses import ASELossV1_1


class ASETrainerV1_1:
    """
    Trainer for ASE v1.1 with thermodynamic learning.

    Features:
    - Multi-component loss tracking
    - Thermodynamic parameter monitoring
    - Gradient clipping
    - Learning rate scheduling
    - Checkpointing
    """

    def __init__(
        self,
        model: AtomicSemanticModel,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize trainer.

        Args:
            model: ASE model to train
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Warmup steps for scheduler
            max_grad_norm: Max gradient norm for clipping
        """
        self.model = model
        self.max_grad_norm = max_grad_norm

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000,
        )

        # Loss function
        self.loss_fn = ASELossV1_1(
            model.thermo,
            model.catalyst_detector,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

    def train_epoch(
        self,
        dataloader: DataLoader,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            log_interval: Steps between logging

        Returns:
            Dict with average losses
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        num_batches = 0

        for batch in dataloader:
            # Move to device
            token_ids = batch["token_ids"].to(self.device)

            # Forward pass
            output = self.model(token_ids)

            # Build targets
            targets = {}
            if "teacher_embedding" in batch:
                targets["sentence_embedding"] = batch["teacher_embedding"].to(
                    self.device
                )
            if "sentiment" in batch:
                targets["sentiment_label"] = batch["sentiment"].to(self.device)
            if "tokens" in batch:
                targets["tokens"] = batch["tokens"]

            # Compute loss
            losses = self.loss_fn(output, targets)

            # Backward pass
            self.optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

            self.optimizer.step()
            self.scheduler.step()

            # Track losses
            total_loss += losses["total"].item()
            for k, v in losses.items():
                if k != "total" and isinstance(v, torch.Tensor):
                    loss_components[k] = loss_components.get(k, 0) + v.item()

            num_batches += 1
            self.global_step += 1

        # Compute averages
        result = {"total_loss": total_loss / num_batches}
        for k, v in loss_components.items():
            result[k] = v / num_batches

        self.epoch += 1
        return result

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate model.

        Args:
            dataloader: Evaluation data loader

        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()

        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            token_ids = batch["token_ids"].to(self.device)
            output = self.model(token_ids)

            # Sentiment accuracy
            if "sentiment" in batch:
                pred = (output["net_charges"] > 0).long()
                correct += (pred.cpu() == batch["sentiment"]).sum().item()
                total += len(batch["sentiment"])

            num_batches += 1

        result = {
            "thermo_params": {
                "threshold_base": self.model.thermo.threshold_base.item(),
                "ionization_mult": self.model.thermo.ionization_multiplier.item(),
            },
        }

        if total > 0:
            result["charge_accuracy"] = correct / total

        return result

    def save(self, path: str) -> None:
        """
        Save checkpoint.

        Args:
            path: Checkpoint file path
        """
        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "config": self.model.config,
            },
            path,
        )

    def load(self, path: str) -> int:
        """
        Load checkpoint.

        Args:
            path: Checkpoint file path

        Returns:
            Loaded epoch number
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        return self.epoch

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class TrainingLogger:
    """Simple training logger."""

    def __init__(self, log_dir: str = None):
        """
        Initialize logger.

        Args:
            log_dir: Directory for logs
        """
        self.log_dir = log_dir
        self.history = {"train": [], "eval": []}

    def log_train(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log training metrics."""
        self.history["train"].append({"epoch": epoch, **metrics})
        print(f"Epoch {epoch} - Train: {self._format_metrics(metrics)}")

    def log_eval(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics."""
        self.history["eval"].append({"epoch": epoch, **metrics})
        print(f"Epoch {epoch} - Eval: {self._format_metrics(metrics)}")

    def _format_metrics(self, metrics: Dict) -> str:
        """Format metrics for printing."""
        parts = []
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            elif isinstance(v, dict):
                # Nested dict (like thermo_params)
                for kk, vv in v.items():
                    if isinstance(vv, float):
                        parts.append(f"{kk}={vv:.4f}")
        return ", ".join(parts)


def train_ase(
    model: AtomicSemanticModel,
    train_loader: DataLoader,
    eval_loader: DataLoader = None,
    epochs: int = 10,
    lr: float = 1e-4,
    save_path: str = None,
) -> Dict[str, list]:
    """
    Convenience function to train ASE model.

    Args:
        model: ASE model
        train_loader: Training data loader
        eval_loader: Optional evaluation data loader
        epochs: Number of epochs
        lr: Learning rate
        save_path: Path to save best checkpoint

    Returns:
        Training history
    """
    trainer = ASETrainerV1_1(model, lr=lr)
    logger = TrainingLogger()

    best_metric = float("inf")

    for epoch in range(epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        logger.log_train(epoch, train_metrics)

        # Evaluate
        if eval_loader is not None:
            eval_metrics = trainer.evaluate(eval_loader)
            logger.log_eval(epoch, eval_metrics)

            # Save best
            if save_path and train_metrics["total_loss"] < best_metric:
                best_metric = train_metrics["total_loss"]
                trainer.save(save_path)

    return logger.history
