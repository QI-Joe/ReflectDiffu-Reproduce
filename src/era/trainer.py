"""
ERA Trainer Module

Implements complete training pipeline with:
- AdamW optimizer with different learning rates for encoder/head
- Linear scheduler with warmup
- Early stopping and checkpointing
- TensorBoard logging
- Gradient clipping and accumulation
- Support for both CRF and non-CRF models

Based on NuNER specifications in EAR.md.
"""

import os
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

# Import local modules
from .era_model import ERA_BERT_CRF
from .config import ERAConfig
from .evaluator import ERAEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """
    Tracks training state for checkpointing and resuming.
    """
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    patience_counter: int = 0
    total_train_loss: float = 0.0
    learning_rates: Dict[str, float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "total_train_loss": self.total_train_loss,
            "learning_rates": self.learning_rates or {}
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict) -> "TrainingState":
        """Load from dictionary."""
        return cls(
            epoch=state_dict.get("epoch", 0),
            global_step=state_dict.get("global_step", 0),
            best_metric=state_dict.get("best_metric", 0.0),
            best_epoch=state_dict.get("best_epoch", 0),
            patience_counter=state_dict.get("patience_counter", 0),
            total_train_loss=state_dict.get("total_train_loss", 0.0),
            learning_rates=state_dict.get("learning_rates", {})
        )


class ERATrainer:
    """
    ERA model trainer with advanced features.
    
    Features:
    - Different learning rates for encoder and head (NuNER style)
    - Gradient accumulation for large effective batch sizes
    - Early stopping with patience
    - Automatic checkpointing and resuming
    - TensorBoard logging
    - Memory optimization techniques
    """
    
    def __init__(
        self,
        model: ERA_BERT_CRF,
        config: ERAConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        evaluator: Optional[ERAEvaluator] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: ERA model to train
            config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            test_dataloader: Test data loader (optional)
            evaluator: Model evaluator instance
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.evaluator = evaluator or ERAEvaluator(config)
        
        # Set device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)
        
        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.logging_dir = Path(config.logging_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training state
        self.state = TrainingState()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup logging
        self.writer = self._setup_tensorboard()
        
        # Training metrics
        self.train_losses = []
        self.eval_metrics = []
        
        logger.info("ERA Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Logging dir: {self.logging_dir}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        if device.type == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        else:
            logger.info("Using CPU")
        
        return device
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Setup AdamW optimizer with different learning rates.
        
        Following NuNER approach:
        - Lower learning rate for BERT encoder
        - Higher learning rate for classification head
        """
        # Separate parameters for different learning rates
        encoder_params = []
        head_params = []
        
        # BERT encoder parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bert." in name:
                    encoder_params.append(param)
                else:
                    head_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {
                "params": encoder_params,
                "lr": self.config.encoder_lr,
                "weight_decay": self.config.weight_decay
            },
            {
                "params": head_params,
                "lr": self.config.head_lr,
                "weight_decay": self.config.weight_decay
            }
        ]
        
        # Create optimizer
        optimizer = AdamW(
            param_groups,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )
        
        logger.info(f"Optimizer setup:")
        logger.info(f"  Encoder LR: {self.config.encoder_lr}")
        logger.info(f"  Head LR: {self.config.head_lr}")
        logger.info(f"  Weight decay: {self.config.weight_decay}")
        logger.info(f"  Encoder params: {len(encoder_params)}")
        logger.info(f"  Head params: {len(head_params)}")
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        # Calculate total training steps
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        total_steps = total_steps // self.config.gradient_accumulation_steps
        
        # Calculate warmup steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        # Create scheduler
        if self.config.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif self.config.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            scheduler = None
        
        logger.info(f"Scheduler setup:")
        logger.info(f"  Type: {self.config.scheduler_type}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        
        return scheduler
    
    def _setup_tensorboard(self) -> SummaryWriter:
        """Setup TensorBoard logging."""
        log_dir = self.logging_dir / "tensorboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        writer = SummaryWriter(log_dir=str(log_dir))
        
        # Log configuration
        config_text = json.dumps(self.config.to_dict(), indent=2)
        writer.add_text("Config", config_text, 0)
        
        logger.info(f"TensorBoard logging to: {log_dir}")
        
        return writer
    
    def train(self) -> Dict[str, float]:
        """
        Run complete training loop.
        
        Returns:
            Dictionary with final training metrics
        """
        logger.info("Starting ERA training...")
        logger.info(f"  Epochs: {self.config.num_epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"  Total steps: {len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps}")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.state.epoch = epoch
            
            # Train for one epoch
            train_loss = self._train_epoch()
            
            # Evaluate on validation set
            eval_metrics = self._evaluate()
            
            # Log metrics
            self._log_epoch_metrics(train_loss, eval_metrics)
            
            # Save checkpoint
            self._save_checkpoint()
            
            # Check early stopping
            if self._should_stop_early(eval_metrics):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation on test set
        if self.test_dataloader is not None:
            logger.info("Running final evaluation on test set...")
            test_metrics = self._evaluate(self.test_dataloader)
            self._log_test_metrics(test_metrics)
        
        # Load best model
        self._load_best_checkpoint()
        
        logger.info("Training completed!")
        
        return {
            "best_epoch": self.state.best_epoch,
            "best_metric": self.state.best_metric,
            "final_train_loss": self.state.total_train_loss
        }
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Initialize gradient accumulation
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs["loss"]
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.state.global_step += 1
                
                # Log training metrics
                if self.state.global_step % self.config.logging_steps == 0:
                    self._log_training_step(loss.item() * self.config.gradient_accumulation_steps)
        
        # Handle remaining gradients
        if len(self.train_dataloader) % self.config.gradient_accumulation_steps != 0:
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.
        
        Args:
            dataloader: Data loader to evaluate on (default: eval_dataloader)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if dataloader is None:
            dataloader = self.eval_dataloader
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                if "loss" in outputs:
                    total_loss += outputs["loss"].item()
                
                # Collect predictions and labels
                predictions = outputs["predictions"].cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                num_batches += 1
        
        # Compute metrics using evaluator
        metrics = self.evaluator.compute_metrics(all_predictions, all_labels)
        
        if num_batches > 0:
            metrics["eval_loss"] = total_loss / num_batches
        
        return metrics
    
    def _log_training_step(self, loss: float):
        """Log training step metrics."""
        self.writer.add_scalar("train/loss", loss, self.state.global_step)
        
        # Log learning rates
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr_name = "encoder" if i == 0 else "head"
            self.writer.add_scalar(f"train/lr_{lr_name}", param_group["lr"], self.state.global_step)
    
    def _log_epoch_metrics(self, train_loss: float, eval_metrics: Dict[str, float]):
        """Log epoch-level metrics."""
        # Training metrics
        self.writer.add_scalar("epoch/train_loss", train_loss, self.state.epoch)
        
        # Evaluation metrics
        for metric_name, value in eval_metrics.items():
            self.writer.add_scalar(f"epoch/{metric_name}", value, self.state.epoch)
        
        # Log to console
        logger.info(f"Epoch {self.state.epoch}:")
        logger.info(f"  Train loss: {train_loss:.4f}")
        for metric_name, value in eval_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Store for history
        self.train_losses.append(train_loss)
        self.eval_metrics.append(eval_metrics)
    
    def _log_test_metrics(self, test_metrics: Dict[str, float]):
        """Log test metrics."""
        logger.info("Test Results:")
        for metric_name, value in test_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
            self.writer.add_scalar(f"test/{metric_name}", value, 0)
    
    def _should_stop_early(self, eval_metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop early.
        
        Args:
            eval_metrics: Current evaluation metrics
            
        Returns:
            True if training should stop
        """
        current_metric = eval_metrics.get(self.config.metric_for_best_model, 0.0)
        
        # Check if this is the best metric so far
        is_better = False
        if self.config.greater_is_better:
            is_better = current_metric > self.state.best_metric
        else:
            is_better = current_metric < self.state.best_metric
        
        if is_better:
            self.state.best_metric = current_metric
            self.state.best_epoch = self.state.epoch
            self.state.patience_counter = 0
            
            # Save best model
            self._save_best_checkpoint()
            logger.info(f"New best {self.config.metric_for_best_model}: {current_metric:.4f}")
        else:
            self.state.patience_counter += 1
            logger.info(f"No improvement for {self.state.patience_counter} epochs")
        
        # Check early stopping
        return self.state.patience_counter >= self.config.early_stopping_patience
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{self.state.epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        state_file = checkpoint_dir / "training_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
        
        # Save optimizer and scheduler state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.state.epoch,
            "global_step": self.state.global_step
        }, checkpoint_dir / "optimizer.pt")
        
        logger.debug(f"Checkpoint saved: {checkpoint_dir}")
    
    def _save_best_checkpoint(self):
        """Save best model checkpoint."""
        best_dir = self.output_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(best_dir)
        
        # Save training state
        state_file = best_dir / "training_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
        
        logger.debug(f"Best model saved: {best_dir}")
    
    def _load_best_checkpoint(self):
        """Load best model checkpoint."""
        best_dir = self.output_dir / "best_model"
        
        if best_dir.exists():
            logger.info(f"Loading best model from {best_dir}")
            self.model = ERA_BERT_CRF.from_pretrained(best_dir)
            self.model = self.model.to(self.device)
        else:
            logger.warning("Best model checkpoint not found")
    
    def resume_training(self, checkpoint_dir: str):
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_dir)
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_path}")
        
        logger.info(f"Resuming training from {checkpoint_path}")
        
        # Load model
        self.model = ERA_BERT_CRF.from_pretrained(checkpoint_path)
        self.model = self.model.to(self.device)
        
        # Load training state
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state_dict = json.load(f)
            self.state = TrainingState.from_dict(state_dict)
        
        # Load optimizer and scheduler state
        optimizer_file = checkpoint_path / "optimizer.pt"
        if optimizer_file.exists():
            checkpoint = torch.load(optimizer_file, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler and checkpoint["scheduler"]:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        logger.info(f"Resumed from epoch {self.state.epoch}, step {self.state.global_step}")
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'writer'):
            self.writer.close()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Utility functions
def create_trainer_from_config(
    model: ERA_BERT_CRF,
    config: ERAConfig,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    test_dataloader: Optional[DataLoader] = None
) -> ERATrainer:
    """
    Factory function to create trainer from configuration.
    
    Args:
        model: ERA model
        config: Training configuration
        train_dataloader: Training data
        eval_dataloader: Evaluation data
        test_dataloader: Test data (optional)
        
    Returns:
        Configured ERATrainer
    """
    trainer = ERATrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        test_dataloader=test_dataloader
    )
    
    return trainer


# Example usage
if __name__ == "__main__":
    # This would typically be called from main training script
    print("ERATrainer module loaded successfully")