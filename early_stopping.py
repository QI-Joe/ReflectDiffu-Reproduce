import torch
import numpy as np
from typing import Optional, Dict, Any
import logging
import os
from datetime import datetime

# Local time
GLOBAL_TIME = datetime.now().strftime("%m-%d-%H-%M")
logger = logging.getLogger(__name__)

from src.intent_twice.intent_emotion_capture import get_batch_integrator
BATCH_INTEGRATOR = get_batch_integrator()



class EarlyStopping:
    """
    Simple and straightforward early stopping implementation based on loss and BLEU-1 score.
    
    The early stopping decision is made using a combination of:
    1. Loss improvement (lower is better)
    2. BLEU-1 score improvement (higher is better)
    3. Patience mechanism
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        loss_weight: float = 0.7,
        bleu1_weight: float = 0.3,
        mode: str = "auto",  # "auto", "loss", "bleu1"
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            loss_weight: Weight for loss in combined metric (0-1)
            bleu1_weight: Weight for BLEU-1 in combined metric (0-1)
            mode: Monitoring mode - "auto", "loss", "bleu1"
            restore_best_weights: Whether to restore best model weights
            verbose: Whether to print stopping information
        """
        self.patience = patience
        self.min_delta = min_delta
        self.loss_weight = loss_weight
        self.bleu1_weight = bleu1_weight
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # Normalize weights
        total_weight = self.loss_weight + self.bleu1_weight
        if total_weight > 0:
            self.loss_weight /= total_weight
            self.bleu1_weight /= total_weight
        
        # Internal state
        self.best_score = None
        self.best_loss = float('inf')
        self.best_bleu1 = 0.0
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
        
        # History for analysis
        self.history = {
            'epoch': [],
            'loss': [],
            'bleu1': [],
            'combined_score': [],
            'improved': []
        }
        
        if self.verbose:
            print(f"🛑 Early Stopping initialized:")
            print(f"  Mode: {self.mode}")
            print(f"  Patience: {self.patience}")
            print(f"  Min delta: {self.min_delta}")
            if self.mode == "auto":
                print(f"  Loss weight: {self.loss_weight:.2f}")
                print(f"  BLEU-1 weight: {self.bleu1_weight:.2f}")
    
    def __call__(
        self, 
        loss: float, 
        bleu1: float, 
        model: torch.nn.Module, 
        epoch: int
    ) -> bool:
        """
        Check if training should stop early.
        
        Args:
            loss: Current validation loss
            bleu1: Current BLEU-1 score (as percentage, e.g., 25.5)
            model: Model to save weights from
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        # Calculate current score based on mode
        if self.mode == "loss":
            current_score = -loss  # Negative because lower loss is better
        elif self.mode == "bleu1":
            current_score = bleu1  # Higher BLEU-1 is better
        else:  # mode == "auto"
            normalized_loss = -loss / 100.0  # Normalize loss (negative for "higher is better")
            normalized_bleu1 = bleu1 / 100.0  # Normalize BLEU-1 (already "higher is better")
            current_score = (self.loss_weight * normalized_loss + 
                           self.bleu1_weight * normalized_bleu1)
        
        # Track history
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['bleu1'].append(bleu1)
        self.history['combined_score'].append(current_score)
        
        # Check for improvement
        improved = False
        if self.best_score is None or current_score > (self.best_score + self.min_delta):
            self.best_score = current_score
            self.best_loss = loss
            self.best_bleu1 = bleu1
            self.counter = 0
            improved = True
            
            # Save best weights
                        # Save best weights
            if improved:
                store_path = r"output/best_model/"
                if os.path.exists(store_path) is False:
                    os.makedirs(store_path)
                torch.save(model, os.path.join(store_path, f'best_model_{GLOBAL_TIME}_{self.best_bleu1:.2f}.pt'))
            
            if self.verbose:
                print(f"✅ Epoch {epoch}: Improvement detected!")
                print(f"   Loss: {loss:.4f} (best: {self.best_loss:.4f})")
                print(f"   BLEU-1: {bleu1:.2f}% (best: {self.best_bleu1:.2f}%)")
                if self.mode == "auto":
                    print(f"   Combined score: {current_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ Epoch {epoch}: No improvement (patience: {self.counter}/{self.patience})")
                print(f"   Loss: {loss:.4f} (best: {self.best_loss:.4f})")
                print(f"   BLEU-1: {bleu1:.2f}% (best: {self.best_bleu1:.2f}%)")
        
        self.history['improved'].append(improved)
        
        # Check if we should stop
        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"🛑 Early stopping triggered at epoch {epoch}")
                print(f"   Best loss: {self.best_loss:.4f} at epoch {self.get_best_epoch()}")
                print(f"   Best BLEU-1: {self.best_bleu1:.2f}%")
        
        return self.should_stop
    
    def get_best_epoch(self) -> Optional[int]:
        """Get the epoch number with the best score."""
        if not self.history['improved']:
            return None
        
        # Find the last epoch where improvement occurred
        for i in reversed(range(len(self.history['improved']))):
            if self.history['improved'][i]:
                return self.history['epoch'][i]
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of early stopping state."""
        return {
            'should_stop': self.should_stop,
            'best_loss': self.best_loss,
            'best_bleu1': self.best_bleu1,
            'best_epoch': self.get_best_epoch(),
            'patience_counter': self.counter,
            'total_epochs': len(self.history['epoch']),
            'mode': self.mode,
            'patience': self.patience
        }
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot training history (requires matplotlib).
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            epochs = self.history['epoch']
            
            # Loss plot
            axes[0, 0].plot(epochs, self.history['loss'], 'b-', label='Loss')
            axes[0, 0].axhline(y=self.best_loss, color='r', linestyle='--', label=f'Best: {self.best_loss:.4f}')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # BLEU-1 plot
            axes[0, 1].plot(epochs, self.history['bleu1'], 'g-', label='BLEU-1')
            axes[0, 1].axhline(y=self.best_bleu1, color='r', linestyle='--', label=f'Best: {self.best_bleu1:.2f}%')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('BLEU-1 (%)')
            axes[0, 1].set_title('BLEU-1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Combined score plot
            if self.mode == "auto":
                axes[1, 0].plot(epochs, self.history['combined_score'], 'purple', label='Combined Score')
                axes[1, 0].axhline(y=self.best_score, color='r', linestyle='--', label=f'Best: {self.best_score:.4f}')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Combined Score')
                axes[1, 0].set_title('Combined Score (Loss + BLEU-1)')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Improvement timeline
            improvement_epochs = [e for e, imp in zip(epochs, self.history['improved']) if imp]
            axes[1, 1].scatter(improvement_epochs, [1]*len(improvement_epochs), 
                             color='green', s=50, label='Improvement')
            no_improvement_epochs = [e for e, imp in zip(epochs, self.history['improved']) if not imp]
            axes[1, 1].scatter(no_improvement_epochs, [0]*len(no_improvement_epochs), 
                             color='red', s=30, label='No Improvement')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Improvement')
            axes[1, 1].set_title('Improvement Timeline')
            axes[1, 1].set_ylim(-0.1, 1.1)
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                if self.verbose:
                    print(f"📊 Training history plot saved to: {save_path}")
            
            plt.show()
            
        except ImportError:
            if self.verbose:
                print("⚠️ Matplotlib not available. Cannot plot history.")


def create_early_stopping(
    patience: int = 5,
    min_delta: float = 0.001,
    mode: str = "auto",
    loss_weight: float = 0.7,
    bleu1_weight: float = 0.3,
    **kwargs
) -> EarlyStopping:
    """
    Factory function to create an EarlyStopping instance with common configurations.
    
    Args:
        patience: Number of epochs to wait
        min_delta: Minimum improvement threshold
        mode: "auto" (combined), "loss" (loss only), or "bleu1" (BLEU-1 only)
        loss_weight: Weight for loss in combined mode
        bleu1_weight: Weight for BLEU-1 in combined mode
        **kwargs: Additional arguments for EarlyStopping
        
    Returns:
        Configured EarlyStopping instance
    """
    return EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        loss_weight=loss_weight,
        bleu1_weight=bleu1_weight,
        mode=mode,
        **kwargs
    )
