"""
Configuration System for ERA

Defines hyperparameters from NuNER specifications:
- Frozen layers=6, lr=3e-5, batch_size=48, dropout=0.1, etc.
- Uses IO tagging scheme only: {O: 0, EM: 1}
- Configurable for different model architectures
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
from pathlib import Path


@dataclass
class ERAConfig:
    """
    Configuration class for ERA system based on NuNER specifications.
    
    All hyperparameters are set according to the EAR.md specifications
    and NuNER paper recommendations.
    """
    
    # ==================== Model Configuration ====================
    bert_model: str = "google-bert/bert-base-uncased"  # Base model architecture
    num_labels: int = 2  # IO scheme: O, EM
    max_length: int = 512  # Maximum sequence length
    
    # ==================== Training Configuration (NuNER) ====================
    # Core NuNER hyperparameters from paper
    frozen_layers: int = 6  # Freeze first 6 layers of BERT
    learning_rate: float = 3e-5  # 0.00003
    batch_size: int = 48  # NuNER batch size
    gradient_accumulation_steps: int = 1  # For smaller GPUs, can increase
    
    # Optimizer settings (NuNER specifications)
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    
    # Learning rate scheduler
    scheduler_type: str = "linear"  # Linear schedule with warmup
    warmup_ratio: float = 0.1  # 10% warmup
    
    # Training dynamics
    num_epochs: int = 10  # NuNER epochs
    early_stopping_patience: int = 3  # Early stopping after 3 epochs without improvement
    save_strategy: str = "epoch"  # Save every epoch
    evaluation_strategy: str = "epoch"  # Evaluate every epoch
    
    # Regularization
    dropout_rate: float = 0.1  # Standard BERT dropout
    label_smoothing: float = 0.0  # No label smoothing by default
    
    # ==================== Model Architecture ====================
    hidden_size: int = 768  # RoBERTa-base hidden size
    use_crf: bool = True  # Use CRF layer for sequence consistency
    crf_lr_multiplier: float = 1.0  # CRF learning rate multiplier
    
    # Different learning rates for different components
    encoder_lr: float = 3e-5  # BERT encoder learning rate
    head_lr: float = 5e-5  # Classification head learning rate (higher)
    
    # ==================== Data Configuration ====================
    data_dir: str = "./dataset"
    cache_dir: str = "./cache"
    models_dir: str = "./models"
    
    # Data splitting (ReflectDiffu specification: 8:1:1)
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Data processing
    max_train_samples: Optional[int] = None  # Limit training samples for testing
    max_eval_samples: Optional[int] = None  # Limit evaluation samples
    oversampling_ratio: float = 1.5  # Oversample emotion reason samples
    
    # ==================== Paths and Directories ====================
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    cache_dir: str = "./cache"
    
    # Model paths
    bert_model_path: Optional[str] = None  # Path to local BERT model
    llm_model_path: Optional[str] = None  # Path to local LLM model
    
    # ==================== Hardware Configuration ====================
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    fp16: bool = False  # Use mixed precision training
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # ==================== Logging and Monitoring ====================
    log_level: str = "INFO"
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])  # ["tensorboard", "wandb"]
    logging_steps: int = 50  # Log every N steps
    save_steps: int = 500  # Save checkpoint every N steps
    eval_steps: int = 500  # Evaluate every N steps
    
    # Experiment tracking
    run_name: Optional[str] = None
    experiment_name: str = "era_training"
    notes: str = ""
    
    # ==================== Evaluation Configuration ====================
    metric_for_best_model: str = "eval_f1"  # Metric to use for best model selection
    greater_is_better: bool = True  # Whether higher metric values are better
    
    # Evaluation settings
    eval_batch_size: int = 32  # Can be larger than training batch size
    prediction_loss_only: bool = False  # Also compute metrics during evaluation
    
    # ==================== Reproducibility ====================
    seed: int = 42
    data_seed: int = 42  # Separate seed for data splitting
    
    # ==================== Advanced Settings ====================
    # Gradient settings
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Loss settings
    ignore_index: int = -100  # Index to ignore in loss calculation
    class_weights: Optional[List[float]] = None  # Class weights for imbalanced data
    
    # CRF settings
    crf_reduction: str = "mean"  # CRF loss reduction
    
    # ==================== Debug and Testing ====================
    debug_mode: bool = False  # Enable debug mode
    fast_dev_run: bool = False  # Run only a few batches for testing
    overfit_batches: int = 0  # Overfit on N batches for debugging
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Adjust num_labels based on tagging scheme
        self.num_labels = 2  # O, EM
        
        # Ensure output directories exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate ratios
        total_ratio = self.train_ratio + self.valid_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")
        
        # Adjust batch size for gradient accumulation
        if self.gradient_accumulation_steps > 1:
            effective_batch_size = self.batch_size * self.gradient_accumulation_steps
            print(f"Effective batch size: {effective_batch_size} "
                  f"(batch_size={self.batch_size} × gradient_accumulation_steps={self.gradient_accumulation_steps})")
    
    @classmethod
    def from_json(cls, json_path: str) -> "ERAConfig":
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            ERAConfig instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def to_json(self, json_path: str):
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON configuration
        """
        # Convert to dictionary, handling special types
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {k: v for k, v in self.__dict__.items()}
    
    def update_from_dict(self, update_dict: Dict[str, Any]):
        """
        Update configuration from dictionary.
        
        Args:
            update_dict: Dictionary with configuration updates
        """
        for key, value in update_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration key: {key}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model-specific configuration.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            "bert_model": self.bert_model,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "hidden_size": self.hidden_size,
            "use_crf": self.use_crf,
            "dropout_rate": self.dropout_rate,
            "frozen_layers": self.frozen_layers
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training-specific configuration.
        
        Returns:
            Dictionary with training configuration
        """
        return {
            "learning_rate": self.learning_rate,
            "encoder_lr": self.encoder_lr,
            "head_lr": self.head_lr,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_epochs": self.num_epochs,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "optimizer": self.optimizer,
            "scheduler_type": self.scheduler_type
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data-specific configuration.
        
        Returns:
            Dictionary with data configuration
        """
        return {
            "data_dir": self.data_dir,
            "cache_dir": self.cache_dir,
            "train_ratio": self.train_ratio,
            "valid_ratio": self.valid_ratio,
            "test_ratio": self.test_ratio,
            "max_length": self.max_length,
            "oversampling_ratio": self.oversampling_ratio
        }
    
    def print_config(self):
        """Print configuration in a formatted way."""
        print("=" * 50)
        print("ERA Configuration")
        print("=" * 50)
        
        sections = {
            "Model": self.get_model_config(),
            "Training": self.get_training_config(),
            "Data": self.get_data_config(),
        }
        
        for section_name, section_config in sections.items():
            print(f"\n{section_name} Configuration:")
            print("-" * 30)
            for key, value in section_config.items():
                print(f"  {key:25}: {value}")
        
        print("\n" + "=" * 50)


# Pre-defined configuration variants
class ERAConfigPresets:
    """
    Pre-defined configuration presets for different scenarios.
    """
    
    @staticmethod
    def nuner_baseline() -> ERAConfig:
        """
        NuNER baseline configuration as specified in EAR.md.
        
        Returns:
            ERAConfig with NuNER baseline settings
        """
        return ERAConfig(
            bert_model="google-bert/bert-base-uncased",
            frozen_layers=6,
            learning_rate=3e-5,
            encoder_lr=3e-5,
            head_lr=5e-5,
            batch_size=48,
            num_epochs=10,
            warmup_ratio=0.1,
            dropout_rate=0.1,
            use_crf=True,
            early_stopping_patience=3
        )
    
    @staticmethod
    def fast_debug() -> ERAConfig:
        """
        Fast debug configuration for testing.
        
        Returns:
            ERAConfig with debug settings
        """
        config = ERAConfig()
        config.batch_size = 4
        config.num_epochs = 2
        config.max_train_samples = 100
        config.max_eval_samples = 50
        config.debug_mode = True
        config.logging_steps = 5
        config.eval_steps = 10
        return config
    
    @staticmethod
    def small_gpu() -> ERAConfig:
        """
        Configuration for small GPU memory.
        
        Returns:
            ERAConfig optimized for small GPUs
        """
        config = ERAConfig()
        config.batch_size = 16
        config.gradient_accumulation_steps = 3  # Effective batch size = 48
        config.fp16 = True
        config.dataloader_num_workers = 2
        config.max_length = 256  # Shorter sequences
        return config


def create_config_from_args(args) -> ERAConfig:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        ERAConfig instance
    """
    config = ERAConfig()
    
    # Update from args if they exist
    if hasattr(args, 'bert_model') and args.bert_model:
        config.bert_model = args.bert_model
    if hasattr(args, 'batch_size') and args.batch_size:
        config.batch_size = args.batch_size
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.learning_rate = args.learning_rate
    if hasattr(args, 'num_epochs') and args.num_epochs:
        config.num_epochs = args.num_epochs
    if hasattr(args, 'output_dir') and args.output_dir:
        config.output_dir = args.output_dir
    if hasattr(args, 'data_dir') and args.data_dir:
        config.data_dir = args.data_dir
    
    return config


# Example usage
if __name__ == "__main__":
    # Test configuration creation
    config = ERAConfigPresets.nuner_baseline()
    config.print_config()
    
    # Test JSON serialization
    config.to_json("test_config.json")
    loaded_config = ERAConfig.from_json("test_config.json")
    print("Config loaded successfully from JSON")
    
    # Clean up
    os.remove("test_config.json")