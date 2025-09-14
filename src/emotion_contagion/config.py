"""
Configuration for Emotion-Contagion Encoder

Defines hyperparameters and settings based on Mean-pooling.md specifications.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EmotionContagionConfig:
    """
    Configuration class for Emotion-Contagion encoder based on Mean-pooling.md specs.
    
    Core parameters for reproducing ReflectDiffu Section 3.2:
    - EC = EW + EP + ER embedding composition
    - Transformer encoder settings
    - Attention mechanism configuration
    """
    
    # ==================== Embedding Configuration ====================
    # Word embeddings (GloVe 300d as per paper)
    word_embedding_dim: int = 300  # Demb = 300 (GloVe)
    model_dim: int = 300  # D (hidden dimension, aligned with word embeddings)
    vocab_size: int = 50000  # Will be set based on actual vocabulary
    
    # Position embeddings
    max_position_embeddings: int = 512  # Maximum sequence length
    position_embedding_type: str = "sinusoidal"  # "sinusoidal" or "learned"
    
    # Reason embeddings (em/noem labels from ERA)
    num_reason_labels: int = 2  # em, noem
    
    # ==================== Transformer Encoder Configuration ====================
    num_encoder_layers: int = 4  # N layers as per baseline
    num_attention_heads: int = 8  # Multi-head attention
    feedforward_dim: int = 1024  # FFN intermediate dimension (4 * model_dim)
    dropout_rate: float = 0.1  # Dropout for encoder/attention/FFN
    
    # Normalization
    layer_norm_eps: float = 1e-5
    norm_first: bool = True  # Pre-LN (LayerNorm before sub-layer) for stability
    
    # ==================== Attention Configuration ====================
    attention_type: str = "cross"  # "cross" or "gate" (Method A vs B)
    attention_dropout: float = 0.1
    
    # For gate-reweight method (Method B)
    gate_activation: str = "sigmoid"  # "sigmoid" or "softmax"
    
    # ==================== Training Configuration ====================
    # Initialization
    initializer_range: float = 0.02
    
    # Numerical stability
    attention_scale: bool = True  # Scale attention by sqrt(d_k)
    use_stable_softmax: bool = True  # Subtract max for numerical stability
    
    # ==================== ERA Integration ====================
    era_hidden_dim: int = 768  # ERA model hidden dimension (RoBERTa-base)
    era_projection_dim: Optional[int] = None  # If None, use model_dim
    
    # ==================== Output Configuration ====================
    return_all_hidden_states: bool = False  # Return H for downstream modules
    return_attention_weights: bool = False  # For visualization/analysis
    
    def __post_init__(self):
        """Validate and set derived parameters."""
        # Ensure feedforward_dim is reasonable
        if self.feedforward_dim is None:
            self.feedforward_dim = 4 * self.model_dim
            
        # Set ERA projection dimension if not specified
        if self.era_projection_dim is None:
            self.era_projection_dim = self.model_dim
            
        # Validate attention heads
        if self.model_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )