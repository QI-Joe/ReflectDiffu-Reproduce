"""
Emotion-Contagion Encoder Module

Implements the first stage of ReflectDiffu's emotion-contagion encoder:
- EC = EW + EP + ER (word + position + reason embeddings)
- H = TRSEnc(EC) (Transformer encoder)
- Attention(H, h̃) (reason-guided attention)
- Q = mean-pooling(Attention(H, h̃)) (global context summary)

Based on Mean-pooling.md specifications from ReflectDiffu paper Section 3.2.
"""

__version__ = "1.0.0"
__author__ = "ReflectDiffu-Reproduce Team"

from .data_processor import EmotionContagionDataProcessor, EmotionContagionDataset
from .encoder import EmotionContagionEncoder
from .config import EmotionContagionConfig

__all__ = [
    "EmotionContagionDataProcessor",
    "EmotionContagionDataset",
    "EmotionContagionEncoder", 
    "EmotionContagionConfig"
]