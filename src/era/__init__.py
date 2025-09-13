"""
ERA (Emotion Reason Annotator) - A BERT+CRF based sequence labeling system
for emotion reason word detection in dialogues.

Based on ReflectDiffu framework and NuNER training paradigm.
"""

__version__ = "1.0.0"
__author__ = "ReflectDiffu-Reproduce Team"

from .era_model import ERA_BERT_CRF
from .data_processor import EmpathyDataProcessor
from .trainer import ERATrainer
from .evaluator import ERAEvaluator
from .config import ERAConfig

__all__ = [
    "ERA_BERT_CRF",
    "EmpathyDataProcessor", 
    "ERATrainer",
    "ERAEvaluator",
    "ERAConfig"
]