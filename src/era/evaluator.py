"""
ERA Evaluator Module

Implements comprehensive evaluation metrics:
- Token-level and entity-level F1/Precision/Recall
- Confusion matrix and classification report
- Error analysis functionality
- Support for both BIO and IO tagging schemes
- Detailed performance breakdown by label types

Based on NuNER evaluation practices and EAR.md specifications.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass

# Import evaluation metrics
try:
    from sklearn.metrics import (
        classification_report, 
        confusion_matrix, 
        precision_recall_fscore_support,
        accuracy_score
    )
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("Warning: scikit-learn not installed. Some evaluation features will be disabled.")
    classification_report = None
    confusion_matrix = None

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """
    Container for evaluation results.
    """
    # Token-level metrics
    token_accuracy: float = 0.0
    token_precision: float = 0.0
    token_recall: float = 0.0
    token_f1: float = 0.0
    
    # Entity-level metrics (for BIO scheme)
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0
    
    # Class-specific metrics
    class_metrics: Dict[str, Dict[str, float]] = None
    
    # Confusion matrix
    confusion_matrix: np.ndarray = None
    
    # Error analysis
    error_analysis: Dict[str, int] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        result = {
            "token_accuracy": self.token_accuracy,
            "token_precision": self.token_precision,
            "token_recall": self.token_recall,
            "token_f1": self.token_f1,
            "entity_precision": self.entity_precision,
            "entity_recall": self.entity_recall,
            "entity_f1": self.entity_f1,
        }
        
        # Add class-specific metrics
        if self.class_metrics:
            for class_name, metrics in self.class_metrics.items():
                for metric_name, value in metrics.items():
                    result[f"{class_name}_{metric_name}"] = value
        
        return result


class ERAEvaluator:
    """
    Comprehensive evaluator for ERA model performance.
    
    Supports:
    - Token-level evaluation (standard NER metrics)
    - Entity-level evaluation (for BIO scheme)
    - Confusion matrix and classification reports
    - Error analysis and detailed breakdowns
    - Both macro and micro averaging
    """
    
    def __init__(
        self, 
        config,
        label_names: Optional[List[str]] = None,
        ignore_index: int = -100
    ):
        """
        Initialize evaluator.
        
        Args:
            config: ERA configuration object
            label_names: List of label names (e.g., ["O", "B-EM", "I-EM"])
            ignore_index: Index to ignore in evaluation
        """
        self.config = config
        self.tagging_scheme = getattr(config, 'tagging_scheme', 'BIO')
        self.ignore_index = ignore_index
        
        # Set up label names
        if label_names is None:
            if self.tagging_scheme.upper() == "BIO":
                self.label_names = ["O", "B-EM", "I-EM"]
            else:  # IO
                self.label_names = ["O", "EM"]
        else:
            self.label_names = label_names
        
        # Create label mappings
        self.id_to_label = {i: label for i, label in enumerate(self.label_names)}
        self.label_to_id = {label: i for i, label in enumerate(self.label_names)}
        
        logger.info(f"ERAEvaluator initialized:")
        logger.info(f"  Tagging scheme: {self.tagging_scheme}")
        logger.info(f"  Labels: {self.label_names}")
        logger.info(f"  Ignore index: {self.ignore_index}")
    
    def compute_metrics(
        self, 
        predictions: Union[List[List[int]], np.ndarray], 
        labels: Union[List[List[int]], np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: Predicted labels [batch_size, seq_len] or [num_samples, seq_len]
            labels: Ground truth labels [batch_size, seq_len] or [num_samples, seq_len]
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert to numpy arrays if needed
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Flatten and filter out ignored indices
        flat_predictions, flat_labels = self._flatten_and_filter(predictions, labels)
        
        if len(flat_predictions) == 0:
            logger.warning("No valid predictions found for evaluation")
            return {}
        
        # Compute token-level metrics
        token_metrics = self._compute_token_metrics(flat_predictions, flat_labels)
        
        # Compute entity-level metrics (for BIO scheme)
        entity_metrics = {}
        if self.tagging_scheme.upper() == "BIO":
            entity_metrics = self._compute_entity_metrics(predictions, labels)
        
        # Compute class-specific metrics
        class_metrics = self._compute_class_metrics(flat_predictions, flat_labels)
        
        # Compute confusion matrix
        conf_matrix = self._compute_confusion_matrix(flat_predictions, flat_labels)
        
        # Error analysis
        error_analysis = self._compute_error_analysis(flat_predictions, flat_labels)
        
        # Combine all metrics
        all_metrics = {
            **token_metrics,
            **entity_metrics,
            **{f"class_{k}": v for k, v in class_metrics.items()},
            "num_samples": len(flat_predictions),
            "num_classes": len(self.label_names)
        }
        
        # Store additional results for detailed analysis
        results = EvaluationResults(
            token_accuracy=token_metrics.get("accuracy", 0.0),
            token_precision=token_metrics.get("precision", 0.0),
            token_recall=token_metrics.get("recall", 0.0),
            token_f1=token_metrics.get("f1", 0.0),
            entity_precision=entity_metrics.get("entity_precision", 0.0),
            entity_recall=entity_metrics.get("entity_recall", 0.0),
            entity_f1=entity_metrics.get("entity_f1", 0.0),
            class_metrics=class_metrics,
            confusion_matrix=conf_matrix,
            error_analysis=error_analysis
        )
        
        return all_metrics
    
    def _flatten_and_filter(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        Flatten predictions and labels, filtering out ignored indices.
        
        Args:
            predictions: Prediction array
            labels: Label array
            
        Returns:
            Tuple of (flat_predictions, flat_labels)
        """
        flat_predictions = []
        flat_labels = []
        
        # Flatten arrays
        predictions_flat = predictions.flatten()
        labels_flat = labels.flatten()
        
        # Filter out ignored indices
        for pred, label in zip(predictions_flat, labels_flat):
            if label != self.ignore_index:
                flat_predictions.append(pred)
                flat_labels.append(label)
        
        return flat_predictions, flat_labels
    
    def _compute_token_metrics(
        self, 
        predictions: List[int], 
        labels: List[int]
    ) -> Dict[str, float]:
        """
        Compute token-level classification metrics.
        
        Args:
            predictions: Flat list of predictions
            labels: Flat list of labels
            
        Returns:
            Dictionary with token-level metrics
        """
        if classification_report is None:
            logger.warning("scikit-learn not available, computing basic metrics only")
            
            # Basic accuracy calculation
            correct = sum(p == l for p, l in zip(predictions, labels))
            accuracy = correct / len(predictions) if predictions else 0.0
            
            return {"accuracy": accuracy}
        
        # Compute standard metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Compute macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        # Compute micro averages
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='micro', zero_division=0
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1
        }
    
    def _compute_entity_metrics(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute entity-level metrics for BIO tagging scheme.
        
        Args:
            predictions: Prediction array [batch_size, seq_len]
            labels: Label array [batch_size, seq_len]
            
        Returns:
            Dictionary with entity-level metrics
        """
        if self.tagging_scheme.upper() != "BIO":
            return {}
        
        total_predicted_entities = 0
        total_gold_entities = 0
        total_correct_entities = 0
        
        for pred_seq, label_seq in zip(predictions, labels):
            # Extract entities from sequences
            pred_entities = self._extract_entities_bio(pred_seq)
            gold_entities = self._extract_entities_bio(label_seq)
            
            # Count entities
            total_predicted_entities += len(pred_entities)
            total_gold_entities += len(gold_entities)
            
            # Count correct entities (exact match)
            correct_entities = pred_entities & gold_entities
            total_correct_entities += len(correct_entities)
        
        # Compute metrics
        precision = total_correct_entities / total_predicted_entities if total_predicted_entities > 0 else 0.0
        recall = total_correct_entities / total_gold_entities if total_gold_entities > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "entity_precision": precision,
            "entity_recall": recall,
            "entity_f1": f1,
            "total_predicted_entities": total_predicted_entities,
            "total_gold_entities": total_gold_entities,
            "total_correct_entities": total_correct_entities
        }
    
    def _extract_entities_bio(self, sequence: np.ndarray) -> Set[Tuple[int, int]]:
        """
        Extract entities from BIO-tagged sequence.
        
        Args:
            sequence: BIO-tagged sequence
            
        Returns:
            Set of (start, end) entity spans
        """
        entities = set()
        current_start = None
        
        for i, label_id in enumerate(sequence):
            if label_id == self.ignore_index:
                continue
            
            label = self.id_to_label.get(label_id, "O")
            
            if label.startswith("B-"):
                # Start of new entity
                if current_start is not None:
                    # End previous entity
                    entities.add((current_start, i - 1))
                current_start = i
                
            elif label.startswith("I-"):
                # Continue current entity
                if current_start is None:
                    # Invalid I- without B-, treat as beginning
                    current_start = i
                    
            else:  # O tag
                # End current entity
                if current_start is not None:
                    entities.add((current_start, i - 1))
                    current_start = None
        
        # Handle entity at end of sequence
        if current_start is not None:
            entities.add((current_start, len(sequence) - 1))
        
        return entities
    
    def _compute_class_metrics(
        self, 
        predictions: List[int], 
        labels: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics.
        
        Args:
            predictions: Flat list of predictions
            labels: Flat list of labels
            
        Returns:
            Dictionary with per-class metrics
        """
        if classification_report is None:
            return {}
        
        # Get unique labels
        unique_labels = sorted(list(set(labels + predictions)))
        
        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, labels=unique_labels, zero_division=0
        )
        
        class_metrics = {}
        for i, label_id in enumerate(unique_labels):
            label_name = self.id_to_label.get(label_id, f"label_{label_id}")
            class_metrics[label_name] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": support[i]
            }
        
        return class_metrics
    
    def _compute_confusion_matrix(
        self, 
        predictions: List[int], 
        labels: List[int]
    ) -> Optional[np.ndarray]:
        """
        Compute confusion matrix.
        
        Args:
            predictions: Flat list of predictions
            labels: Flat list of labels
            
        Returns:
            Confusion matrix
        """
        if confusion_matrix is None:
            return None
        
        # Get unique labels
        unique_labels = sorted(list(set(labels + predictions)))
        
        # Compute confusion matrix
        cm = confusion_matrix(labels, predictions, labels=unique_labels)
        
        return cm
    
    def _compute_error_analysis(
        self, 
        predictions: List[int], 
        labels: List[int]
    ) -> Dict[str, int]:
        """
        Perform error analysis.
        
        Args:
            predictions: Flat list of predictions
            labels: Flat list of labels
            
        Returns:
            Dictionary with error counts
        """
        error_counts = defaultdict(int)
        
        for pred, label in zip(predictions, labels):
            if pred != label:
                pred_name = self.id_to_label.get(pred, f"pred_{pred}")
                label_name = self.id_to_label.get(label, f"true_{label}")
                error_type = f"{label_name}_to_{pred_name}"
                error_counts[error_type] += 1
        
        return dict(error_counts)
    
    def print_evaluation_report(
        self, 
        predictions: Union[List[List[int]], np.ndarray], 
        labels: Union[List[List[int]], np.ndarray]
    ):
        """
        Print detailed evaluation report.
        
        Args:
            predictions: Predicted labels
            labels: Ground truth labels
        """
        metrics = self.compute_metrics(predictions, labels)
        
        print("=" * 50)
        print("ERA Model Evaluation Report")
        print("=" * 50)
        
        # Token-level metrics
        print("\nToken-level Metrics:")
        print("-" * 20)
        for metric in ["accuracy", "precision", "recall", "f1", "macro_f1", "micro_f1"]:
            if metric in metrics:
                print(f"{metric:15}: {metrics[metric]:.4f}")
        
        # Entity-level metrics (if available)
        if any(k.startswith("entity_") for k in metrics.keys()):
            print("\nEntity-level Metrics:")
            print("-" * 20)
            for metric in ["entity_precision", "entity_recall", "entity_f1"]:
                if metric in metrics:
                    print(f"{metric:15}: {metrics[metric]:.4f}")
        
        # Class-specific metrics
        class_metrics = {k: v for k, v in metrics.items() if k.startswith("class_")}
        if class_metrics:
            print("\nPer-class Metrics:")
            print("-" * 20)
            for class_metric, value in class_metrics.items():
                print(f"{class_metric:20}: {value:.4f}")
        
        print("=" * 50)
    
    def save_evaluation_results(
        self, 
        predictions: Union[List[List[int]], np.ndarray], 
        labels: Union[List[List[int]], np.ndarray],
        output_file: str
    ):
        """
        Save detailed evaluation results to file.
        
        Args:
            predictions: Predicted labels
            labels: Ground truth labels
            output_file: Output file path
        """
        import json
        
        metrics = self.compute_metrics(predictions, labels)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, np.integer):
                serializable_metrics[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_file}")


# Utility functions
def compare_models(
    evaluator: ERAEvaluator,
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    metric: str = "f1"
) -> Dict[str, float]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        evaluator: ERAEvaluator instance
        model_results: Dictionary of {model_name: (predictions, labels)}
        metric: Metric to use for comparison
        
    Returns:
        Dictionary of {model_name: metric_value}
    """
    comparison = {}
    
    for model_name, (predictions, labels) in model_results.items():
        metrics = evaluator.compute_metrics(predictions, labels)
        comparison[model_name] = metrics.get(metric, 0.0)
    
    return comparison


def create_evaluator_from_config(config) -> ERAEvaluator:
    """
    Factory function to create evaluator from configuration.
    
    Args:
        config: ERA configuration object
        
    Returns:
        Configured ERAEvaluator
    """
    return ERAEvaluator(config=config)


# Example usage
if __name__ == "__main__":
    # Test evaluator with dummy data
    from .config import ERAConfig
    
    config = ERAConfig()
    evaluator = ERAEvaluator(config)
    
    # Dummy data
    predictions = np.array([[0, 1, 2, 0], [1, 0, 0, 2]])
    labels = np.array([[0, 1, 2, 0], [1, 0, 1, 2]])
    
    metrics = evaluator.compute_metrics(predictions, labels)
    print("Test metrics:", metrics)