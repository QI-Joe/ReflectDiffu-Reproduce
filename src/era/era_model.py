"""
ERA Model Architecture: BERT + Linear + CRF

Implements ERA_BERT_CRF class following NuNER specifications:
- BERT encoder with configurable frozen layers
- Linear projection layer
- CRF layer for sequence consistency
- Support for both BIO and IO tagging schemes
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModel, 
    AutoTokenizer,
    RobertaModel,
    BertModel,
    PreTrainedModel,
    PretrainedConfig
)
from typing import Optional, Tuple, Dict, Any
import logging
from torchcrf import CRF

logger = logging.getLogger(__name__)


class ERAConfig(PretrainedConfig):
    """
    Configuration class for ERA model.
    
    Extends Hugging Face PretrainedConfig for compatibility.
    """
    
    model_type = "era_bert_crf"
    
    def __init__(
        self,
        bert_model: str = "bert-base",
        num_labels: int = 2,
        hidden_size: int = 768,
        dropout_rate: float = 0.1,
        use_crf: bool = True,
        frozen_layers: int = 6,
        crf_reduction: str = "mean",
        ignore_index: int = -100,
        **kwargs
    ):
        """
        Initialize ERA configuration.
        
        Args:
            bert_model: Pre-trained BERT model name
            num_labels: Number of labels (2 for IO scheme: O, EM)
            hidden_size: Hidden size of BERT model
            dropout_rate: Dropout rate for classification head
            use_crf: Whether to use CRF layer
            frozen_layers: Number of BERT layers to freeze
            crf_reduction: CRF loss reduction method
            ignore_index: Index to ignore in loss calculation
        """
        """
        Initialize ERA configuration.
        
        Args:
            bert_model: Pre-trained BERT model name
            num_labels: Number of labels (2 for IO, 3 for BIO)
            hidden_size: Hidden size of BERT model
            dropout_rate: Dropout rate for classification head
            use_crf: Whether to use CRF layer
            frozen_layers: Number of BERT layers to freeze
            crf_reduction: CRF loss reduction method
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__(**kwargs)
        
        self.bert_model = bert_model
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.use_crf = use_crf
        self.frozen_layers = frozen_layers
        self.crf_reduction = crf_reduction
        self.ignore_index = ignore_index


class ERA_BERT_CRF(PreTrainedModel):
    """
    ERA model: BERT encoder + Linear projection + CRF layer.
    
    Architecture:
    1. BERT/RoBERTa encoder (with configurable frozen layers)
    2. Dropout layer
    3. Linear classification head
    4. Optional CRF layer for sequence consistency
    
    Based on NuNER specifications and EAR.md requirements.
    """
    
    config_class = ERAConfig
    
    def __init__(self, config: ERAConfig):
        """
        Initialize ERA model.
        
        Args:
            config: ERAConfig instance
        """
        super().__init__(config)
        
        self.config = config
        self.num_labels = config.num_labels
        self.use_crf = config.use_crf and CRF is not None
        
        if not self.use_crf and config.use_crf:
            logger.warning("CRF requested but torchcrf not available. Using softmax instead.")
        
        # Initialize BERT encoder
        self.bert = self._load_bert_model(config.bert_model)
        
        # Freeze specified layers
        self._freeze_bert_layers(config.frozen_layers)
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # CRF layer (optional)
        if self.use_crf:
            self.crf = CRF(config.num_labels, batch_first=True)
        else:
            self.crf = None
        
        # Loss function for non-CRF case
        self.loss_fct = CrossEntropyLoss(ignore_index=config.ignore_index)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"ERA model initialized:")
        logger.info(f"  - Encoder: {config.bert_model}")
        logger.info(f"  - Frozen layers: {config.frozen_layers}")
        logger.info(f"  - Use CRF: {self.use_crf}")
        logger.info(f"  - Dropout: {config.dropout_rate}")
    
    def _load_bert_model(self, model_name: str) -> nn.Module:
        """
        Load BERT model based on model name.
        
        Args:
            model_name: Name or path of BERT model
            
        Returns:
            BERT model instance
        """
        try:
            if "roberta" in model_name.lower():
                bert_model = RobertaModel.from_pretrained(model_name)
            elif "bert" in model_name.lower():
                bert_model = BertModel.from_pretrained(model_name)
            else:
                # Try generic AutoModel
                bert_model = AutoModel.from_pretrained(model_name)
            
            logger.info(f"Loaded BERT model: {model_name}")
            return bert_model
            
        except Exception as e:
            logger.error(f"Failed to load BERT model {model_name}: {e}")
            raise
    
    def _freeze_bert_layers(self, frozen_layers: int):
        """
        Freeze the first N layers of BERT.
        
        Args:
            frozen_layers: Number of layers to freeze (0 = no freezing)
        """
        if frozen_layers <= 0:
            logger.info("No BERT layers frozen")
            return
        
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers
        total_layers = len(self.bert.encoder.layer)
        layers_to_freeze = min(frozen_layers, total_layers)
        
        for i in range(layers_to_freeze):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        logger.info(f"Frozen {layers_to_freeze}/{total_layers} BERT layers")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} "
                   f"({100 * trainable_params / total_params:.1f}%)")
    
    def _init_weights(self):
        """Initialize classification head weights."""
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        logger.info("Initialized classification head weights")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of ERA model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional)
            labels: Ground truth labels [batch_size, seq_len] (optional)
            return_dict: Whether to return as dictionary
        """
        # BERT encoder
        bert_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if token_type_ids is not None:
            bert_inputs["token_type_ids"] = token_type_ids
        
        bert_outputs = self.bert(**bert_inputs)
        hidden_states = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Classification head
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)  # [batch_size, seq_len, num_labels]
        
        # Prepare outputs
        outputs = {
            "logits": logits,
            "hidden_states": hidden_states
        }
        
        # CRF 
        if labels is not None:
            # Create mask for CRF (exclude ignored tokens)
            crf_mask = (labels != self.config.ignore_index) & (attention_mask.bool())
            
            # Replace ignore_index with 0 for CRF (will be masked anyway)
            crf_labels = labels.clone()
            crf_labels[labels == self.config.ignore_index] = 0
            
            # Compute CRF loss
            loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction=self.config.crf_reduction)
            outputs["loss"] = loss
        
        # CRF predictions (Viterbi decoding)
        if attention_mask is not None:
            crf_mask = attention_mask.bool()
        else:
            crf_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        predictions = self.crf.decode(logits, mask=crf_mask)
        
        # Convert list of sequences to tensor (pad with ignore_index)
        batch_size, seq_len = input_ids.shape
        pred_tensor = torch.full(
            (batch_size, seq_len), 
            self.config.ignore_index, 
            dtype=torch.long, 
            device=input_ids.device
        )
        
        for i, pred_seq in enumerate(predictions):
            seq_length = min(len(pred_seq), seq_len)
            pred_tensor[i, :seq_length] = torch.tensor(pred_seq[:seq_length])
        
        outputs["predictions"] = pred_tensor
            
        return outputs
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Make predictions without computing loss.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)
            
        Returns:
            Predicted labels [batch_size, seq_len]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None
            )
            return outputs["predictions"]
    
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get model logits without loss computation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)
            
        Returns:
            Logits [batch_size, seq_len, num_labels]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None
            )
            return outputs["logits"]
    
    def save_pretrained(self, save_directory: str):
        """
        Save model to directory.
        
        Args:
            save_directory: Directory to save model
        """
        super().save_pretrained(save_directory)
        logger.info(f"ERA model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load model from directory.
        
        Args:
            pretrained_model_name_or_path: Path to saved model
            
        Returns:
            ERA model instance
        """
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        logger.info(f"ERA model loaded from {pretrained_model_name_or_path}")
        return model


def create_era_model(
    bert_model: str = "google-bert/bert-base-uncased",
    num_labels: int = 3,
    use_crf: bool = True,
    frozen_layers: int = 6,
    dropout_rate: float = 0.1,
    device: str = "auto"
) -> ERA_BERT_CRF:
    """
    Factory function to create ERA model with default settings.
    
    Args:
        bert_model: BERT model name
        num_labels: Number of labels
        use_crf: Whether to use CRF
        frozen_layers: Number of layers to freeze
        dropout_rate: Dropout rate
        device: Device to place model on
        
    Returns:
        ERA model instance
    """
    # Create configuration
    config = ERAConfig(
        bert_model=bert_model,
        num_labels=num_labels,
        use_crf=use_crf,
        frozen_layers=frozen_layers,
        dropout_rate=dropout_rate
    )
    
    # Create model
    model = ERA_BERT_CRF(config)
    
    # Move to device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    logger.info(f"ERA model created and moved to {device}")
    
    return model
