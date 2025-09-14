"""
Emotion-Contagion Encoder Implementation

Implements the core emotion-contagion encoder from ReflectDiffu Section 3.2:
1. EC = EW + EP + ER (word + position + reason embeddings)
2. H = TRSEnc(EC) (Transformer encoder)  
3. Attention(H, h̃) (reason-guided attention)
4. Q = mean-pooling(Attention(H, h̃)) (global context summary)

Based on Mean-pooling.md specifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
import logging
from foundation_emb import WordEmbedding, SinusoidalPositionalEncoding, ReasonEmbedding

from .config import EmotionContagionConfig
from contrastive_expert import CONExpert

logger = logging.getLogger(__name__)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with Pre-LN."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with Pre-LN architecture.
        
        Args:
            src: Input tensor [B, L, D]
            src_key_padding_mask: Padding mask [B, L] (True for padding)
            
        Returns:
            Output tensor [B, L, D]
        """
        # Pre-LN self-attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # Pre-LN feedforward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class CrossAttention(nn.Module):
    """Cross-attention mechanism: Attention(H, h̃)."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.d_model = d_model
    
    def forward(
        self, 
        H: torch.Tensor, 
        h_tilde: torch.Tensor, 
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention: H queries h̃.
        
        Args:
            H: Encoder output [B, L, D]
            h_tilde: ERA reasoning representation [B, L, D]
            key_padding_mask: Padding mask for h̃ [B, L] (True for padding)
            
        Returns:
            Attention output Z [B, L, D]
        """
        # H as Query, h̃ as Key and Value
        Z, attn_weights = self.multihead_attn(
            query=H,
            key=h_tilde,
            value=h_tilde,
            key_padding_mask=key_padding_mask
        )
        
        return Z


class GateReweight(nn.Module):
    """Gate-based reweighting mechanism (Alternative to cross-attention)."""
    
    def __init__(self, d_model: int, activation: str = "sigmoid"):
        super().__init__()
        self.gate_projection = nn.Linear(d_model, 1)
        self.activation = activation
    
    def forward(
        self, 
        H: torch.Tensor, 
        h_tilde: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Gate-based reweighting: Z = a ⊙ H.
        
        Args:
            H: Encoder output [B, L, D]
            h_tilde: ERA reasoning representation [B, L, D]
            attention_mask: Attention mask [B, L] (1 for valid, 0 for padding)
            
        Returns:
            Reweighted output Z [B, L, D]
        """
        # Compute gate weights from h̃
        gate_logits = self.gate_projection(h_tilde)  # [B, L, 1]
        
        if self.activation == "sigmoid":
            gate_weights = torch.sigmoid(gate_logits)
        elif self.activation == "softmax":
            # Apply mask before softmax
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
                gate_logits = gate_logits.masked_fill(~mask.bool(), float('-inf'))
            gate_weights = F.softmax(gate_logits, dim=1)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        # Apply mask to gate weights
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
            gate_weights = gate_weights * mask
        
        # Reweight H
        Z = gate_weights * H
        
        return Z


class EmotionContagionEncoder(nn.Module):
    """
    Complete Emotion-Contagion Encoder implementation.
    
    Implements the full pipeline from Mean-pooling.md:
    1. EC = EW + EP + ER
    2. H = TRSEnc(EC)
    3. Z = Attention(H, h̃)
    4. Q = mean-pooling(Z)
    """
    
    def __init__(self, config: EmotionContagionConfig):
        super().__init__()
        self.config = config
        
        # ==================== Embedding Layers ====================
        self.word_embedding = WordEmbedding(config.vocab_size, config.word_embedding_dim)
        
        # Projection layer if word_embedding_dim != model_dim
        if config.word_embedding_dim != config.model_dim:
            self.word_projection = nn.Linear(config.word_embedding_dim, config.model_dim)
        else:
            self.word_projection = nn.Identity()
        
        # Positional encoding
        if config.position_embedding_type == "sinusoidal":
            self.position_embedding = SinusoidalPositionalEncoding(
                config.max_position_embeddings, 
                config.model_dim
            )
        else:
            self.position_embedding = nn.Embedding(
                config.max_position_embeddings, 
                config.model_dim
            )
        
        # Reason embedding
        self.reason_embedding = ReasonEmbedding(config.num_reason_labels, config.model_dim)
        
        # ==================== Transformer Encoder ====================
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout_rate
            )
            for _ in range(config.num_encoder_layers)
        ])
        
        # ==================== Attention Mechanism ====================
        if config.attention_type == "cross":
            self.attention = CrossAttention(
                d_model=config.model_dim,
                nhead=config.num_attention_heads,
                dropout=config.attention_dropout
            )
        elif config.attention_type == "gate":
            self.attention = GateReweight(
                d_model=config.model_dim,
                activation=config.gate_activation
            )
        else:
            raise ValueError(f"Unknown attention type: {config.attention_type}")
        
        # ==================== ERA Integration ====================
        if config.era_projection_dim != config.model_dim:
            self.era_projection = nn.Linear(config.era_hidden_dim, config.model_dim)
        else:
            self.era_projection = nn.Identity()
            
        self.conExport = CONExpert(config.model_dim)
    
    def build_vocab_from_data(self, data_loader):
        """Build vocabulary from data loader."""
        all_tokens = []
        for batch in data_loader:
            all_tokens.extend(batch["tokens"])
        self.word_embedding.build_vocab(all_tokens)
    
    def forward(
        self,
        tokens: List[List[str]],
        label_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        h_tilde: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through emotion-contagion encoder.
        
        Args:
            tokens: Token sequences [B, L] (list of strings)
            label_ids: Reason label IDs [B, L]
            attention_mask: Attention mask [B, L] (1 for valid, 0 for padding)
            h_tilde: ERA reasoning representation [B, L, D_era] (optional)
            
        Returns:
            Dictionary containing:
            - H: Encoder output [B, L, D]
            - Q: Global context summary [B, D]
            - Z: Attention output [B, L, D] (if h_tilde provided)
        """
        batch_size, seq_len = label_ids.shape
        device = label_ids.device
        
        # ==================== Step 1: EC = EW + EP + ER ====================
        # Word embeddings
        EW = self.word_embedding(tokens)  # [B, L, D_emb]
        EW = self.word_projection(EW)     # [B, L, D]
        
        # Position embeddings
        if isinstance(self.position_embedding, SinusoidalPositionalEncoding):
            EP = self.position_embedding(seq_len).unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, D]
        else:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            EP = self.position_embedding(positions)  # [B, L, D]
        
        # Reason embeddings
        ER = self.reason_embedding(label_ids)  # [B, L, D]
        
        # Combine embeddings
        EC = EW + EP + ER  # [B, L, D]
        
        # ==================== Step 2: H = TRSEnc(EC) ====================
        # Convert attention_mask to padding mask (True for padding positions)
        padding_mask = ~attention_mask.bool()  # [B, L]
        
        H = EC
        for layer in self.encoder_layers:
            H = layer(H, src_key_padding_mask=padding_mask)  # [B, L, D]
        
        # ==================== Step 3: Attention(H, h̃) ====================
        if h_tilde is not None:
            # Project h̃ to model dimension if needed
            h_tilde_proj = self.era_projection(h_tilde)  # [B, L, D]
            
            # Apply attention mechanism
            if isinstance(self.attention, CrossAttention):
                Z = self.attention(H, h_tilde_proj, key_padding_mask=padding_mask)
            else:  # GateReweight
                Z = self.attention(H, h_tilde_proj, attention_mask=attention_mask)
        else:
            # If no h̃ provided, use H directly
            Z = H
        
        # ==================== Step 4: Q = mean-pooling(Z) ====================
        # Masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        masked_Z = Z * mask_expanded  # [B, L, D]
        
        # Sum over sequence length and divide by valid length
        sum_Z = masked_Z.sum(dim=1)  # [B, D]
        seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
        Q = sum_Z / seq_lengths  # [B, D]
        
        # ==================== Step 5: Contrastive Expert ====================
        P = self.conExport.forward(Q, tokens)
        
        # ==================== Return Results ====================
        results = {
            "H": H,  # Encoder output [B, L, D]
            "Q": Q,  # Global context summary [B, D]
            "P": P   # Contrastive Expert output [B, D]
        }
        
        return results