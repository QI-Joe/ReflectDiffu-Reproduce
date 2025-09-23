import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
import logging

class WordEmbedding(nn.Module):
    """Word embedding layer with vocabulary mapping."""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Legacy word-to-index mapping (kept for backward compatibility if no tokenizer attached)
        self.word_to_idx = {}
        self.idx_to_word = {}
        # External tokenizer reference (e.g., HuggingFace AutoTokenizer)
        self._external_tokenizer = None
        self._enforce_external_only = False  # when True, forbid fallback local mapping

    def attach_tokenizer(self, tokenizer, enforce_external_only: bool = True):
        """Attach an external tokenizer providing stable ids.

        Args:
            tokenizer: object with __len__, encode / __call__, pad_token_id, etc.
            enforce_external_only: if True, disable local tokens_to_ids path to prevent id drift.
        """
        self._external_tokenizer = tokenizer
        self._enforce_external_only = enforce_external_only
        # If external tokenizer vocab size differs, resize embedding.
        ext_size = len(tokenizer)
        if ext_size != self.vocab_size:
            old_weight = self.embedding.weight.data
            self.embedding = nn.Embedding(ext_size, self.embedding_dim)
            # Partial copy (min old,new)
            copy_n = min(old_weight.size(0), ext_size)
            self.embedding.weight.data[:copy_n] = old_weight[:copy_n]
            self.vocab_size = ext_size
         
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices."""
        unk_idx = self.word_to_idx.get("[UNK]", 0)
        return [self.word_to_idx.get(token, unk_idx) for token in tokens]
    
    def forward(self, tokens) -> torch.Tensor:
        """
        Forward pass for word embeddings.
        
        Args:
            tokens: Either List[List[str]] (legacy) or a LongTensor of shape [B, L]
            
        Returns:
            Word embeddings [B, L, D_emb]
        """
        if torch.is_tensor(tokens):
            input_ids = tokens.to(self.embedding.weight.device)
        else:
            if self._external_tokenizer is not None and self._enforce_external_only:
                raise RuntimeError("String token sequences passed to WordEmbedding while external tokenizer attached with enforce_external_only=True. Pre-tokenize upstream and pass LongTensor IDs.")
            batch_indices = []
            for token_seq in tokens:
                # Legacy path (will be removed once full tokenizer migration complete)
                indices = self.tokens_to_ids(token_seq)
                batch_indices.append(indices)
            input_ids = torch.tensor(batch_indices, dtype=torch.long, device=self.embedding.weight.device)
        
        return self.embedding(input_ids)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-learnable)."""
    
    def __init__(self, max_length: int, d_model: int):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encodings for sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encodings [seq_len, d_model]
        """
        return self.pe[:seq_len]


class ReasonEmbedding(nn.Module):
    """Reason embedding for em/noem labels."""
    
    def __init__(self, num_labels: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_labels, d_model)
    
    def forward(self, label_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for reason embeddings.
        
        Args:
            label_ids: Label IDs [B, L]
            
        Returns:
            Reason embeddings [B, L, D]
        """
        return self.embedding(label_ids)

class IntentSemanticScorer(nn.Module):
    """
    用可学习的意图原型/嵌入 E_intent ∈ [9, D] 与 Q ∈ [B, D] 做相似度，得到 psemantic ∈ [B, 9]。
    可选：一个线性层将 Q 投影到 D_proj 后再算相似度（若想与 EmpHi intent_embeddings 尺寸对齐）。
    """
    def __init__(self, d_in: int, num_intents: int = 9, use_proj: bool = False, d_proj: int = None):
        super().__init__()
        self.num_intents = num_intents
        self.use_proj = use_proj
        if use_proj:
            assert d_proj is not None, "d_proj 必须提供"
            self.proj = nn.Linear(d_in, d_proj)
            d_final = d_proj
        else:
            self.proj = nn.Identity()
            d_final = d_in

        # 意图原型，随机初始化；也可以从 EmpHi.intent_embeddings 拷贝初始化
        # should nn.Embedding(9 intentions) and then insert into nn.Parameter
        self.intent_prototypes = nn.Parameter(torch.randn(num_intents, d_final) * 0.02)

    @torch.no_grad()
    def init_from_pretrained(self, pretrained_intent_emb: torch.Tensor):
        """
        用外部预训练的意图嵌入初始化（例如 EmpHi 的 intent_embeddings.weight，形状 [9, D或d_proj]）
        """
        assert pretrained_intent_emb.shape == self.intent_prototypes.shape
        self.intent_prototypes.copy_(pretrained_intent_emb)

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """
        输入:
            Q: [B, D] Emotion-Contagion Encoder 的全局表示
        输出:
            psemantic: [B, 9] 在线意图语义分布
        """
        Qp = self.proj(Q)                      # [B, Df]
        Qn = F.normalize(Qp, dim=-1)           # 归一化
        En = F.normalize(self.intent_prototypes, dim=-1)  # [9, Df]
        sims = torch.matmul(Qn, En.t())        # [B, 9]
        psemantic = F.softmax(sims, dim=-1)    # [B, 9]
        return psemantic, sims  # 返回相似度便于调试
    
