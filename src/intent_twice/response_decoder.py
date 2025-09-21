import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class PaperCompliantResponseDecoder(nn.Module):
    """
    Response Decoder
    
    关键特性：
    - 单层Transformer decoder block
    - Emofused作为Cross-Attention的唯一KV
    - Pointer-Generator机制
    - 简化的位置编码
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding层
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        
        # 单层Decoder Block
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropouts
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Output projections
        self.vocab_proj = nn.Linear(d_model, vocab_size)
        
        # Pointer-Generator gate
        # Input: [hidden_state, context_vector, input_embedding]
        self.pointer_gate = nn.Linear(3 * d_model, 1)
        
    def forward(
        self,
        trg_input_ids: torch.Tensor,  # [B, T] 输入token ids
        emofused: torch.Tensor,  # [B, L_emof, D] 来自IntentTwice的Emofused
        src_token_ids: torch.Tensor,  # [B, L_src] 源序列token ids (for copying)
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T] 目标序列padding mask
        emofused_key_padding_mask: Optional[torch.Tensor] = None,  # [B, L_emof] Emofused padding mask
        extended_vocab_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        论文3.3.2的核心实现: P(Rt | ER<t, Emofused, C)
        
        Args:
            trg_input_ids: Target input tokens [B, T]
            emofused: Fused emotional context from IntentTwice [B, L_emof, D]
            src_token_ids: Source tokens for copying [B, L_src]
            tgt_key_padding_mask: Target padding mask [B, T]
            emofused_key_padding_mask: Emofused padding mask [B, L_emof]
            extended_vocab_size: Extended vocabulary size for OOV handling
            
        Returns:
            Dictionary containing:
            - Pw: Final probability distribution [B, T, V_ext]
            - Pgen: Generation probability [B, T, V]
            - Pcopy: Copy probability [B, T, V_ext]
            - p_mix: Mixing coefficient [B, T, 1]
            - hidden: Hidden states [B, T, D]
            - attn_weights: Cross-attention weights [B, T, L_emof]
        """
        device = trg_input_ids.device
        B, T = trg_input_ids.size()
        
        # 1. Token + Position Embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(trg_input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        
        # 2. Causal mask for self-attention
        causal_mask = self._generate_causal_mask(T, device)
        
        # 3. Single Decoder Layer
        
        # Self-Attention (with causal mask)
        x_norm = self.norm1(x)
        self_attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        x = x + self.dropout1(self_attn_out)
        
        # Cross-Attention (核心：使用Emofused作为KV)
        x_norm = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            query=x_norm,  # [B, T, D]
            key=emofused,   # [B, L_emof, D] - 这是关键！
            value=emofused, # [B, L_emof, D] - 使用Emofused作为唯一KV
            key_padding_mask=emofused_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        x = x + self.dropout2(cross_attn_out)
        
        # Feed-forward
        x_norm = self.norm3(x)
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x_norm))))
        x = x + self.dropout3(ff_out)
        
        hidden = x  # [B, T, D]
        
        # 4. 计算context vector (用于Pointer-Generator)
        # cross_attn_weights: [B, nhead, T, L_emof]
        attn_weights_avg = cross_attn_weights.mean(dim=1)  # [B, T, L_emof]
        context_vector = torch.bmm(attn_weights_avg, emofused)  # [B, T, D]
        
        # 5. Generation Distribution
        vocab_logits = self.vocab_proj(hidden)  # [B, T, V]
        Pgen = F.softmax(vocab_logits, dim=-1)  # [B, T, V]
        
        # 6. Pointer-Generator Gate
        # 输入embedding (用于gate计算)
        input_embeds = self.token_embed(trg_input_ids)  # [B, T, D]
        gate_input = torch.cat([hidden, context_vector, input_embeds], dim=-1)  # [B, T, 3D]
        p_mix = torch.sigmoid(self.pointer_gate(gate_input))  # [B, T, 1]
        
        # 7. Copy Distribution
        # 使用attention weights作为copy distribution
        Vext = extended_vocab_size if extended_vocab_size is not None else self.vocab_size
        Pcopy = self._scatter_copy_attention(
            attn_weights_avg,  # [B, T, L_emof]
            src_token_ids,     # [B, L_src]
            Vext
        )  # [B, T, V_ext]
        
        # 8. Extend Pgen to match extended vocabulary
        if Vext > self.vocab_size:
            Pgen_ext = torch.zeros(B, T, Vext, device=device, dtype=Pgen.dtype)
            Pgen_ext[:, :, :self.vocab_size] = Pgen
        else:
            Pgen_ext = Pgen
        
        # 9. Final Mixture: P(w) = p_mix * P_gen + (1 - p_mix) * P_copy
        Pw = p_mix * Pgen_ext + (1.0 - p_mix) * Pcopy
        
        return {
            "Pw": Pw,                           # [B, T, V_ext] 最终概率分布
            "Pgen": Pgen_ext,                   # [B, T, V_ext] 生成概率
            "Pcopy": Pcopy,                     # [B, T, V_ext] 复制概率
            "p_mix": p_mix,                     # [B, T, 1] 混合权重
            "hidden": hidden,                   # [B, T, D] 隐藏状态
            "attn_weights": attn_weights_avg,   # [B, T, L_emof] 注意力权重
            "context_vector": context_vector,   # [B, T, D] 上下文向量
        }
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """生成因果mask"""
        mask = torch.full((size, size), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask
    
    def _scatter_copy_attention(
        self, 
        attn_weights: torch.Tensor,  # [B, T, L_emof]
        src_token_ids: torch.Tensor,  # [B, L_src]
        vocab_size: int
    ) -> torch.Tensor:
        """
        将attention weights散布到词汇表空间
        
        注意：这里假设src_token_ids与emofused的序列长度匹配
        实际使用时需要确保这种对应关系
        """
        B, T, L_emof = attn_weights.size()
        L_src = src_token_ids.size(1)
        device = attn_weights.device
        
        # 确保长度匹配（简化处理）
        if L_emof != L_src:
            # 截断或填充to匹配
            if L_emof > L_src:
                # Pad src_token_ids
                pad_size = L_emof - L_src
                src_token_ids = F.pad(src_token_ids, (0, pad_size), value=0)
            else:
                # Truncate src_token_ids
                src_token_ids = src_token_ids[:, :L_emof]
        
        # 初始化copy distribution
        copy_dist = torch.zeros(B, T, vocab_size, device=device, dtype=attn_weights.dtype)
        
        # 展开indices用于scatter
        src_indices = src_token_ids.unsqueeze(1).expand(-1, T, -1)  # [B, T, L_emof]
        
        # Scatter add attention weights to vocabulary positions
        copy_dist.scatter_add_(dim=2, index=src_indices, src=attn_weights)
        
        return copy_dist


class PaperCompliantDecoderWithLoss(nn.Module):
    """
    带损失计算的完整解码器
    """
    
    def __init__(
        self,
        decoder: PaperCompliantResponseDecoder,
        label_smoothing: float = 0.0,
        coverage_weight: float = 0.0
    ):
        super().__init__()
        self.decoder = decoder
        self.label_smoothing = label_smoothing
        self.coverage_weight = coverage_weight
        
    def forward(
        self,
        trg_input_ids: torch.Tensor,
        emofused: torch.Tensor,
        src_token_ids: torch.Tensor,
        gold_ids: torch.Tensor,  # [B, T] Ground truth tokens
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        emofused_key_padding_mask: Optional[torch.Tensor] = None,
        extended_vocab_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 + 损失计算
        """
        # 解码器前向传播
        decoder_out = self.decoder(
            trg_input_ids=trg_input_ids,
            emofused=emofused,
            src_token_ids=src_token_ids,
            tgt_key_padding_mask=tgt_key_padding_mask,
            emofused_key_padding_mask=emofused_key_padding_mask,
            extended_vocab_size=extended_vocab_size,
        )
        
        # 计算损失
        loss_dict = self._compute_loss(
            Pw=decoder_out["Pw"],
            gold_ids=gold_ids,
            pad_mask=~tgt_key_padding_mask if tgt_key_padding_mask is not None else None,
            attn_weights=decoder_out["attn_weights"]
        )
        
        return {**decoder_out, **loss_dict}
    
    def _compute_loss(
        self,
        Pw: torch.Tensor,  # [B, T, V]
        gold_ids: torch.Tensor,  # [B, T]
        pad_mask: Optional[torch.Tensor] = None,  # [B, T] True for valid positions
        attn_weights: Optional[torch.Tensor] = None,  # [B, T, L] for coverage
    ) -> Dict[str, torch.Tensor]:
        """
        计算交叉熵损失和可选的coverage损失
        """
        B, T, V = Pw.shape
        device = Pw.device
        eps = 1e-12
        
        # 1. Cross-Entropy Loss
        # Clamp gold_ids to valid range
        gold_ids_clamped = gold_ids.clamp(min=0, max=V-1)
        
        # Gather probabilities at gold positions
        gold_probs = Pw.gather(dim=2, index=gold_ids_clamped.unsqueeze(-1)).squeeze(-1)  # [B, T]
        gold_probs = gold_probs.clamp(min=eps)
        
        # Negative log likelihood
        nll = -torch.log(gold_probs)  # [B, T]
        
        # Apply padding mask
        if pad_mask is not None:
            mask_float = pad_mask.float()
            ce_loss = (nll * mask_float).sum() / (mask_float.sum() + eps)
        else:
            ce_loss = nll.mean()
        
        # 2. Coverage Loss (optional)
        coverage_loss = torch.tensor(0.0, device=device)
        if self.coverage_weight > 0.0 and attn_weights is not None:
            # Simple coverage: penalize repeated attention
            coverage_loss = self._compute_coverage_loss(attn_weights, pad_mask)
        
        # 3. Total Loss
        total_loss = ce_loss + self.coverage_weight * coverage_loss
        
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "coverage_loss": coverage_loss,
        }
    
    def _compute_coverage_loss(
        self,
        attn_weights: torch.Tensor,  # [B, T, L]
        pad_mask: Optional[torch.Tensor] = None  # [B, T]
    ) -> torch.Tensor:
        """
        简单的coverage损失：惩罚重复注意
        """
        B, T, L = attn_weights.shape
        device = attn_weights.device
        
        coverage_loss = 0.0
        coverage = torch.zeros(B, L, device=device)
        
        for t in range(T):
            attn_t = attn_weights[:, t, :]  # [B, L]
            overlap = torch.minimum(attn_t, coverage).sum(dim=1)  # [B]
            
            if pad_mask is not None:
                mask = pad_mask[:, t].float()  # [B]
                coverage_loss += (overlap * mask).sum()
            else:
                coverage_loss += overlap.sum()
            
            coverage = coverage + attn_t
        
        # Normalize
        if pad_mask is not None:
            coverage_loss = coverage_loss / (pad_mask.float().sum() + 1e-12)
        else:
            coverage_loss = coverage_loss / (B * T)
        
        return coverage_loss


def create_paper_compliant_decoder(
    vocab_size: int,
    d_model: int = 128,
    nhead: int = 8,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    max_len: int = 512,
    with_loss: bool = True,
    label_smoothing: float = 0.0,
    coverage_weight: float = 0.0,
) -> nn.Module:
    """
    工厂函数：创建符合论文要求的单层Response Decoder
    """
    decoder = PaperCompliantResponseDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
    )
    
    if with_loss:
        return PaperCompliantDecoderWithLoss(
            decoder=decoder,
            label_smoothing=label_smoothing,
            coverage_weight=coverage_weight,
        )
    else:
        return decoder


# 用法示例和测试函数
def test_paper_compliant_decoder():
    """
    测试函数：验证解码器是否正确工作
    """
    # 参数设置
    batch_size = 2
    seq_len = 10
    emofused_len = 8
    vocab_size = 1000
    d_model = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建解码器
    decoder = create_paper_compliant_decoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=4,
        dim_feedforward=256,
        with_loss=True
    ).to(device)
    
    # 模拟输入数据
    trg_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    emofused = torch.randn(batch_size, emofused_len, d_model, device=device)
    src_token_ids = torch.randint(0, vocab_size, (batch_size, emofused_len), device=device)
    gold_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # 前向传播
    with torch.no_grad():
        output = decoder(
            trg_input_ids=trg_input_ids,
            emofused=emofused,
            src_token_ids=src_token_ids,
            gold_ids=gold_ids,
        )
    
    # 验证输出形状
    print("=== Paper-Compliant Decoder Test ===")
    print(f"Pw shape: {output['Pw'].shape}")  # [B, T, V]
    print(f"Pgen shape: {output['Pgen'].shape}")  # [B, T, V]
    print(f"Pcopy shape: {output['Pcopy'].shape}")  # [B, T, V]
    print(f"p_mix shape: {output['p_mix'].shape}")  # [B, T, 1]
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"CE Loss: {output['ce_loss'].item():.4f}")
    print("Test passed!")


if __name__ == "__main__":
    test_paper_compliant_decoder()