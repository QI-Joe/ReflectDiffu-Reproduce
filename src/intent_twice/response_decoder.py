"""
Response Decoder with Pointer-Generator for ReflectDiffu

Implements Section 3.3.2 (Response Decoding) per markdown/resp[on]de-decoder.md:
- Transformer decoder with causal self-attn and cross-attn over Emofused memory
- Pointer-generator mixture of generation and copy distributions
- Optional coverage loss

Inputs (training):
- trg_input_ids: [B, T] teacher-forced target tokens (shifted right)
- memory: [B, Sctx, D] fused context for cross-attention (e.g., H + Emofused broadcast)
- src_token_ids: [B, Ssrc] source token ids to scatter copy attn to vocab/extended vocab
- src_key_padding_mask: [B, Sctx] True for padding in memory
- tgt_key_padding_mask: [B, T] True for padding in targets
- extended_vocab_size: Optional[int] if using per-batch extended vocab

Outputs:
- dict with Pw [B, T, Vext], Pgen, Pcopy_ext, p_mix, attn_weights, hidden

Loss:
- ResponseLoss computes token-level CE over Pw and optional coverage loss
"""

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Causal mask for decoder self-attention: [T, T] with -inf above diagonal."""
    mask = torch.full((sz, sz), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (batch-first)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)


class DecoderBlock(nn.Module):
    """Pre-LN Transformer decoder block returning cross-attn weights."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,  # [B, T, D]
        memory: torch.Tensor,  # [B, S, D]
        tgt_mask: Optional[torch.Tensor] = None,  # [T, T]
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T]
        memory_key_padding_mask: Optional[torch.Tensor] = None,  # [B, S]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pre-LN self-attention
        y = self.norm1(x)
        y, _ = self.self_attn(y, y, y, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = x + self.dropout1(y)

        # Pre-LN cross-attention
        y = self.norm2(x)
        y, attn_weights = self.cross_attn(
            y, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=True, average_attn_weights=False
        )  # attn_weights: [B, heads, T, S]
        x = x + self.dropout2(y)

        # Feed-forward
        y = self.norm3(x)
        y = self.linear2(self.dropout(F.relu(self.linear1(y))))
        x = x + self.dropout3(y)
        return x, attn_weights, y


class ResponseDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.out_proj.weight = self.token_embed.weight  # weight tying

        # Pointer-generator gate: sigmoid(linear([h_t, ctx_t, x_t])) -> [B,T,1]
        self.gate = nn.Linear(3 * d_model, 1)

    def forward(
        self,
        trg_input_ids: torch.Tensor,  # [B, T]
        memory: torch.Tensor,  # [B, Sctx, D]
        src_token_ids: torch.Tensor,  # [B, Ssrc] for scatter
        src_key_padding_mask: Optional[torch.Tensor] = None,  # [B, Sctx]
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T]
        extended_vocab_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        device = trg_input_ids.device
        B, T = trg_input_ids.size()

        # Embedding + positional
        x = self.token_embed(trg_input_ids)  # [B, T, D]
        x = self.pos_enc(x)

        # Build causal mask once
        tgt_mask = _generate_square_subsequent_mask(T, device)

        # Pass through decoder blocks, track last cross-attn weights and context
        h = x
        last_attn: Optional[torch.Tensor] = None
        last_ctx: Optional[torch.Tensor] = None
        pre = h
        for layer in self.layers:
            h, attn_weights, _ = layer(
                h, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            last_attn = attn_weights  # [B, H, T, S]
            # Preserve a copy of h to compute delta if needed
            # We'll compute context from attention weights below
            pre = h

        # Hidden states
        Hdec = h  # [B, T, D]

        # Generation distribution over base vocab
        logits_gen = self.out_proj(Hdec)  # [B, T, V]
        Pgen = F.softmax(logits_gen, dim=-1)

        # Copy attention over source tokens: average heads -> [B, T, S]
        if last_attn is None:
            raise RuntimeError("Decoder produced no attention weights")
        CopyAttn = last_attn.mean(dim=1)  # [B, T, S]

        # Context vectors: sum_j attn_tj * memory_j
        # last_attn: [B, H, T, S], average heads -> [B, T, S]
        attn_avg = last_attn.mean(dim=1)  # [B, T, S]
        context_vec = torch.bmm(attn_avg, memory)  # [B, T, D]

        # Mixture gate p_mix
        gate_inp = torch.cat([Hdec, context_vec, x], dim=-1)  # [B, T, 3D]
        p_mix = torch.sigmoid(self.gate(gate_inp))  # [B, T, 1]

        # Scatter copy attn to extended vocab
        Vext = extended_vocab_size if extended_vocab_size is not None else self.vocab_size
        Pcopy_ext = scatter_copy_to_vocab(CopyAttn, src_token_ids, Vext)  # [B, T, Vext]

        # Expand Pgen to extended vocab if needed
        if Vext == self.vocab_size:
            Pgen_ext = Pgen
        else:
            Pgen_ext = torch.zeros(B, T, Vext, device=device, dtype=Pgen.dtype)
            Pgen_ext[:, :, : self.vocab_size] = Pgen

        # Final mixture
        Pw = p_mix * Pgen_ext + (1.0 - p_mix) * Pcopy_ext  # [B, T, Vext]

        return {
            "Pw": Pw,
            "Pgen": Pgen_ext,
            "Pcopy_ext": Pcopy_ext,
            "p_mix": p_mix,
            "CopyAttn": CopyAttn,
            "Hdec": Hdec,
        }


class DecoderWithLoss(nn.Module):
    """Convenience wrapper to compute Pw and Lres in one call for training."""

    def __init__(self, decoder: ResponseDecoder, coverage_weight: float = 0.0, label_smoothing: float = 0.0):
        super().__init__()
        self.decoder = decoder
        self.criterion = ResponseLoss(coverage_weight=coverage_weight, label_smoothing=label_smoothing)

    def forward(
        self,
        trg_input_ids: torch.Tensor,
        memory: torch.Tensor,
        src_token_ids: torch.Tensor,
        gold_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        extended_vocab_size: Optional[int] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.decoder(
            trg_input_ids=trg_input_ids,
            memory=memory,
            src_token_ids=src_token_ids,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            extended_vocab_size=extended_vocab_size,
        )
        loss_dict = self.criterion(
            Pw=out["Pw"], gold_ids=gold_ids, pad_mask=pad_mask if pad_mask is not None else (tgt_key_padding_mask == False if tgt_key_padding_mask is not None else None),
            CopyAttn=out.get("CopyAttn")
        )
        return {**out, **loss_dict}


def scatter_copy_to_vocab(copy_attn: torch.Tensor, src_token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Project copy attention over source positions [B,T,Ssrc] to vocab distribution [B,T,Vext]
    using scatter-add based on src_token_ids [B,Ssrc]. If src_token_ids already refer to an
    extended vocab, set vocab_size accordingly.
    """
    B, T, S = copy_attn.shape
    device = copy_attn.device
    out = torch.zeros(B, T, vocab_size, device=device, dtype=copy_attn.dtype)
    # Expand indices to [B,T,S]
    idx = src_token_ids.unsqueeze(1).expand(-1, T, -1)  # [B,T,S]
    out.scatter_add_(dim=2, index=idx, src=copy_attn)
    return out


class ResponseLoss(nn.Module):
    def __init__(self, coverage_weight: float = 0.0, label_smoothing: float = 0.0):
        super().__init__()
        self.coverage_weight = coverage_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        Pw: torch.Tensor,  # [B, T, V]
        gold_ids: torch.Tensor,  # [B, T] indices in same vocab space as Pw
        pad_mask: Optional[torch.Tensor] = None,  # [B, T] 1 for valid, 0 for pad or bool
        CopyAttn: Optional[torch.Tensor] = None,  # [B, T, S]
    ) -> Dict[str, torch.Tensor]:
        B, T, V = Pw.shape
        device = Pw.device

        # Cross-entropy over extended vocab
        # Gather probs at gold indices; add epsilon for stability
        eps = 1e-12
        gold_ids_clamped = gold_ids.clamp(min=0, max=V - 1)
        Pw_flat = Pw.view(B * T, V)
        idx_flat = gold_ids_clamped.view(-1, 1)
        gold_prob = Pw_flat.gather(1, idx_flat).view(B, T).clamp_min(eps)  # [B, T]
        nll = -torch.log(gold_prob)

        if pad_mask is not None:
            mask = pad_mask.float()
            nll = nll * mask
            ce = nll.sum() / (mask.sum() + eps)
        else:
            ce = nll.mean()

        # Coverage loss: sum_t sum_i min(attn_ti, cov_prev_ti)
        cov_loss = torch.tensor(0.0, device=device)
        if self.coverage_weight > 0.0 and CopyAttn is not None:
            cov_prev = torch.zeros_like(CopyAttn[:, 0, :])  # [B, S]
            cov_total = 0.0
            valid_steps = 0.0
            for t in range(T):
                attn_t = CopyAttn[:, t, :]  # [B, S]
                cov_t = torch.minimum(attn_t, cov_prev).sum(dim=1)  # [B]
                if pad_mask is not None:
                    m = pad_mask[:, t].float()  # [B]
                    cov_total += (cov_t * m).sum()
                    valid_steps += m.sum()
                else:
                    cov_total += cov_t.sum()
                    valid_steps += float(B)
                cov_prev = cov_prev + attn_t
            cov_loss = cov_total / (valid_steps + eps)

        loss = ce + self.coverage_weight * cov_loss
        return {"loss": loss, "LCE": ce, "Lcov": cov_loss}


def fuse_memory_with_state(H: torch.Tensor, Emofused: torch.Tensor) -> torch.Tensor:
    """
    Utility to create decoder memory from encoder outputs and fused state.
    Returns H + broadcast(Emofused) with shape [B, L, D].
    """
    return H + Emofused.unsqueeze(1)
