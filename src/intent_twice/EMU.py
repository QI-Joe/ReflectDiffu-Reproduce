import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class EMUConfig:
    def __init__(self, 
        hidden_dim: int,      # D
        intent_vocab_size: int,  # C = 32
        emotion_vocab_size: int = 32,
        diffusion_steps: int = 20,
        beta_start: float = 1e-4,
        beta_end: float = 5e-2,
        use_cross_attn: bool = False,   # True 用标准多头注意力融合；False 用简化门控
        n_heads: int = 4
    ):
        self.D = hidden_dim
        self.K = intent_vocab_size
        self.T = diffusion_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.use_cross_attn = use_cross_attn
        self.n_heads = n_heads
        
class IntentAwareCVAE(nn.Module):
    def __init__(self, D: int, K: int, hid: int=1024, t_dim: int = 64):
        """
        K is emotion dimension, should be the dim for Intent_first
        """
        super().__init__()
        self.D, self.K, self.t_dim = D, K, t_dim
        
        self.enc = nn.Sequential(
            nn.Linear(self.D + self.K + self.t_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
        
        self.enc_mu = nn.Linear(hid, self.D)
        self.enc_logvar = nn.Linear(hid, self.D)
        
        self.dec = nn.Sequential(
            nn.Linear(self.D + self.K + self.t_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, self.D)
        )
        
        self.time_mlp = nn.Linear(1, t_dim) # first set as 1 layer only, in prevent of overfitting
        
    def forward(self, Q_t: torch.Tensor, t: torch.Tensor, intentfirstL: torch.Tensor) -> Tuple[torch.Tensor]:
        # B, D = Q_t.shape
        time_emb = self.time_mlp(t.unsqueeze(-1).float())  # [B, t_dim]
        
        enc_in = torch.cat([Q_t, intentfirstL, time_emb], dim=-1)  # [B, D+K+t_dim]
        h = self.enc(enc_in)  # [B, hid]
        mu = self.enc_mu(h)  # [B, D]
        
        logvar = self.enc_logvar(h)  # [B, D]
        std = torch.exp(0.5 * logvar)  # [B, D]
        eps = torch.randn_like(std)  # [B, D]
        z = mu + eps * std  # Reparameterization trick [B, D]
        
        dec_in = torch.cat([z, intentfirstL, time_emb], dim=-1)  # [B, D+K+t_dim]
        epsilon_hat = self.dec(dec_in)  # [B, D]
        return mu, logvar, epsilon_hat

class EMU(nn.Module):
    """
    EMU: Emotion-Contagion Q Module for Intent-aware CVAE
    """
    def __init__(self, cfg: EMUConfig):
        super().__init__()
        self.cfg = cfg
        self.pos_cvae = IntentAwareCVAE(cfg.D, cfg.K)
        self.neg_cvae = IntentAwareCVAE(cfg.D, cfg.K)

        # 预计算beta schedule
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)  # [T]
        self.register_buffer("alphas_cumprod", alphas_cumprod)  # [T]
        
        self.attn_pos = nn.MultiheadAttention(cfg.D, cfg.n_heads, batch_first=True)
        self.attn_neg = nn.MultiheadAttention(cfg.D, cfg.n_heads, batch_first=True)
        self.out_proj = nn.Linear(2*cfg.D, cfg.D)

    def q_sample(self, Q0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向加噪: Q_t = sqrt(alpha_bar_t) Q_0 + sqrt(1-alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(Q0)
        alpha_bar_t = self.alphas_cumprod[t]  # [B]
        sqrt_ab = torch.sqrt(alpha_bar_t).unsqueeze(-1)     # [B,1]
        sqrt_one_minus = torch.sqrt(1 - alpha_bar_t).unsqueeze(-1)
        return sqrt_ab * Q0 + sqrt_one_minus * noise  # [B, D]
    
    def p_sample_step(self, cvae: IntentAwareCVAE, Q_t: torch.Tensor, t: torch.Tensor, intentfirst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        反向一步: Q_{t-1} = 1/sqrt(alpha_t) * (Q_t - (1-alpha_t) * epsilon_hat / sqrt(1-alpha_bar_t)) + sigma_t * z
        教学简化版：用论文(7)的结构近似
        """
        beta_t = self.betas[t]  # [B]
        # 预测噪声
        mu, logvar, eps_hat = cvae(Q_t, t, intentfirst)

        alpha_bar_t = self.alphas_cumprod[t]
        one_minus_alpha_bar_t = 1 - alpha_bar_t
        # 近似论文式(7)
        coef = 1.0 / torch.sqrt(1.0 - beta_t)
        coef = coef.unsqueeze(-1)  # [B,1]
        numer = Q_t - (beta_t.unsqueeze(-1) * eps_hat) / torch.sqrt(one_minus_alpha_bar_t).unsqueeze(-1)

        Q_prev = coef * numer
        return Q_prev, (mu, logvar)

    def forward(self, Q: torch.Tensor, intentfirst: torch.Tensor, is_pos_mask: torch.Tensor, H: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Q: [B, D]
        intentfirst: [B, K]
        is_pos_mask: [B] bool, True表示走正向子模型, 否则负向
        H: [B, T, D] 上下文
        返回:
          Emofused: [B, D]
          stats: KL等统计
        """
        device = Q.device
        B, D = Q.shape
        T = self.cfg.T

        # 随机挑一组时间步用于训练（DDPM标准训练会随机t；推理用逐步）
        t = torch.randint(low=0, high=T, size=(B,), device=device)

        # 加噪
        Qt = self.q_sample(Q, t)

        # 选择子模型
        cvae_pos = self.pos_cvae
        cvae_neg = self.neg_cvae

        # === Plan A: 简单循环多步近似 ===
        # 思路：重复使用同一随机 t，对 Qt 迭代多次 p_sample_step；
        # 这是最小侵入式改造，不追求严格的时间序列递减，仅做多次残差精炼。
        n_steps = B
        Qpos_cur = Qt
        Qneg_cur = Qt
        kld_pos_list = []
        kld_neg_list = []
        mu_pos = logvar_pos = mu_neg = logvar_neg = None  # 占位便于后续引用最后一步

        for i in range(n_steps):
            Qpos_cur, (mu_pos, logvar_pos) = self.p_sample_step(cvae_pos, Qpos_cur, t, intentfirst)
            Qneg_cur, (mu_neg, logvar_neg) = self.p_sample_step(cvae_neg, Qneg_cur, t, intentfirst)
            # 逐步 KL
            kld_pos_step = -0.5 * torch.sum(1 + logvar_pos - mu_pos.pow(2) - logvar_pos.exp(), dim=-1)  # [B]
            kld_neg_step = -0.5 * torch.sum(1 + logvar_neg - mu_neg.pow(2) - logvar_neg.exp(), dim=-1)  # [B]
            kld_pos_list.append(kld_pos_step)
            kld_neg_list.append(kld_neg_step)

        # 聚合 KL：均值保持尺度与单步接近（兼容原 loss 权重）
        if len(kld_pos_list) == 1:
            kld_pos = kld_pos_list[0]
            kld_neg = kld_neg_list[0]
        else:
            kld_pos = torch.stack(kld_pos_list, dim=0).mean(0)  # [B]
            kld_neg = torch.stack(kld_neg_list, dim=0).mean(0)

        # 使用最后一次迭代得到的 Qpos_cur / Qneg_cur 作为精炼结果
        Qprev_pos = Qpos_cur
        Qprev_neg = Qneg_cur

        # 构造 Emopos / Emoneg：保持原有 mask 回退逻辑（未选中极性仍用原 Q）
        Emopos = torch.where(is_pos_mask.unsqueeze(-1), Qprev_pos, Q)  # [B, D]
        Emoneg = torch.where(~is_pos_mask.unsqueeze(-1), Qprev_neg, Q) # [B, D]

        # Cross-Attention 融合
        qpos = Emopos.unsqueeze(1)  # [B,1,D]
        qneg = Emoneg.unsqueeze(1)  # [B,1,D]
        attn_pos, _ = self.attn_pos(qpos, H, H)  # [B,1,D]
        attn_neg, _ = self.attn_neg(qneg, H, H)
        fused = torch.cat([attn_pos.squeeze(1), attn_neg.squeeze(1)], dim=-1)  # [B, 2D]
        Emofused = self.out_proj(fused)  # [B, D]

        stats = {
            "emopos": Emopos,
            "emoneg": Emoneg,
            "kldpos": kld_pos,
            "kldneg": kld_neg
        }
        return Emofused, stats
        