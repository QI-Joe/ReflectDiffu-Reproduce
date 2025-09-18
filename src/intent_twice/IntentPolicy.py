import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentPolicy(nn.Module):
    def __init__(self, state_dim: int, intent_embed: nn.Embedding, refer_map: dict, hidden: int = 512, tau: float = 1.0):
        super().__init__()
        self.intent_embed = intent_embed  # [K, I]
        self.refer_map = refer_map        # group_id -> [top3 intent ids]
        self.tau = tau

        # 状态+候选拼接打分
        in_dim = state_dim + intent_embed.embedding_dim
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # baseline for REINFORCE
        self.register_buffer("baseline", torch.tensor(0.0))

    def forward(self, Emopos, Emoneg, Emofused, group_ids, is_pos_mask):
        """
        Emopos/Emoneg/Emofused: [B, D]
        group_ids: [B]  -> intentrefer candidates
        is_pos_mask: [B] bool
        """
        device = Emofused.device
        B, D = Emofused.size()
        # 取候选id
        cand_ids = torch.stack([
            torch.tensor(self.refer_map[int(g.item())], device=device, dtype=torch.long)
            for g in group_ids
        ])  # [B, 3]
        cand_emb = self.intent_embed(cand_ids)  # [B, 3, I]

        # 拼接 [Emofused; cand_emb_i] 对每个候选打分
        s = Emofused.unsqueeze(1).expand(-1, cand_emb.size(1), -1)  # [B, 3, D]
        feats = torch.cat([s, cand_emb], dim=-1)  # [B, 3, D+I]
        logits = self.scorer(feats).squeeze(-1)   # [B, 3]
        pact = F.softmax(logits, dim=-1)         # [B, 3]

        # 采样一个动作
        chosen_idx = torch.multinomial(pact, 1).squeeze(-1)  # [B]
        chosen_intent_ids = cand_ids.gather(1, chosen_idx.unsqueeze(-1)).squeeze(-1)  # [B]

        # 奖励：sigmoid(Emopos/Emoneg dot e_intent)
        e_vec = self.intent_embed(chosen_intent_ids)  # [B, I]
        # 若 I != D，可加线性投影对齐；此处假设 I==D
        dot_pos = (Emopos * e_vec).sum(-1) / self.tau
        dot_neg = (Emoneg * e_vec).sum(-1) / self.tau
        reward = torch.where(is_pos_mask, torch.sigmoid(dot_pos), torch.sigmoid(dot_neg))  # [B]

        # baseline更新
        with torch.no_grad():
            self.baseline = 0.9 * self.baseline + 0.1 * reward.mean()

        adv = reward - self.baseline
        logp = F.log_softmax(logits, dim=-1)
        chosen_logp = logp.gather(1, chosen_idx.unsqueeze(-1)).squeeze(-1)

        Lpolicy = -(chosen_logp * adv.detach()).mean()
        Lintent = F.cross_entropy(logits, chosen_idx)

        return {
            "logits": logits,               # [B,3]
            "pact": pact,                   # [B,3]
            "chosen_idx": chosen_idx,       # [B]
            "chosen_intent_ids": chosen_intent_ids,  # [B]
            "reward": reward.mean().item(),
            "Lpolicy": Lpolicy,
            "Lintent": Lintent
        }
        
