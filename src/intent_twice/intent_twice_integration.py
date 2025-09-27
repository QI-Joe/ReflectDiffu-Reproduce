"""
Intent Twice Integration Module - Standalone

One-file, minimal integration of Intent Twice as specified in
markdown/intent-twice-intergrate.md. This module avoids tight imports and
accepts existing encoder/EMU/IntentPolicy modules at runtime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from src.intent_twice.EMU import EMU
from src.intent_twice.intent_emotion_capture import get_batch_integrator

logger = logging.getLogger(__name__)
BATCH_INTEGRATOR = get_batch_integrator()

class IntentTwiceConfig:
    """Configuration for Intent Twice integration."""
    
    def __init__(
        self,
        model_dim: int = 768,
        emotion_dim: int = 32,
        intent_vocab_size: int = 100,
        intent_embed_dim: int = 768,
        diffusion_steps: int = 20,
        policy_hidden: int = 512,
        tau: float = 1.0,
        beta_start: float = 1e-4,
        beta_end: float = 5e-2,
        alpha: float = 0.5
    ):
        self.model_dim = model_dim
        self.emotion_dim = emotion_dim
        self.intent_vocab_size = intent_vocab_size
        self.intent_embed_dim = intent_embed_dim
        self.diffusion_steps = diffusion_steps
        self.policy_hidden = policy_hidden
        self.tau = tau
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.alpha = alpha

class EmotionMappings:
    """
    Emotion -> fine-grained emotion-group -> Top-3 Intentrefer
    与论文 Table 1 对齐的工程化映射。
    注意：下面的意图名称->id 需要与你项目的 intent 词表严格一致！
    """

    # 32类情感（与数据集顺序一致，保持你现有的列表不变）
    EMOTIONS = [
        "sentimental", "afraid", "proud", "faithful", "terrified", "joyful", 
        "angry", "sad", "jealous", "hopeful", "prepared", "embarrassed",
        "excited", "annoyed", "lonely", "ashamed", "guilty", "surprised",
        "nostalgic", "confident", "furious", "disappointed", "caring", 
        "trusting", "disgusted", "anticipating", "anxious", "grateful",
        "impressed", "apprehensive", "devastated", "content"
    ]

    # 全量意图词表（示例，需与你项目真实意图词表一一对应）
    INTENTS = [
        "acknowledging", "encouraging", "neutral",
        "sympathizing", "consoling", "suggesting", "wishing",
        "agreeing", "questioning"  # 如果你的任务里有其它意图，请补齐
    ]
    INTENT2ID: Dict[str, int] = {name: i for i, name in enumerate(INTENTS)}

    # 定义情感组（按论文Table 1；faithful重复出现，工程上需选一组）
    GROUPS = {
        # G0: surprised, proud, impressed, nostalgic, trusting, faithful, prepared
        0: ["surprised", "proud", "impressed", "nostalgic", "trusting", "faithful", "prepared"],

        # G1: excited, confident, joyful, grateful, content, caring  (faithful 在G0已归属)
        1: ["excited", "confident", "joyful", "grateful", "content", "caring"],

        # G2: angry, disappointed
        2: ["disgusted", "furious", "devastated", "angry", "disappointed", "annoyed", "ashamed"],

        # G3: hopeful, sentimental
        3: ["hopeful", "sentimental"],

        # G4: anticipating, lonely, afraid, anxious, guilty, embarrassed, sad,
        #     apprehensive, terrified, jealous
        4: ["anticipating", "lonely", "afraid", "anxious", "guilty", "embarrassed",
            "sad", "apprehensive", "terrified", "jealous"],
    }

    # 每组的 Top-3 Intentrefer（把名称映射为 id）
    REFER_BY_GROUP_NAMES = {
        0: ["acknowledging", "encouraging", "questioning"],
        1: ["encouraging", "sympathizing", "acknowledging"],
        2: ["consoling", "suggesting", "encouraging"],
        3: ["encouraging", "wishing", "consoling"],
        4: ["agreeing", "encouraging", "neutral"],
    }
    # NOTE: Can't compute REFER_BY_GROUP here using a comprehension because
    # comprehensions within class scope don't see class-level names in Python 3.
    # We'll assign it right after the class definition.
    REFER_BY_GROUP: Dict[int, List[int]]

    # polarity（正/负）仍可保留，用于选择 Emopos/Emoneg
    POSITIVE = set(["surprised", "proud", "impressed", "nostalgic", "trusting", "faithful", "prepared",
                    "excited", "confident", "joyful", "grateful", "content", "caring", "hopeful", "sentimental"])
    # 其他为负面
    @classmethod
    def emotion_name(cls, eid: int) -> str:
        return cls.EMOTIONS[eid]

    @classmethod
    def get_group_id(cls, emotion_id: int) -> int:
        name = cls.emotion_name(emotion_id)
        for gid, members in cls.GROUPS.items():
            if name in members:
                return gid
        # 若未匹配到（不应发生），可回退到大负面组G4
        return 4

    @classmethod
    def get_refer_candidates(cls, emotion_id: int) -> List[int]:
        gid = cls.get_group_id(emotion_id)
        return cls.REFER_BY_GROUP[gid]

    @classmethod
    def is_positive(cls, emotion_id: int) -> bool:
        return cls.emotion_name(emotion_id) in cls.POSITIVE
    
    @classmethod
    def get_emotion_group_id(cls, emotion_name: str):
        for grd, members in cls.GROUPS.items():
            if emotion_name in members:
                return grd

# Post-class initialization that depends on class attributes
EmotionMappings.REFER_BY_GROUP = {
    gid: [EmotionMappings.INTENT2ID[n] for n in names]
    for gid, names in EmotionMappings.REFER_BY_GROUP_NAMES.items()
}

class IntentTwiceModule(nn.Module):
    """
    Complete Intent Twice integration module.
    
    Implements the full pipeline:
    1. Emotion-Contagion encoding (Q, p)
    2. EMU sampling (Emopos, Emoneg, Emofused)  
    3. IntentPolicy action selection
    4. Loss computation and backpropagation
    """
    
    def __init__(
        self,
        config: IntentTwiceConfig,
        encoder_module: Optional[nn.Module] = None,
        emu_module: Optional[nn.Module] = None,
        intent_policy_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        # external modules can be provided here or at forward time
        self.encoder_module = encoder_module
        self.emu_module = emu_module
        self.intent_policy_module = intent_policy_module

        # Initialize intent embeddings (shared with policy)
        self.intent_embed = nn.Embedding(config.intent_vocab_size, config.intent_embed_dim)
        # Policy head will be provided by user; intent_embed is accessible via intent_policy_module.intent_embed
        
        # Projection layers for dimension alignment
        if config.intent_embed_dim != config.model_dim:
            self.intent_proj = nn.Linear(config.intent_embed_dim, config.model_dim)
        else:
            self.intent_proj = nn.Identity()
    
    def compute_masks_and_groups(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute is_pos_mask and group_ids from emotion probability distribution.
        
        Args:
            p: Emotion probability distribution [B, E]
            
        Returns:
            group_ids: [B] tensor of group IDs
            is_pos_mask: [B] boolean tensor for positive/negative polarity
        """
        device = p.device
        batch_size = p.size(0)
        
        # Get top-1 emotion IDs
        emo_ids = p.argmax(dim=-1)  # [B]
        
        # Compute group IDs and polarity masks
        group_ids = torch.tensor([
            EmotionMappings.get_group_id(int(emo_id.item())) 
            for emo_id in emo_ids
        ], device=device, dtype=torch.long)
        
        # Use EmotionMappings.is_positive to determine polarity
        is_pos_mask = torch.tensor([
            EmotionMappings.is_positive(int(emo_id.item())) for emo_id in emo_ids
        ], device=device, dtype=torch.bool)
        
        return group_ids, is_pos_mask
    
    def forward(
        self,
        encoder_out: Dict[str, torch.Tensor],
        intent_out: Dict[str, torch.Tensor],
        return_components: bool = False,
        emu_module: EMU = None,
        intent_policy_module: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Complete Intent Twice forward pass.
        
        Args:
            tokens: Token sequences [B, L] (list of strings)
            label_ids: Reason label IDs [B, L]  
            attention_mask: Attention mask [B, L]
            h_tilde: ERA reasoning representation [B, L, D_era] (optional)
            return_components: Whether to return intermediate components
            
        Returns:
            Dictionary containing:
            - Q: Global context summary [B, D]
            - p: Emotion probability distribution [B, E]
            - Emofused: Fused emotion representation [B, D] 
            - chosen_intent_vec: Selected intent vector [B, D]
            - Ltwice: Intent Twice loss (scalar)
            - chosen_intent_ids: Selected intent IDs [B]
            - monitor: Dictionary of monitoring metrics
        """
        device = encoder_out["Q"].device
               
        Q = encoder_out["Q"]  # [B, D]
        H = encoder_out["H"]  # [B, L, D]
        # P may be the Contrastive Expert output; if it already matches emotion_dim, treat as probs
        p_semantic = encoder_out["P_semantic"].to(device)  # [B, intent_dim]
        p_intent = intent_out["p_intent"]
        p = p_semantic + self.config.alpha * p_intent  # [B, intent_dim == 9]
        BATCH_INTEGRATOR.add("final_p", p.detach().cpu().tolist())
        
        # ==================== Step 2: Compute Masks and Groups ====================
        group_ids, is_pos_mask = self.compute_masks_and_groups(p)
        
        # ==================== Step 3: EMU Sampling ====================
        # Create intent_first from emotion distribution (simplified)
        intent_first = p  # [B, E] - use emotion dist as intent conditioning
        
        # EMU forward pass
        emu = emu_module or self.emu_module
        if emu is None:
            raise ValueError("emu_module must be provided either in __init__ or forward")

        Emofused, emu_stats = emu.forward(
            Q=Q,
            intentfirst=intent_first,
            is_pos_mask=is_pos_mask,
            H=H
        )
        
        Emopos = emu_stats["emopos"]  # [B, D]
        Emoneg = emu_stats["emoneg"]  # [B, D]
        
        # Compute KL losses from EMU stats (averaged over batch)
        Lklpos = emu_stats.get("kldpos", torch.zeros(Q.size(0), device=device))
        Lklneg = emu_stats.get("kldneg", torch.zeros(Q.size(0), device=device))
        # Ensure tensor shapes then reduce to scalars
        if isinstance(Lklpos, torch.Tensor):
            Lklpos = Lklpos.mean()
        if isinstance(Lklneg, torch.Tensor):
            Lklneg = Lklneg.mean()
        
        # ==================== Step 4: Intent Policy ====================
        pol = intent_policy_module or self.intent_policy_module
        if pol is None:
            raise ValueError("intent_policy_module must be provided either in __init__ or forward")

        policy_out = pol(
            Emopos=Emopos,
            Emoneg=Emoneg, 
            Emofused=Emofused,
            group_ids=group_ids,
            is_pos_mask=is_pos_mask
        )
        
        chosen_intent_ids = policy_out["chosen_intent_ids"]  # [B]
        Lpolicy = policy_out["Lpolicy"]
        Lintent = policy_out["Lintent"]
        
        # ==================== Step 5: Intent Vector Generation ====================
        chosen_intent_emb = self.intent_embed(chosen_intent_ids)  # [B, I]
        chosen_intent_vec = self.intent_proj(chosen_intent_emb)   # [B, D]

        # ==================== Step 6: Loss Computation ====================
        # Follow loss-compute.md: L_twice = L_kl_pos + L_kl_neg + L_intent
        # (policy RL term can be optimized separately, not added by default)
        Ltwice = Lklpos + Lklneg + Lintent
        
        # ==================== Return Results ====================
        results = {
            "Q": Q,
            "p": p,
            "Emofused": Emofused,
            "chosen_intent_vec": chosen_intent_vec,
            "chosen_intent_ids": chosen_intent_ids,
            "group_ids": group_ids,
            "is_pos_mask": is_pos_mask,
            "Ltwice": Ltwice,
            "monitor": {
                "reward": policy_out["reward"],
                "Lklpos": float(Lklpos),
                "Lklneg": float(Lklneg),
                "Lintent": float(Lintent),
                "Lpolicy": float(Lpolicy)
            }
        }
        
        if return_components:
            results.update({
                "H": H,
                "Emopos": Emopos,
                "Emoneg": Emoneg,
                "policy_logits": policy_out["logits"],
                "policy_probs": policy_out["pact"]
            })
        
        return results


def train_step(
    intent_twice_module: IntentTwiceModule,
    batch: Dict[str, torch.Tensor],
    decoder_loss_fn: callable,
    delta: float = 1.0,
    zeta: float = 1.0, 
    eta: float = 1.0
) -> Dict[str, float]:
    """
    Complete training step with Intent Twice integration.
    
    Args:
        intent_twice_module: The integrated Intent Twice module
        batch: Training batch with tokens, label_ids, attention_mask
        decoder_loss_fn: Function to compute decoder loss Lres
        delta, zeta, eta: Loss weighting coefficients
        
    Returns:
        Dictionary of losses and metrics
    """
    # 0) Forward: Intent Twice
    itw = intent_twice_module(
        tokens=batch["tokens"],
        label_ids=batch["label_ids"], 
        attention_mask=batch["attention_mask"],
        h_tilde=batch.get("h_tilde", None)
    )
    
    # 1) Emotion classification loss (L_em = CE + NT-Xent)
    Q = itw["Q"]
    p = itw["p"]
    device = Q.device
    Lem_ce = torch.tensor(0.0, device=device)
    Lem_ntx = torch.tensor(0.0, device=device)
    if "emotion_labels" in batch:
        Lem_ce = F.cross_entropy(p, batch["emotion_labels"])  # [B,E] vs [B]
        # If encoder exposes NT-Xent loss, use it over Q
        encoder = intent_twice_module.encoder_module
        if hasattr(encoder, "loss") and callable(getattr(encoder, "loss")):
            Lem_ntx = encoder.loss(Q, batch["emotion_labels"])  # scalar
    Lem = Lem_ce + Lem_ntx

    # 2) Intent Twice loss
    Ltwice = itw["Ltwice"]
    
    # 3) Response decoding loss (optional)
    Lres = torch.tensor(0.0, device=device)
    if decoder_loss_fn is not None and all(k in batch for k in ["trg_input_ids", "gold_response_ids", "src_token_ids"]):
        # Build memory for decoder: use encoder H with Emofused fuse (inline to avoid import cycles)
        enc = intent_twice_module.encoder_module
        # Recompute encoder to get H; we already ran it within module, but we don't have H here; run lightweight forward
        enc_out = enc(
            tokens=batch["tokens"],
            label_ids=batch["label_ids"],
            attention_mask=batch["attention_mask"],
            h_tilde=batch.get("h_tilde", None)
        )
        H = enc_out["H"]  # [B, L, D]
        Emofused = itw["Emofused"]  # [B, D]
        memory = H + Emofused.unsqueeze(1)  # [B, L, D]

        # Decoder module and loss fn are provided by caller; compute Pw inside fn
        dec_out = decoder_loss_fn(
            trg_input_ids=batch["trg_input_ids"],
            memory=memory,
            src_token_ids=batch["src_token_ids"],
            src_key_padding_mask=batch.get("src_key_padding_mask", batch["attention_mask"] == 0),
            tgt_key_padding_mask=batch.get("tgt_key_padding_mask", None),
            gold_ids=batch["gold_response_ids"],
        )
        if isinstance(dec_out, dict) and "loss" in dec_out:
            Lres = dec_out["loss"]
        elif torch.is_tensor(dec_out):
            Lres = dec_out

    # 4) Total loss
    L = delta * Lem + zeta * Ltwice + eta * Lres
    
    out: Dict[str, float] = {
        "loss": float(L),
        "Lem": float(Lem),
        "Lem_ce": float(Lem_ce),
        "Lem_ntx": float(Lem_ntx),
        "Ltwice": float(Ltwice),
    }
    if torch.is_tensor(Lres):
        out["Lres"] = float(Lres)
    out.update(itw["monitor"])  # reward, KLs, etc.
    return out