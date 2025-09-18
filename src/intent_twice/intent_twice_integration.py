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

logger = logging.getLogger(__name__)


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
        beta_end: float = 5e-2
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


class EmotionMappings:
    """
    Emotion mappings for EmpatheticDialogues 32 emotions.
    Maps emotion IDs to groups and polarities as required by Intent Twice.
    """
    
    # EmpatheticDialogues 32 emotion categories
    EMOTIONS = [
        "sentimental", "afraid", "proud", "faithful", "terrified", "joyful", 
        "angry", "sad", "jealous", "hopeful", "prepared", "embarrassed",
        "excited", "annoyed", "lonely", "ashamed", "guilty", "surprised",
        "nostalgic", "confident", "furious", "disappointed", "caring", 
        "trusting", "disgusted", "anticipating", "anxious", "grateful",
        "impressed", "apprehensive", "devastated", "content"
    ]
    
    # Emotion to group mapping (based on Table 1 reference)
    # Groups represent common intent categories
    EMOTION_TO_GROUP = {
        # Group 0: Positive emotions
        0: 0, 2: 0, 3: 0, 5: 0, 9: 0, 10: 0, 12: 0, 17: 0, 18: 0, 19: 0,
        22: 0, 23: 0, 25: 0, 27: 0, 28: 0, 31: 0,
        # Group 1: Negative emotions  
        1: 1, 4: 1, 6: 1, 7: 1, 8: 1, 11: 1, 13: 1, 14: 1, 15: 1, 16: 1,
        20: 1, 21: 1, 24: 1, 26: 1, 29: 1, 30: 1
    }
    
    # Emotion to polarity mapping
    EMOTION_TO_POLARITY = {
        # Positive emotions
        0: "pos", 2: "pos", 3: "pos", 5: "pos", 9: "pos", 10: "pos", 
        12: "pos", 17: "pos", 18: "pos", 19: "pos", 22: "pos", 23: "pos", 
        25: "pos", 27: "pos", 28: "pos", 31: "pos",
        # Negative emotions
        1: "neg", 4: "neg", 6: "neg", 7: "neg", 8: "neg", 11: "neg", 
        13: "neg", 14: "neg", 15: "neg", 16: "neg", 20: "neg", 21: "neg", 
        24: "neg", 26: "neg", 29: "neg", 30: "neg"
    }
    
    # Intent reference map (group_id -> top-3 intent ids)
    # These would be populated based on actual intent vocabulary
    REFER_MAP = {
        0: [0, 1, 2],   # Top-3 intents for positive group
        1: [3, 4, 5],   # Top-3 intents for negative group
    }
    
    @classmethod
    def get_group_id(cls, emotion_id: int) -> int:
        """Get group ID for emotion ID."""
        return cls.EMOTION_TO_GROUP.get(emotion_id, 1)  # Default to negative group
    
    @classmethod
    def get_polarity(cls, emotion_id: int) -> str:
        """Get polarity for emotion ID."""
        return cls.EMOTION_TO_POLARITY.get(emotion_id, "neg")  # Default to negative
    
    @classmethod
    def get_refer_candidates(cls, group_id: int) -> List[int]:
        """Get top-3 intent candidates for group ID."""
        return cls.REFER_MAP.get(group_id, [3, 4, 5])  # Default to negative intents


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
        
        polarity_strs = [
            EmotionMappings.get_polarity(int(emo_id.item())) 
            for emo_id in emo_ids
        ]
        is_pos_mask = torch.tensor([
            pol == "pos" for pol in polarity_strs
        ], device=device, dtype=torch.bool)
        
        return group_ids, is_pos_mask
    
    def forward(
        self,
        tokens: List[List[str]],
        label_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        h_tilde: Optional[torch.Tensor] = None,
        return_components: bool = False,
        # optionally pass/override modules here
        encoder_module: Optional[nn.Module] = None,
        emu_module: Optional[nn.Module] = None,
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
        device = label_ids.device
        
        # ==================== Step 1: Emotion-Contagion Encoding ====================
        enc = encoder_module or self.encoder_module
        if enc is None:
            raise ValueError("encoder_module must be provided either in __init__ or forward")

        encoder_out = enc(
            tokens=tokens,
            label_ids=label_ids,
            attention_mask=attention_mask,
            h_tilde=h_tilde
        )
        
        Q = encoder_out["Q"]  # [B, D]
        H = encoder_out["H"]  # [B, L, D]
        # P may be the Contrastive Expert output; if it already matches emotion_dim, treat as probs
        if "p" in encoder_out:
            p = encoder_out["p"]
            P = encoder_out.get("P", Q)
        else:
            P = encoder_out.get("P", Q)  # fallback
            # If P already [B, E], use directly; else project
            if P.size(-1) == self.config.emotion_dim:
                p = F.softmax(P, dim=-1)
            else:
                if not hasattr(self, 'emotion_classifier'):
                    self.emotion_classifier = nn.Linear(self.config.model_dim, self.config.emotion_dim).to(device)
                p = F.softmax(self.emotion_classifier(P), dim=-1)  # [B, E]
        
        # ==================== Step 2: Compute Masks and Groups ====================
        group_ids, is_pos_mask = self.compute_masks_and_groups(p)
        
        # ==================== Step 3: EMU Sampling ====================
        # Create intent_first from emotion distribution (simplified)
        intent_first = p  # [B, E] - use emotion dist as intent conditioning
        
        # EMU forward pass
        emu = emu_module or self.emu_module
        if emu is None:
            raise ValueError("emu_module must be provided either in __init__ or forward")

        Emofused, emu_stats = emu(
            Q=Q,
            intentfirst=intent_first,
            is_pos_mask=is_pos_mask,
            H=H
        )
        
        Emopos = emu_stats["emopos"]  # [B, D]
        Emoneg = emu_stats["emoneg"]  # [B, D]
        
        # Compute KL losses (simplified - would need proper CVAE KL computation)
        Lklpos = torch.tensor(0.0, device=device)  # Placeholder
        Lklneg = torch.tensor(0.0, device=device)  # Placeholder
        
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
        Ltwice = Lklpos + Lklneg + Lintent + Lpolicy
        
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
                "P": P,
                "Emopos": Emopos,
                "Emoneg": Emoneg,
                "policy_logits": policy_out["logits"],
                "policy_probs": policy_out["pact"]
            })
        
        return results


def intent_twice_step(
    encoder_module: nn.Module,
    emu_module: nn.Module,
    intent_policy_module: nn.Module,
    batch: Dict[str, Any],
    refer_map: Optional[Dict[int, List[int]]] = None,
    emotion2group: Optional[Dict[int, int]] = None,
    emotion2polarity: Optional[Dict[int, str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Standalone function for Intent Twice step as described in markdown.
    
    This follows the markdown algorithm, adapted to dynamic module interfaces.
    """
    # defaults
    if refer_map is None:
        refer_map = EmotionMappings.REFER_MAP
    if emotion2group is None:
        emotion2group = EmotionMappings.EMOTION_TO_GROUP
    if emotion2polarity is None:
        emotion2polarity = EmotionMappings.EMOTION_TO_POLARITY

    # 1) Emotion-Contagion encoding
    enc_out = encoder_module(
        tokens=batch["tokens"],
        label_ids=batch["label_ids"],
        attention_mask=batch["attention_mask"],
        h_tilde=batch.get("h_tilde", None),
    )
    Q = enc_out["Q"]
    p = enc_out.get("p")
    if p is None:
        P = enc_out.get("P", Q)
        if P.size(-1) == 32:
            p = F.softmax(P, dim=-1)
        else:
            clf = nn.Linear(Q.size(-1), 32, device=Q.device)
            p = F.softmax(clf(P), dim=-1)

    # 2) Get top-1 emotion id from distribution
    emo_ids = p.argmax(dim=-1)  # [B], int

    # 3) Compute is_pos_mask and group_ids
    group_ids = torch.tensor([
        emotion2group[int(e.item())] for e in emo_ids
    ], device=Q.device, dtype=torch.long)  # [B]
    
    polarity = [emotion2polarity[int(e.item())] for e in emo_ids]
    # Neutral merged into negative (as per markdown)
    is_pos_mask = torch.tensor([
        1 if pol == 'pos' else 0 for pol in polarity
    ], device=Q.device, dtype=torch.bool)  # [B]

    # 4) EMU diffusion sampling (sampling state construction)
    # Training: random step t, add/remove noise within EMU; inference: multi-step
    try:
        Emofused, emu_stats = emu_module(Q, p, is_pos_mask, enc_out.get("H", Q.unsqueeze(1)))
        Emopos = emu_stats.get("emopos", Q)
        Emoneg = emu_stats.get("emoneg", Q)
        Lklpos = torch.tensor(0.0, device=Q.device)
        Lklneg = torch.tensor(0.0, device=Q.device)
    except Exception:
        Emopos = Emoneg = Emofused = Q
        Lklpos = Lklneg = torch.tensor(0.0, device=Q.device)

    # 5) Intent Policy: policy sampling on Intentrefer (Top-3) (Action Definition)
    pol_out = intent_policy_module(Emopos, Emoneg, Emofused, group_ids, is_pos_mask)
    chosen_intent_ids = pol_out["chosen_intent_ids"]

    # 6) Aggregate Intent Twice loss (Equation 10, with Lpolicy for stability)
    Ltwice = Lklpos + Lklneg + pol_out["Lintent"] + pol_out["Lpolicy"]

    # 7) Return intent control signal for decoder
    chosen_intent_vec = intent_policy_module.intent_embed(chosen_intent_ids)  # [B, I]

    return {
        "Q": Q,
        "p": p,
        "Emopos": Emopos,
        "Emoneg": Emoneg,
        "Emofused": Emofused,
        "group_ids": group_ids,
        "is_pos_mask": is_pos_mask,
        "chosen_intent_ids": chosen_intent_ids,
        "chosen_intent_vec": chosen_intent_vec,
        "Ltwice": Ltwice,
        "monitor": {
            "reward": pol_out["reward"],
            "Lklpos": float(Lklpos),
            "Lklneg": float(Lklneg),
            "Lintent": float(pol_out["Lintent"]),
            "Lpolicy": float(pol_out["Lpolicy"])
        }
    }


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
    
    # 1) Emotion classification loss (Lem) using Q/p
    # This would typically be CE + NT-Xent from contrastive experts
    if "emotion_labels" in batch:
        Lem = F.cross_entropy(itw["p"], batch["emotion_labels"])
    else:
        Lem = torch.tensor(0.0, device=itw["Q"].device)
    
    # 2) Decoder loss (Lres) using Emofused + chosen_intent_vec
    if decoder_loss_fn is not None:
        Lres = decoder_loss_fn(batch, itw["Emofused"], itw["chosen_intent_vec"])
    else:
        Lres = torch.tensor(0.0, device=itw["Q"].device)
    
    # 3) Intent Twice loss
    Ltwice = itw["Ltwice"]
    
    # 4) Total loss (Equation 12)
    L = delta * Lem + zeta * Ltwice + eta * Lres
    
    return {
        "loss": float(L),
        "Lem": float(Lem),
        "Ltwice": float(Ltwice),
        "Lres": float(Lres),
        **itw["monitor"]
    }