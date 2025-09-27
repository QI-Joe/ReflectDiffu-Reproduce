import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List, Dict, Any
import os

# ================= Import project modules =================
from src.emotion_contagion.data_processor import EmotionContagionDataProcessor
from src.emotion_contagion.config import EmotionContagionConfig
from src.emotion_contagion.encoder import EmotionContagionEncoder, EmotionClassifier, ce_loss_for_emotion
from src.emotion_contagion.foundation_emb import IntentSemanticScorer
from src.intent_twice.intent_twice_integration import IntentTwiceConfig, IntentTwiceModule, EmotionMappings
from src.intent_twice.EMU import EMUConfig, EMU
from src.intent_twice.IntentPolicy import IntentPolicy

from src.intent_twice.response_decoder import create_paper_compliant_decoder
from portable_inference import build_random_intent, get_agent, get_intent_distribution
from src.intent_twice.intent_emotion_capture import get_batch_integrator

# Inline tokenizer loader (avoids import path issues during refactor)
from src.tokenizer_loader import get_tokenizer
TOKENIZER = get_tokenizer()
BATCH_INTEGRATOR = get_batch_integrator()

def build_emotion_contagion_encoder(vocab_size: int = 5000, model_dim: int = 128, enc_cfg: Dict[str, Any] | None = None):
    """构建情感传染编码器 (now parameterized by unified config)."""
    vocab_size = len(TOKENIZER)
    enc_cfg = enc_cfg or {}
    cfg = EmotionContagionConfig(
        vocab_size=vocab_size,
        word_embedding_dim=model_dim,
        model_dim=model_dim,
        max_position_embeddings=enc_cfg.get('max_position_embeddings', 256),
        num_reason_labels=enc_cfg.get('num_reason_labels', 2),
        num_encoder_layers=enc_cfg.get('layers', 2),
        num_attention_heads=enc_cfg.get('heads', 4),
        feedforward_dim=enc_cfg.get('ff_dim', 256),
        attention_type=enc_cfg.get('attention_type', 'cross'),
        attention_dropout=enc_cfg.get('attention_dropout', 0.1),
        dropout_rate=enc_cfg.get('dropout', 0.1),
        era_hidden_dim=model_dim,
        era_projection_dim=model_dim,
    )
    return EmotionContagionEncoder(cfg, external_tokenizer=TOKENIZER)


def build_intent_twice_modules(model_dim: int = 128, config_section: Dict[str, Any] | None = None):
    """构建Intent Twice模块 (parameterized)."""
    cs = config_section or {}
    emotion_dim = cs.get('emotion_dim', 32)
    intent_vocab_size = cs.get('intent_vocab_size', 9)
    diffusion_steps = cs.get('diffusion_steps', 10)
    beta_start = cs.get('beta_start', 1e-4)
    beta_end = cs.get('beta_end', 5e-2)
    tau = cs.get('tau', 1.0)
    it_cfg = IntentTwiceConfig(
        model_dim=model_dim,
        emotion_dim=emotion_dim,
        intent_vocab_size=intent_vocab_size,
        intent_embed_dim=model_dim,
        diffusion_steps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        tau=tau
    )
    emu_cfg = EMUConfig(
        hidden_dim=model_dim,
        intent_vocab_size=intent_vocab_size,
        diffusion_steps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        use_cross_attn=cs.get('use_cross_attn', False),
        n_heads=cs.get('n_heads', 4)
    )
    emu = EMU(emu_cfg)
    refer_map = {i: [0,1,2] for i in range(5)}  # keep simple; externalize later if needed
    policy = IntentPolicy(
        state_dim=model_dim,
        intent_embed=nn.Embedding(it_cfg.intent_vocab_size, it_cfg.intent_embed_dim),
        refer_map=refer_map
    )
    return IntentTwiceModule(
        config=it_cfg,
        encoder_module=None,
        emu_module=emu,
        intent_policy_module=policy
    )



def compute_joint_loss(Lem: torch.Tensor, Ltwice: torch.Tensor, Lres: torch.Tensor, 
                      delta: float = 1.0, zeta: float = 1.0, eta: float = 1.0) -> torch.Tensor:
    """
    L = δ·Lem + ζ·Ltwice + η·Lres
    
    Returns:
        Joint loss
    """
    joint_loss = delta * Lem + zeta * Ltwice + eta * Lres
    return joint_loss

class ReflectDiffu(nn.Module):
    def __init__(self, vocab_size: int = 1000, model_dim: int = 128, device: torch.device = None,
                 use_era: bool = False, coverage_weight: float = 0.0, unified_config: Dict[str, Any] | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_era = use_era
        self.coverage_weight = coverage_weight
        self.unified_config = unified_config or {}
        
        # Placeholders for components
        self.encoder = None
        self.it_module = None
        self.decoder_with_loss = None
        self.era_model = None
        self.emotion_cls = None
        self.semantic_scorer = None
        
        # Data components
        self.dp = None
        self.raw_data = None
        self.batch_size = 4
        self.batches = []
        
    def _init_data_loader(self, data_path: str, batch_size: int = 4, max_length: int = 64):
        """Initialize data loader and create batches."""
        print("\\n=== Initializing Data Loader ===")
        self.batch_size = batch_size
        self.dp = EmotionContagionDataProcessor(max_length=max_length)
        self.raw_data = self.dp.load_data(data_path)
        
        # Process all data first (tokenizer provides fixed vocab)
        self.agent = get_agent()

        # Fixed tokenizer vocab size placeholder (not used but kept for compatibility)
        self.vocab_tokens = None
        
        # Create batches
        self.batches = []
        total_samples = len(self.raw_data)
        for i in range(0, total_samples, self.batch_size):
            end_idx = min(i+self.batch_size, total_samples)
            convs = self.raw_data[i:end_idx]
            batch_user, batch_response = self.dp.process_batch(convs)
            batch_p_intent = get_intent_distribution(self.agent, batch_user)
            
            self.batches.append({
                'user': batch_user,
                'response': batch_response,
                'p_intent': batch_p_intent
            })
        self.vocab_size = len(TOKENIZER)
        print(f"Data loaded: {total_samples} total samples, {len(self.batches)} batches, tokenizer vocab size: {self.vocab_size}")
        return self.batches
        
    def _init_encoder(self):
        """Initialize emotion contagion encoder."""
        print("\\n=== Initializing Encoder ===")
        enc_cfg = self.unified_config.get('model', {}).get('encoder', {})
        self.encoder = build_emotion_contagion_encoder(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            enc_cfg=enc_cfg
        ).to(self.device)
        
    # 使用预训练 tokenizer 的 vocab_size 已在构建阶段确定，不再动态 build
        
        # Initialize emotion classifier
        self.emotion_cls = EmotionClassifier(
            d_in=self.model_dim, num_emotions=32
        ).to(self.device)
        
        print("Using h_tilde=None (ERA disabled)")
            
    def _init_it_module(self):
        """Initialize Intent Twice module."""
        print("\\n=== Initializing Intent Twice Module ===")
        it_cfg = self.unified_config.get('model', {}).get('intent_twice', {})
        self.it_module = build_intent_twice_modules(
            model_dim=self.model_dim, config_section=it_cfg
        ).to(self.device)
        self.it_module.encoder_module = self.encoder
        
        # Initialize semantic scorer
        self.semantic_scorer = IntentSemanticScorer(d_in=self.model_dim).to(self.device)
        
    def _init_decoder(self):
        """Initialize paper-compliant response decoder."""
        print("\\n=== Initializing Decoder ===")
        dec_cfg = self.unified_config.get('model', {}).get('decoder', {})
        self.decoder_with_loss = create_paper_compliant_decoder(
            vocab_size=self.vocab_size,
            d_model=self.model_dim,
            nhead=dec_cfg.get('nhead', 4),
            dim_feedforward=dec_cfg.get('ff_dim', 256),
            dropout=dec_cfg.get('dropout', 0.1),
            with_loss=True,
            coverage_weight=self.coverage_weight
        ).to(self.device)
        
    def _prepare_batch_data(self, batch_data=None):
        """Prepare batch data for training."""
        if batch_data is None:
            # Use first batch for compatibility
            batch_data = self.batches[0] if self.batches else {'user': [], 'response': [], 'p_intent': []}
        
        user_data = batch_data['user']
        response_data = batch_data['response']
        p_intent_data = batch_data['p_intent']
        # user_data entries already contain: 'input_ids' (list[int]), 'label_ids', 'attention_mask'

        input_ids = torch.tensor([u['input_ids'] for u in user_data], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([u['attention_mask'] for u in user_data], dtype=torch.long, device=self.device)
        label_ids = torch.tensor([u['label_ids'] for u in user_data], dtype=torch.long, device=self.device)
        emotion_id = torch.tensor([EmotionMappings.EMOTIONS.index(u['matched_emotion']) for u in user_data], dtype=torch.long, device=self.device)
        
        # Prepare decoder inputs from response data
        decoder_data = self._prepare_decoder_data(response_data, user_data)
        
        return {
            'tokens': input_ids,  # now a LongTensor [B, L]
            'label_ids': label_ids,
            'attention_mask': attention_mask,
            'h_tilde': None,  # ERA placeholder
            'p_intent': p_intent_data,
            'emotion_id': emotion_id,
            **decoder_data
        }
        
    def _prepare_decoder_data(self, response_data, user_data):
        """Prepare decoder input data with real response tokens."""
        max_resp_len = self.unified_config.get('model', {}).get('decoder', {}).get('max_resp_len', 16)

        # Response items already preprocessed? If processor updated similarly, expect 'input_ids'.
        resp_ids_list = [r['input_ids'] for r in response_data]
        resp_texts = [' '.join(r.get('tokens', [])) for r in response_data]

        # Ensure each response id sequence is length max_resp_len
        trimmed_resp_ids = []
        for ids in resp_ids_list:
            if len(ids) > max_resp_len:
                trimmed_resp_ids.append(ids[:max_resp_len])
            elif len(ids) < max_resp_len:
                pad_id_local = TOKENIZER.pad_token_id if TOKENIZER.pad_token_id is not None else 0
                trimmed_resp_ids.append(ids + [pad_id_local] * (max_resp_len - len(ids)))
            else:
                trimmed_resp_ids.append(ids)
        resp_ids = torch.tensor(trimmed_resp_ids, dtype=torch.long, device=self.device)

        # Special tokens
        pad_id = TOKENIZER.pad_token_id if TOKENIZER.pad_token_id is not None else 0
        bos_id = TOKENIZER.cls_token_id if TOKENIZER.cls_token_id is not None else (TOKENIZER.bos_token_id or pad_id)
        eos_id = TOKENIZER.sep_token_id if TOKENIZER.sep_token_id is not None else (TOKENIZER.eos_token_id or pad_id)

        # Teacher forcing input (prepend BOS, shift right)
        bos_col = torch.full((resp_ids.size(0), 1), bos_id, device=self.device, dtype=torch.long)
        trg_input_ids = torch.cat([bos_col, resp_ids[:, :-1]], dim=1)
        gold_ids = resp_ids
        tgt_key_padding_mask = (trg_input_ids == pad_id)

        # Source ids: reuse already prepared encoder ids from user_data (do NOT re-tokenize)
        src_ids_from_encoder = torch.tensor([u['input_ids'] for u in user_data], dtype=torch.long, device=self.device)

        return {
            'trg_input_ids': trg_input_ids,
            'gold_ids': gold_ids,
            'tgt_key_padding_mask': tgt_key_padding_mask,
            'src_ids_from_encoder': src_ids_from_encoder,
            'responses_tokens': resp_texts
        }
        
    def forward(self, batch_data=None):
        """
        Forward pass through the complete ReflectDiffu pipeline.
        
        Args:
            batch_data: Optional batch data dict with 'user', 'response', 'p_intent' keys.
                       If None, will prepare from internal data.
            
        Returns:
            Dict containing all losses and intermediate outputs
        """
        prepared_data = self._prepare_batch_data(batch_data)
        
        # Extract batch components
        tokens = prepared_data['tokens']
        label_ids = prepared_data['label_ids']
        attention_mask = prepared_data['attention_mask']
        h_tilde = prepared_data['h_tilde']
        p_intent = prepared_data['p_intent']
        emotion_id = prepared_data['emotion_id']
        
        # 1. Emotion Contagion Encoder
        enc_out = self.encoder.forward(
            tokens=tokens,
            label_ids=label_ids,
            attention_mask=attention_mask,
            h_tilde=h_tilde
        )
        H, Q, P = enc_out['H'], enc_out['Q'], enc_out['P']
        
        # Emotion classification and loss
        emotion_labels = emotion_id
        ntx_loss = self.encoder.loss(P, emotion_labels)
        # emo_logits = self.emotion_cls.forward(Q)
        Lce_loss = ce_loss_for_emotion(P, emotion_labels)
        Lem = ntx_loss + Lce_loss
        
        # 2. Intent Twice Integration
        P_semantic, _ = self.semantic_scorer.forward(Q)
        enc_out["P_semantic"] = P_semantic
        
        BATCH_INTEGRATOR.add("input_data", batch_data)
        BATCH_INTEGRATOR.add("emotion_p", P.clone().detach().cpu().tolist())
        BATCH_INTEGRATOR.add("intent_p", P_semantic.clone().detach().cpu().tolist())
        
        it_out = self.it_module.forward(
            encoder_out=enc_out, 
            intent_out={'p_intent': p_intent}
        )
        Emofused = it_out['Emofused']
        Ltwice = it_out['Ltwice']
        
        # 3. Response Decoder
        if Emofused.dim() == 2:
            Emofused = Emofused.unsqueeze(1)
        
        # Prepare src_token_ids for pointer-generator
        src_ids_from_encoder = prepared_data['src_ids_from_encoder']
        L_emof = Emofused.size(1)
        if src_ids_from_encoder.size(1) < L_emof:
            pad_cols = L_emof - src_ids_from_encoder.size(1)
            pad_id = TOKENIZER.pad_token_id if TOKENIZER.pad_token_id is not None else 0
            src_token_ids = F.pad(src_ids_from_encoder, (0, pad_cols), value=pad_id)
        else:
            src_token_ids = src_ids_from_encoder[:, :L_emof]
        
        # Decoder forward
        dec_out = self.decoder_with_loss.forward(
            trg_input_ids=prepared_data['trg_input_ids'],
            emofused=Emofused,
            src_token_ids=src_token_ids,
            gold_ids=prepared_data['gold_ids'],
            tgt_key_padding_mask=prepared_data['tgt_key_padding_mask'],
            emofused_key_padding_mask=None,
            extended_vocab_size=None,
        )
        Lres = dec_out['loss']
        
        return {
            'Lem': Lem,
            'Ltwice': Ltwice, 
            'Lres': Lres,
            'H': H,
            'Q': Q,
            'P': P,
            'Emofused': Emofused,
            'decoder_output': dec_out,
            'batch_data': prepared_data
        }
        
    def compute_joint_loss(self, outputs, delta=1.0, zeta=1.0, eta=1.0):
        """Compute joint loss from forward outputs."""
        return delta * outputs['Lem'] + zeta * outputs['Ltwice'] + eta * outputs['Lres']
    
    def get_batches(self):
        """Get the list of batches for training."""
        return self.batches
        
    def initialize_all(self, data_path: str, batch_size: int = 4, **kwargs):
        """Convenience method to initialize all components."""
        batches = self._init_data_loader(data_path, batch_size, **kwargs)
        self._init_encoder()
        self._init_it_module()
        self._init_decoder()
        print("\\n✅ All components initialized successfully!")
        return batches