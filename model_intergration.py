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
from src.intent_twice.intent_twice_integration import IntentTwiceConfig, IntentTwiceModule
from src.intent_twice.EMU import EMUConfig, EMU
from src.intent_twice.IntentPolicy import IntentPolicy

from src.intent_twice.response_decoder import create_paper_compliant_decoder
from portable_inference import temp_load_intent, build_agent, get_intent_distribution

def build_emotion_contagion_encoder(vocab_size: int = 5000, model_dim: int = 128):
    """构建情感传染编码器"""
    cfg = EmotionContagionConfig(
        vocab_size=vocab_size,
        word_embedding_dim=model_dim,
        model_dim=model_dim,
        max_position_embeddings=256,
        num_reason_labels=2,
        num_encoder_layers=2,
        num_attention_heads=4,
        feedforward_dim=256,
        attention_type="cross",  # or "gate"
        attention_dropout=0.1,
        dropout_rate=0.1,
        era_hidden_dim=model_dim,
        era_projection_dim=model_dim,
    )
    encoder = EmotionContagionEncoder(cfg)
    return encoder


def build_intent_twice_modules(model_dim: int = 128, emotion_dim: int = 32, intent_vocab_size: int = 9):
    """构建Intent Twice模块"""
    it_cfg = IntentTwiceConfig(
        model_dim=model_dim, 
        emotion_dim=emotion_dim, 
        intent_vocab_size=intent_vocab_size, 
        intent_embed_dim=model_dim
    )
    
    # EMU expects emotion_dim (32) for the CVAE, not intent_vocab_size (9)
    emu_cfg = EMUConfig(hidden_dim=model_dim, intent_vocab_size=intent_vocab_size, diffusion_steps=10)
    emu = EMU(emu_cfg)
    
    # IntentPolicy expects refer_map: use top3 intents among first 9 intents
    refer_map = {i: [0,1,2] for i in range(5)}
    policy = IntentPolicy(
        state_dim=model_dim, 
        intent_embed=nn.Embedding(it_cfg.intent_vocab_size, it_cfg.intent_embed_dim), 
        refer_map=refer_map
    )
    
    module = IntentTwiceModule(
        config=it_cfg, 
        encoder_module=None, 
        emu_module=emu, 
        intent_policy_module=policy
    )
    return module


def try_load_era_model(model_dim: int = 128):
    try:
        # 这里需要根据你的ERA实现来加载
        # from src.era.era_model import ERA_BERT_CRF, ERAConfig
        # era_cfg = ERAConfig(hidden_size=model_dim)
        # era_model = ERA_BERT_CRF(era_cfg)
        # return era_model
        print("ERA model loading not implemented, using h_tilde=None")
        return None
    except Exception as e:
        print(f"Failed to load ERA model: {e}")
        return None


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
                 use_era: bool = False, coverage_weight: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_era = use_era
        self.coverage_weight = coverage_weight
        
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
        
        # Process all data first to get vocab size
        model_path = os.path.join(os.path.dirname(__file__), 'pre-trained', 'model', 'model')
        self.agent = build_agent(model_path)
        
        # Update vocab size based on all data
        self.vocab_tokens = set()
        def update_vocab(samples):
            for sample in samples:
                self.vocab_tokens.update(tok for tok in sample['tokens'] if tok != '[PAD]')
        
        # Create batches
        self.batches = []
        total_samples = len(self.raw_data)
        for i in range(0, total_samples):
            single_conv = self.raw_data[i]
            batch_user, batch_response = self.dp.process_batch(single_conv)
            batch_p_intent = get_intent_distribution(self.agent, batch_user)
            
            update_vocab(batch_user)
            update_vocab(batch_response)
            
            self.batches.append({
                'user': batch_user,
                'response': batch_response,
                'p_intent': batch_p_intent
            })
        self.vocab_size = max(len(self.vocab_tokens) + 20, 100)  
        print(f"Data loaded: {total_samples} total samples, {len(self.batches)} batches, vocab size: {self.vocab_size}")
        return self.batches
        
    def _init_encoder(self):
        """Initialize emotion contagion encoder."""
        print("\\n=== Initializing Encoder ===")
        self.encoder = build_emotion_contagion_encoder(
            vocab_size=self.vocab_size, model_dim=self.model_dim
        ).to(self.device)
        
        self.encoder.word_embedding.build_vocab([self.vocab_tokens])
        
        # Initialize emotion classifier
        self.emotion_cls = EmotionClassifier(
            d_in=self.model_dim, num_emotions=32
        ).to(self.device)
        
        # Try to load ERA model
        if self.use_era:
            self.era_model = try_load_era_model(self.model_dim)
            if self.era_model:
                self.era_model = self.era_model.to(self.device)
                print("ERA model loaded successfully")
            else:
                print("ERA model loading failed, using h_tilde=None")
        else:
            print("Using h_tilde=None (ERA disabled)")
            
    def _init_it_module(self):
        """Initialize Intent Twice module."""
        print("\\n=== Initializing Intent Twice Module ===")
        self.it_module = build_intent_twice_modules(
            model_dim=self.model_dim, emotion_dim=32
        ).to(self.device)
        self.it_module.encoder_module = self.encoder
        
        # Initialize semantic scorer
        self.semantic_scorer = IntentSemanticScorer(d_in=self.model_dim).to(self.device)
        
    def _init_decoder(self):
        """Initialize paper-compliant response decoder."""
        print("\\n=== Initializing Decoder ===")
        self.decoder_with_loss = create_paper_compliant_decoder(
            vocab_size=self.vocab_size,
            d_model=self.model_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
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
        
        # Prepare encoder inputs from user data
        tokens = [s['tokens'] for s in user_data]
        label_ids = torch.tensor([s['label_ids'] for s in user_data], 
                                dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([s['attention_mask'] for s in user_data], 
                                     dtype=torch.long, device=self.device)
        
        # Prepare decoder inputs from response data
        decoder_data = self._prepare_decoder_data(response_data, user_data)
        
        return {
            'tokens': tokens,
            'label_ids': label_ids,
            'attention_mask': attention_mask,
            'h_tilde': None,  # ERA placeholder
            'p_intent': p_intent_data,
            **decoder_data
        }
        
    def _prepare_decoder_data(self, response_data, user_data):
        """Prepare decoder input data with real response tokens."""
        # Add special tokens to vocabulary
        special_tokens = ['<bos>', '<eos>', '[PAD]']
        current_vocab = set(self.encoder.word_embedding.word_to_idx.keys())
        new_tokens = [tok for tok in special_tokens if tok not in current_vocab]
        
        if new_tokens:
            self._extend_vocabulary(new_tokens)
        
        bos_id = self.encoder.word_embedding.word_to_idx['<bos>']
        eos_id = self.encoder.word_embedding.word_to_idx['<eos>']
        pad_id = self.encoder.word_embedding.word_to_idx['[PAD]']
        
        # Extract response tokens from provided response data
        responses_tokens = [sample['tokens'] for sample in response_data]
        
        # Convert to IDs and create teacher-forcing inputs
        T = 16  # target sequence length
        resp_ids_batch = []
        
        def tokens_to_ids(tokens):
            return [self.encoder.word_embedding.word_to_idx.get(t, 
                   self.encoder.word_embedding.word_to_idx.get('[UNK]', 0)) for t in tokens]
        
        for toks in responses_tokens:
            ids = tokens_to_ids(toks)
            ids = ids[:T-1] + [eos_id] if len(ids) >= T else ids + [eos_id]
            if len(ids) < T:
                ids = ids + [pad_id] * (T - len(ids))
            else:
                ids = ids[:T]
            resp_ids_batch.append(ids)
        
        resp_ids = torch.tensor(resp_ids_batch, device=self.device, dtype=torch.long)
        
        # Teacher-forcing setup
        bos_col = torch.full((resp_ids.size(0), 1), bos_id, device=self.device, dtype=torch.long)
        trg_input_ids = torch.cat([bos_col, resp_ids[:, :-1]], dim=1)
        gold_ids = resp_ids
        tgt_key_padding_mask = (trg_input_ids == pad_id)
        
        # Source token IDs for pointer-generator from user data
        src_ids_from_encoder = torch.tensor(
            [[self.encoder.word_embedding.word_to_idx.get(tok, 0) for tok in s['tokens']] 
             for s in user_data],
            device=self.device, dtype=torch.long
        )
        
        return {
            'trg_input_ids': trg_input_ids,
            'gold_ids': gold_ids,
            'tgt_key_padding_mask': tgt_key_padding_mask,
            'src_ids_from_encoder': src_ids_from_encoder,
            'responses_tokens': responses_tokens
        }
        
    def _extend_vocabulary(self, new_tokens):
        """Extend vocabulary with new tokens."""
        current_vocab = set(self.encoder.word_embedding.word_to_idx.keys())
        all_tokens = list(current_vocab) + new_tokens
        
        # Rebuild mappings
        self.encoder.word_embedding.word_to_idx = {word: idx for idx, word in enumerate(sorted(all_tokens))}
        self.encoder.word_embedding.idx_to_word = {idx: word for word, idx in self.encoder.word_embedding.word_to_idx.items()}
        
        # Extend embedding layer
        old_vocab_size = self.encoder.word_embedding.vocab_size
        new_vocab_size = len(self.encoder.word_embedding.word_to_idx)
        if new_vocab_size > old_vocab_size:
            old_weight = self.encoder.word_embedding.embedding.weight.data
            self.encoder.word_embedding.embedding = nn.Embedding(
                new_vocab_size, self.encoder.word_embedding.embedding_dim
            ).to(self.device)
            self.encoder.word_embedding.embedding.weight.data[:old_vocab_size] = old_weight
            self.encoder.word_embedding.vocab_size = new_vocab_size
        
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
        
        # 1. Emotion Contagion Encoder
        enc_out = self.encoder.forward(
            tokens=tokens,
            label_ids=label_ids,
            attention_mask=attention_mask,
            h_tilde=h_tilde
        )
        H, Q, P = enc_out['H'], enc_out['Q'], enc_out['P']
        
        # Emotion classification and loss
        emotion_labels = P.argmax(dim=-1)
        ntx_loss = self.encoder.loss(Q, emotion_labels)
        emo_logits = self.emotion_cls.forward(Q)
        Lce_loss = ce_loss_for_emotion(emo_logits, emotion_labels)
        Lem = ntx_loss + Lce_loss
        
        # 2. Intent Twice Integration
        P_semantic, _ = self.semantic_scorer(Q)
        enc_out["P_semantic"] = P_semantic
        
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
            pad_id = self.encoder.word_embedding.word_to_idx['[PAD]']
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