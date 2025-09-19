"""
Minimal one-pass debug script for EmotionContagion + IntentTwice + ResponseDecoder.

Goals:
1. Load a tiny pickle sample (expects a .pkl file or dir path provided via --ec_data). 
2. Run EmotionContagion encoder end-to-end: get H, Q, P and compute a dummy NT-Xent loss.
3. Instantiate simplified EMU + IntentPolicy through IntentTwiceModule and run forward once.
4. Build a minimal ResponseDecoder and run a teacher-forced single step loss.
5. Print shapes and loss values to verify plumbing.

Assumptions / Simplifications:
- Data already tokenized into required format for EmotionContagionDataProcessor (tokens + reason labels).
- Uses only CPU unless --cuda is passed and CUDA is available.
- Creates a tiny vocab on-the-fly from batch tokens.
- Generates synthetic target response ids (random) for decoder test.
- No advanced error handling; fails fast.

Run:
python start_debug_pipeline.py --ec_data path/to/sample.pkl
"""
import argparse
import os
import torch
import torch.nn.functional as F
from torch import nn

# ================= Import project modules =================
from src.emotion_contagion.data_processor import EmotionContagionDataProcessor
from src.emotion_contagion.config import EmotionContagionConfig
from src.emotion_contagion.encoder import EmotionContagionEncoder, EmotionClassifier, ce_loss_for_emotion
from src.intent_twice.intent_twice_integration import IntentTwiceConfig, IntentTwiceModule
from src.intent_twice.EMU import EMUConfig, EMU
from src.intent_twice.IntentPolicy import IntentPolicy
from src.intent_twice.response_decoder import ResponseDecoder, DecoderWithLoss


def build_emotion_contagion_encoder(vocab_size: int = 5000, model_dim: int = 128):
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


def build_intent_twice_modules(model_dim: int = 128, emotion_dim: int = 32):
    it_cfg = IntentTwiceConfig(model_dim=model_dim, emotion_dim=emotion_dim, intent_vocab_size=64, intent_embed_dim=model_dim)
    emu_cfg = EMUConfig(hidden_dim=model_dim, intent_vocab_size=emotion_dim, diffusion_steps=10)
    emu = EMU(emu_cfg)
    # IntentPolicy expects refer_map: use top3 intents among first 9 intents (reuse mapping logic inside IntentTwiceModule)
    # We'll fabricate a simple map: groups 0..4 -> [0,1,2]
    refer_map = {i: [0,1,2] for i in range(5)}
    policy = IntentPolicy(state_dim=model_dim, intent_embed=nn.Embedding(it_cfg.intent_vocab_size, it_cfg.intent_embed_dim), refer_map=refer_map)
    module = IntentTwiceModule(config=it_cfg, encoder_module=None, emu_module=emu, intent_policy_module=policy)
    return module


def build_decoder(vocab_size: int, model_dim: int = 128):
    dec = ResponseDecoder(vocab_size=vocab_size, d_model=model_dim, nhead=4, num_layers=2, dim_feedforward=256)
    dec_with_loss = DecoderWithLoss(decoder=dec, coverage_weight=0.0)
    return dec_with_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ec_data', type=str, default=r"dataset/emotion_labels.pkl", help='Path to emotion contagion pickle file or directory')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--cuda', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # 1. Load EC data
    dp = EmotionContagionDataProcessor(max_length=64)
    raw_data = dp.load_data(args.ec_data)
    # Just take first batch_size samples
    raw_data = raw_data[:args.batch_size]
    processed = dp.process_batch(raw_data)

    # Build tiny vocab from tokens in batch
    vocab_tokens = sorted({tok for sample in processed for tok in sample['tokens'] if tok != '[PAD]'} )
    vocab_size = max(len(vocab_tokens)+10, 100)  # ensure a minimum size

    # 2. Build EC encoder
    encoder = build_emotion_contagion_encoder(vocab_size=vocab_size, model_dim=128).to(device)
    # Build word embedding vocab dynamically
    encoder.word_embedding.build_vocab([s['tokens'] for s in processed])

    # Prepare batch tensors
    tokens = [s['tokens'] for s in processed]
    label_ids = torch.tensor([s['label_ids'] for s in processed], dtype=torch.long, device=device)
    attention_mask = torch.tensor([s['attention_mask'] for s in processed], dtype=torch.long, device=device)

    # 3. Run EC forward
    enc_out = encoder.forward(tokens=tokens, label_ids=label_ids, attention_mask=attention_mask, h_tilde=None)
    H, Q, P = enc_out['H'], enc_out['Q'], enc_out['P']
    print('EmotionContagion: H', H.shape, 'Q', Q.shape)
    emotion_cls = EmotionClassifier(d_in=Q.size(-1), num_emotions=32).to(device)

    # Dummy supervised loss: create random emotion labels (emotion_dim=32) for NT-Xent test
    emotion_labels = P.argmax(dim=-1)
    ntx_loss = encoder.loss(Q, emotion_labels)
    emo_logits = emotion_cls.forward(Q)
    Lce_loss = ce_loss_for_emotion(emo_logits, emotion_labels)
    Lem = ntx_loss+Lce_loss
    print('EC NT-Xent loss:', float(Lem))

    # 4. IntentTwice minimal forward
    it_module = build_intent_twice_modules(model_dim=128, emotion_dim=32).to(device)
    # Attach the encoder we already built
    it_module.encoder_module = encoder

    it_out = it_module(tokens=tokens, label_ids=label_ids, attention_mask=attention_mask, h_tilde=None)
    print('IntentTwice: Q', it_out['Q'].shape, 'Emofused', it_out['Emofused'].shape, 'Ltwice', float(it_out['Ltwice']))

    # 5. Response decoder test
    decoder_with_loss = build_decoder(vocab_size=vocab_size, model_dim=128).to(device)
    # Create synthetic target sequences (shifted input + gold)
    T = 16
    trg_input_ids = torch.randint(0, vocab_size, (Q.size(0), T), device=device)
    gold_ids = torch.randint(0, vocab_size, (Q.size(0), T), device=device)
    src_token_ids = torch.randint(0, vocab_size, (Q.size(0), H.size(1)), device=device)

    memory = H + it_out['Emofused'].unsqueeze(1)  # [B,L,D]

    dec_out = decoder_with_loss(
        trg_input_ids=trg_input_ids,
        memory=memory,
        src_token_ids=src_token_ids,
        gold_ids=gold_ids,
        src_key_padding_mask=(attention_mask==0),
        tgt_key_padding_mask=None,
        extended_vocab_size=None,
        pad_mask=torch.ones_like(trg_input_ids, dtype=torch.long, device=device)
    )

    print('Decoder Pw:', dec_out['Pw'].shape, 'Loss:', float(dec_out['loss']))

    print('All components executed successfully.')

if __name__ == '__main__':
    main()
