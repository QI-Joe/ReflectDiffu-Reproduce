"""
Example usage of Emotion-Contagion Encoder

Demonstrates how to use the implemented emotion-contagion encoder
with the specified data format.
"""

import torch
import json
from src.emotion_contagion import (
    EmotionContagionConfig,
    EmotionContagionEncoder,
    EmotionContagionDataProcessor
)


def create_sample_data():
    """Create sample data in the specified format for testing."""
    sample_data = [
        {
            "user": True,
            "origin_prompt": "I feel really sad about what happened",
            "sentence1": [
                ["I", "feel", "really", "sad", "about", "what", "happened"],
                ["noem", "em", "em", "em", "noem", "noem", "noem"]
            ]
        },
        {
            "user": False,
            "origin_prompt": "That sounds difficult to deal with",
            "sentence1": [
                ["That", "sounds", "difficult", "to", "deal", "with"],
                ["noem", "noem", "em", "noem", "noem", "noem"]
            ]
        },
        {
            "user": True,
            "origin_prompt": "Yes, I'm feeling overwhelmed and anxious",
            "sentence1": [
                ["Yes", ",", "I'm", "feeling", "overwhelmed", "and", "anxious"],
                ["noem", "noem", "noem", "em", "em", "noem", "em"]
            ]
        }
    ]
    return sample_data


def main():
    """Main example function."""
    print("=== Emotion-Contagion Encoder Example ===\n")
    
    # 1. Create configuration
    config = EmotionContagionConfig(
        word_embedding_dim=300,
        model_dim=300,
        max_position_embeddings=512,
        num_encoder_layers=4,
        num_attention_heads=8,
        attention_type="cross"  # Use cross-attention method
    )
    
    print(f"Configuration:")
    print(f"- Model dimension: {config.model_dim}")
    print(f"- Encoder layers: {config.num_encoder_layers}")
    print(f"- Attention heads: {config.num_attention_heads}")
    print(f"- Attention type: {config.attention_type}\n")
    
    # 2. Create sample data
    sample_data = create_sample_data()
    print(f"Created {len(sample_data)} sample dialogues\n")
    
    # 3. Initialize data processor
    processor = EmotionContagionDataProcessor(max_length=128)
    
    # 4. Process data
    dataset = processor.create_dataset(sample_data)
    dataloader = processor.create_dataloader(dataset, batch_size=2, shuffle=False)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: 2\n")
    
    # 5. Initialize encoder
    encoder = EmotionContagionEncoder(config)
    
    # 6. Build vocabulary from data
    encoder.build_vocab_from_data(dataloader)
    vocab_size = encoder.word_embedding.vocab_size
    print(f"Built vocabulary with {vocab_size} tokens\n")
    
    # 7. Process a batch
    batch = next(iter(dataloader))
    
    print("Batch contents:")
    print(f"- Tokens shape: {len(batch['tokens'])} x {len(batch['tokens'][0])}")
    print(f"- Label IDs shape: {batch['label_ids'].shape}")
    print(f"- Attention mask shape: {batch['attention_mask'].shape}")
    print(f"- Sequence lengths: {batch['seq_len'].tolist()}\n")
    
    # 8. Forward pass without ERA (h̃)
    print("=== Forward Pass (without ERA) ===")
    with torch.no_grad():
        results = encoder(
            tokens=batch["tokens"],
            label_ids=batch["label_ids"],
            attention_mask=batch["attention_mask"]
        )
    
    print(f"Encoder output H shape: {results['H'].shape}")
    print(f"Global context Q shape: {results['Q'].shape}")
    print(f"Q sample values: {results['Q'][0][:5].tolist()}\n")
    
    # 9. Forward pass with simulated ERA (h̃)
    print("=== Forward Pass (with simulated ERA) ===")
    # Simulate ERA output h̃ (normally this would come from trained ERA model)
    batch_size, seq_len = batch["label_ids"].shape
    h_tilde = torch.randn(batch_size, seq_len, 768)  # Simulated ERA hidden states
    
    with torch.no_grad():
        results_with_era = encoder(
            tokens=batch["tokens"],
            label_ids=batch["label_ids"],
            attention_mask=batch["attention_mask"],
            h_tilde=h_tilde
        )
    
    print(f"Encoder output H shape: {results_with_era['H'].shape}")
    print(f"Attention output Z shape: {results_with_era['Z'].shape}")
    print(f"Global context Q shape: {results_with_era['Q'].shape}")
    print(f"Q sample values: {results_with_era['Q'][0][:5].tolist()}\n")
    
    # 10. Compare outputs
    print("=== Comparison ===")
    q_diff = torch.norm(results['Q'] - results_with_era['Q'])
    print(f"Q difference (with vs without ERA): {q_diff.item():.4f}")
    print("(Non-zero difference indicates ERA attention is working)\n")
    
    print("Example completed successfully!")


if __name__ == "__main__":
    main()