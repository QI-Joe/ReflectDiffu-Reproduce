import os
import sys
import torch
import torch.nn.functional as F
import warnings
import pickle

# Add pre-trained directory to Python path for importing agents
pre_trained_path = os.path.join(os.path.dirname(__file__), 'pre-trained')
sys.path.insert(0, pre_trained_path)
# Also add the pre-trained directory to ensure intent_prediction can be found
sys.path.insert(0, os.path.join(pre_trained_path))

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def build_agent(model_path):
    """Build and load the EmpHi agent with portable paths"""
    parser = ParlaiParser(add_parlai_args=True, add_model_args=True)
    parser.set_defaults(
        task='empathetic_dialogues',
        model='pre-trained.agents.emphi:EmpHi',  # Use relative import
        model_file=model_path,
        dict_lower=True,
        num_layers=2,
        embedding_size=300,
        alpha=0.5,
        gamma=0.5,
        tau=1,
        dropout=0,
        implicit=True,
        implicit_dynamic=True,
        explicit=True,
    )
    opt = parser.parse_args([])
    agent = create_agent(opt)
    agent.model.eval()
    return agent

@torch.no_grad()
def get_intent_distribution(agent, text_list):
    """
    Get intent probability distribution for a batch of texts
    
    Args:
        agent: Loaded EmpHi agent
        text_list: List of strings (input sentences)
        
    Returns:
        torch.Tensor: [B, 9] intent probability distribution
    """
    # Convert texts to token IDs using agent's dictionary
    batch_token_ids = []
    batch_lengths = []
    
    for text in text_list:
        tokens = agent.dict.txt2vec(text["origin_prompt"])
        batch_token_ids.append(torch.LongTensor(tokens))
        batch_lengths.append(len(tokens))
    
    # Pad sequences to same length
    max_len = max(batch_lengths)
    padded_tokens = torch.zeros(len(text_list), max_len, dtype=torch.long)
    
    for i, tokens in enumerate(batch_token_ids):
        padded_tokens[i, :len(tokens)] = tokens
    
    lengths = torch.LongTensor(batch_lengths)
    
    # Move to device
    device = next(agent.model.parameters()).device
    padded_tokens = padded_tokens.to(device)
    # lengths = lengths.to(device)
    
    # Forward through encoder
    encoder_states = agent.model.encoder(padded_tokens, lengths)
    context, state, mask = encoder_states
    
    # Get intent logits from last layer's hidden state
    intent_logits = agent.model.intent_prediction(state[-1])  # [B, 9]
    
    # Convert to probability distribution
    intent_probs = F.softmax(intent_logits, dim=-1)
    
    return intent_probs

def demo_intent_inference():
    """Demo function showing how to use the intent inference"""
    # Portable model path relative to this script
    model_path = os.path.join(os.path.dirname(__file__), 'pre-trained', 'model', 'model')
    
    # Build agent
    print("Loading model...")
    agent = build_agent(model_path)
    print("Model loaded successfully!")
    
    # Intent class names (CRITICAL: order must match training data labels 0-8)
    intent_names = [
        'agreeing',       # 0
        'acknowledging',  # 1
        'encouraging',    # 2
        'consoling',      # 3
        'sympathizing',   # 4
        'suggesting',     # 5
        'questioning',    # 6
        'wishing',        # 7
        'neutral'         # 8
    ]
    
    # Example texts
    test_texts = [
        "I understand how you feel about that.",
        "That sounds really exciting!",
        "I'm sorry to hear you're going through this.",
        "Maybe you could try a different approach?",
        "How are you doing today?"
    ]
    
    # Get intent distributions
    print("\nGetting intent predictions...")
    intent_probs = get_intent_distribution(agent, test_texts)
    
    # Display results
    print("\nIntent Predictions:")
    print("="*60)
    
    for i, text in enumerate(test_texts):
        print(f"\nText: '{text}'")
        probs = intent_probs[i].cpu().numpy()
        
        # Show top 3 intents
        top_indices = probs.argsort()[-3:][::-1]
        for idx in top_indices:
            print(f"  {intent_names[idx]:12} : {probs[idx]:.3f}")
    
    print(f"\nIntent distribution shape: {intent_probs.shape}")
    return intent_probs

def temp_load_intent():
    with open(r"pre-trained/p_intent_1060.pkl", "rb") as f:
        p_intent = pickle.load(f)
    return p_intent

if __name__ == '__main__':
    pintent = demo_intent_inference()