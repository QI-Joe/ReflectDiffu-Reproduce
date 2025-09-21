# 🤖 ReflectDiffu-Reproduce

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Reproduction implementation for:** *ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework*

This repository provides a complete implementation for training and evaluating the ReflectDiffu model, which generates empathetic responses by combining emotion-intent contagion with reinforcement learning and diffusion frameworks.

## 🚀 Quick Start

## Final Result Fast Review
You can review current the best result in
```
./output/eval_logs/
```

### Evaluate Pre-trained Model

Run the best pre-trained model with comprehensive evaluation metrics:

```bash
python display.py --model_path output/best_model/best_model.pt
```

**Output:** Evaluation results will be saved in `output/eval_logs/`

**Metrics Evaluated:**
- 📊 **Relevance**: BLEU-1/2/3/4, BARTScore
- 🎯 **Informativeness**: Perplexity (PPL), Distinct-1/2

### Custom Evaluation

```bash
# Evaluate with custom parameters
python display.py --model_path path/to/model.pt --data_path dataset/test.pkl --max_samples 200

# Use specific device
python display.py --model_path output/best_model/best_model.pt --device cuda
```

## 🏗️ Training Setup

### 1. 📁 Data Preparation

Place your data files in the `dataset/` directory:

```
dataset/
├── emotion_labels_user_response.pkl  # Training data
└── emotion_labels_test.pkl           # Test data
```

**Data Format:**
```python
[
    [
        [user_data, response_data]
    ],
    [
        [user_data1, response_data1]
    ],
    # ... more conversation pairs
]
```

**Data Structure:**
- `user_data`: `[(word1, <em>), (word2, <noem>), ...]`
- `response_data`: `[(word1, <em>), (word2, <noem>), ...]`

### 2. 🧠 EmpHi Intent Prediction Setup

Download the required EmpHi models and reflect-Diffu best models from Google Drive and organize as follows:

```
pre-trained/
├── intent_prediction/
│   └── paras.pkl          # Intent prediction parameters
└── model/
    └── model              # Pre-trained EmpHi model

output
└── best_model/
    └── best_model.pt      # trained best model
```

### 3. 🚂 Start Training

```bash
python train.py
```

## 📊 Evaluation Metrics

### Relevance Metrics
- **BLEU-1/2/3/4**: N-gram overlap with reference responses
- **BARTScore**: Semantic similarity using BART model
- **Brevity Penalty**: Length normalization factor

### Informativeness Metrics
- **Perplexity (PPL)**: Language model confidence (lower is better)
- **Distinct-1/2**: Lexical diversity at unigram/bigram level (higher is better)

## 🛠️ Dependencies

Key requirements:
- Python 3.10
- PyTorch 2.0+
- transformers
- sacrebleu
- torcheval (optional, for optimized perplexity computation)

Install dependencies:
```bash
pip install -r requirement
```

## 📝 Usage Examples

### Basic Evaluation
```bash
python display.py --model_path output/best_model/best_model.pt
```

### Training from Scratch
```bash
# Ensure data is in dataset/ and pre-trained models are in pre-trained/
python train.py
```

### Custom Evaluation with Specific Parameters
```bash
python display.py \
    --model_path checkpoints/epoch_10.pt \
    --data_path dataset/emotion_labels_test.pkl \
    --max_samples 500 \
    --device cuda
```

## 📂 Project Structure

```
ReflectDiffu-Reproduce/
├── src/                           # Source code modules
│   ├── emotion_contagion/         # Emotion contagion components
│   ├── intent_twice/              # Intent processing modules
│   └── era/                       # ERA components
├── evaluation/                    # Evaluation scripts
│   ├── relevance_evaluation.py    # BLEU & BARTScore evaluation
│   └── informativeness.py         # Perplexity & Distinct-n evaluation
├── dataset/                       # Training and test data
├── pre-trained/                   # Pre-trained models
├── output/                        # Model outputs and logs
│   ├── best_model/               # Best trained model
│   └── eval_logs/                # Evaluation results
├── display.py                     # Main evaluation script
├── train.py                       # Training script
└── README.md                      # This file
```

