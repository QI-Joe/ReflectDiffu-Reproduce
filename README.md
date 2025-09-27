# 🤖 ReflectDiffu-Reproduce

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Reproduction implementation for:** *ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework*

This repository provides a complete implementation for training and evaluating the ReflectDiffu model, which generates empathetic responses by combining emotion-intent contagion with reinforcement learning and diffusion frameworks.

## Train file explination

### 其它模型组件介绍
[Emotion Contagion部分介绍](src/emotion_contagion/README.md) \
[ERA部分介绍](src/era/README.md)\
[Intent Twice部分介绍](src/intent_twice/README.md)

### 模型整体训练与 `model_intergration.py` 中 `ReflectDiffu.forward` 说明

本节详细解释训练主流程中各组件的输入 / 输出张量、形状、以及它们之间的衔接。源码核心调用链：

`train.py -> ReflectDiffu.initialize_all() -> 迭代 batches -> train_step() -> model.forward(batch)`。

> 各子模块内部更细的结构、公式与背景可参考对应子目录内的 README：
> - 情感传染编码（Emotion Contagion）：`src/emotion_contagion/README.md`
> - ERA（情感原因标签/推理，可选）：`src/era/README.md`
> - Intent Twice（RL + Diffusion 意图二次机制）：`src/intent_twice/README.md`

#### 1. 批数据准备 `_prepare_batch_data`
从 `batches`（由 `_init_data_loader` 生成）中取出一个元素，构造成训练所需张量：

| 字段 | 形状 / 类型 | 含义 | 来源 |
|------|-------------|------|------|
| `tokens` | `[B, L]` LongTensor | 用户输入 token id 序列 | 用户侧 `user_data['input_ids']` |
| `label_ids` | `[B, L]` LongTensor | 每个 token 的情感/原因标签 id | `user_data['label_ids']` |
| `attention_mask` | `[B, L]` LongTensor | 1=有效,0=Padding | `user_data['attention_mask']` |
| `h_tilde` | `None` 或 `[B,L,D]` | ERA 编码特征（当前关闭为 None） | 若开启 ERA 模块 |
| `p_intent` | `List[List[float]]` (长度 B，每个内部维度≈意图词表大小) | 外部意图预测分布（EmpHi / 预训练意图模型） | `get_intent_distribution` |
| `emotion_id` | `[B]` LongTensor | 主导情感类别 id（用于分类监督） | 匹配映射 `matched_emotion` |
| `trg_input_ids` | `[B, T_dec]` LongTensor | 解码器 teacher forcing 输入（左移并加 BOS） | 由响应序列处理 |
| `gold_ids` | `[B, T_dec]` LongTensor | 解码目标（含 PAD） | 响应截断/填充 |
| `tgt_key_padding_mask` | `[B, T_dec]` Bool/Long | 解码端 padding mask | 构造时 (token==pad) |
| `src_ids_from_encoder` | `[B, L]` LongTensor | 复制机制（Pointer）源 tokens | 复用 encoder 输入 |
| `responses_tokens` | `List[str]` | 文本形式仅用于日志 | response tokens 拼接 |

> 备注：`B`=batch size，`L`=编码序列最大长度，`T_dec`=解码序列最大长度。

#### 2. 情感传染编码器阶段
调用：
```python
enc_out = self.encoder.forward(tokens, label_ids, attention_mask, h_tilde)
```
主要输出：
| 名称 | 形状 | 说明 |
|------|------|------|
| `H` | `[B, L, D]` | 编码后上下文序列隐藏表示 |
| `Q` | `[B, D]` | 对序列池化/融合后的全局情感语义向量（后续意图、对话情感交互核心）|
| `P` | `[B, N_emotion]` | 情感类别概率分布 |

损失：
| 名称 | 形状 | 说明 |
|------|------|------|
| `ntx_loss` | 标量 | 对比学习/温度化分布损失（来自 encoder 内部 `loss` 方法）|
| `Lce_loss` | 标量 | 情感多类交叉熵（`ce_loss_for_emotion`）|
| `Lem` | 标量 | 上述二者求和：`Lem = ntx_loss + Lce_loss` |

#### 3. 语义意图分布 + Intent Twice 集成
语义打分：
```python
P_semantic, _ = self.semantic_scorer(Q)   # [B, N_intent]
enc_out['P_semantic'] = P_semantic
```

随后调用意图二次模块：
```python
it_out = self.it_module.forward(encoder_out=enc_out, intent_out={'p_intent': p_intent})
```
意图二次模块（内部含 EMU + Policy + 融合）核心产物：
| 名称 | 形状 | 含义 |
|------|------|------|
| `Emofused` | `[B, L_fuse, D]` 或 `[B, D]` | 正负情感意图方向融合后的情感-意图表示（若为 `[B,D]` 会在后面升维）|
| `Ltwice` | 标量 | Intent Twice 子模块总损失（KL + 意图校正 CE + 可选策略项）|
| （可能还包含）`Emopos/Emoneg` | `[B, D]` | 正/负分支情感方向（若在内部字典暴露）|

#### 4. 响应解码器阶段（Pointer-Generator）
准备：如果 `Emofused` 是 `[B,D]` 则 `unsqueeze(1)` → `[B,1,D]`。

源码调用：
```python
dec_out = self.decoder_with_loss(
    trg_input_ids, emofused=Emofused, src_token_ids=src_ids_from_encoder,
    gold_ids=gold_ids, tgt_key_padding_mask=tgt_key_padding_mask)
```

输出（`PaperCompliantDecoderWithLoss`）：
| 名称 | 形状 | 说明 |
|------|------|------|
| `Pw` | `[B, T_dec, V_ext]` | 最终词分布（混合生成+复制）|
| `Pgen` | `[B, T_dec, V_ext]` | 纯生成 softmax 分布 |
| `Pcopy` | `[B, T_dec, V_ext]` | 根据注意力散射后的复制分布 |
| `p_mix` | `[B, T_dec, 1]` | Pointer-Generator 门：生成 vs 复制 权重 |
| `hidden` | `[B, T_dec, D]` | 解码隐藏状态 |
| `attn_weights` | `[B, T_dec, L_emof]` | 对 `Emofused`（作为唯一 KV） 的注意力权重 |
| `context_vector` | `[B, T_dec, D]` | 注意力加权上下文向量 |
| `loss` | 标量 | 解码总损失（CE + coverage*系数）|
| `ce_loss` | 标量 | 交叉熵部分 |
| `coverage_loss` | 标量 | 覆盖损失（若启用）|

定义响应损失：`Lres = dec_out['loss']`。

#### 5. Forward 最终返回字典
`ReflectDiffu.forward` 返回：
| Key | 形状 | 语义 |
|-----|------|------|
| `Lem` | 标量 | 情感编码相关损失 (对比 + CE) |
| `Ltwice` | 标量 | 意图二次模块损失 |
| `Lres` | 标量 | 解码器损失 |
| `H` | `[B, L, D]` | 编码的上下文序列特征 |
| `Q` | `[B, D]` | 全局情感语义向量（下游语义/意图交互核心）|
| `P` | `[B, N_emotion]` | 情感类别分布 |
| `Emofused` | `[B, L_fuse, D]` | 情感+意图融合表示（供解码 Cross-Attn）|
| `decoder_output` | dict | 上文列出的所有解码阶段中间产物 |
| `batch_data` | dict | 本次 forward 输入的原始/衍生批数据（便于调试或日志）|

> 训练中联合损失（`train_step` 内）：`joint_loss = δ·Lem + ζ·Ltwice + η·Lres`，对应命令行或配置中的 `--delta / --zeta / --eta` 权重。

#### 6. 组件协同总结
1. Data Processor：清洗并对齐用户/回应，产生 token ids 与标签。 
2. Emotion Contagion Encoder：建模上下文情感动态 → 输出序列表示 `H` 与全局表示 `Q` 及情感分布 `P`。 
3. Semantic Scorer：基于 `Q` 额外生成语义意图分布 `P_semantic` 与外部意图分布结合。 
4. Intent Twice (EMU + Policy)：利用正/负分支扩散 + 意图候选策略，校正并融合为 `Emofused`，提供情感-意图一致性。 
5. Response Decoder：仅以 `Emofused` 为跨注意键值进行受控生成；Pointer-Generator 结合复制以提升可复制实体/情感词。 
6. Loss Aggregation：三块损失加权合成，支持灵活调参；可扩展加入 ERA 或策略损失权重。 

#### 7. 快速定位排查建议
| 现象 | 首先查看 |
|------|----------|
| 解码重复 | `coverage_loss` 是否启用；`p_mix` 是否长时间接近 0 or 1 |
| 情感不稳定 | `P` 分布是否塌陷；`Lem` 是否下降 |
| 意图未体现 | `Emofused` 与 `Emopos/Emoneg` 是否几乎无差异；`Ltwice` 是否长期为 0 |
| 输出风格单一 | 检查外部 `p_intent` 是否恒定；策略采样温度/扩散步数 |

---
通过上述分层结构与张量流向，可以快速理解从原始对话输入到最终 empathetic response 生成的全链路：情感与意图在编码阶段耦合，再经 RL-Diffusion 双分支强化后，在解码阶段被集中用于生成控制。若需要深入内部公式或细节，请跳转各自子目录 README。


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

Download the required EmpHi models and reflect-Diffu best models from [Google Drive](https://drive.google.com/drive/folders/148Ftrh_mH8y7_yOap2h9CmMXxYn4cgc0?usp=drive_link) and organize as follows:

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

