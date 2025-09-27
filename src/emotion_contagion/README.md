# Emotion Contagion 模块说明

本目录实现 ReflectDiffu 论文（正文 Section 3.2）中的 **Emotion-Contagion Encoder（情感传染编码器）**：
通过将语义词向量、位置向量与情感原因标签 (em/noem) 融合，结合 ERA (Emotion Reason Annotator) 输出的推理表示，对上下文对话进行编码，得到全局情感—语境表示 `Q`，为后续意图反射 (Intent Twice) 与响应生成提供可控、高可分的表示。

核心公式（与论文对应）：
1) 组合嵌入:    E^C = E^W + E^P + E^R      (公式 (1))
2) Transformer 编码:  H = TRS_Enc(E^C)      (公式 (2))
3) 与 ERA 推理表示交互:  Z = Attention(H, h~) 或门控重加权
4) 全局聚合:    Q = MeanPooling(Z)          (公式 (3))
5. Contrastive Experts：基于批次极性动态选择正/负专家得到分布 `P`

---
## 目录结构
| 文件 | 主要类 / 函数 | 作用概述 |
|------|---------------|----------|
| `config.py` | `EmotionContagionConfig` | 编码器结构与超参数配置 |
| `foundation_emb.py` | `WordEmbedding`, `SinusoidalPositionalEncoding`, `ReasonEmbedding`, `IntentSemanticScorer` | 基础嵌入、位置、原因标签嵌入、意图语义在线打分 |
| `data_processor.py` | `EmotionContagionDataProcessor`, `EmotionContagionDataset` | em/noem 数据对齐、Padding、Mask 生成 |
| `contrastive_expert.py` | `decide_batch_polarity_via_vader`, `CONExpert` | VADER 评估批量极性；正/负专家融合 |
| `encoder.py` | `EmotionContagionEncoder`, `TransformerEncoderLayer`, `CrossAttention`, `GateReweight`, `EmotionClassifier`, `ce_loss_for_emotion` | 主编码、跨注意力/门控、聚合、对比损失、情感分类 |
| `__init__.py` | 导出符号 | 模块封装入口 |

---
## 数据流与样本结构
上游（含 ERA 标注）提供 DialogueSample：
- `user_tokens`, `user_labels` (em/noem)
- `response_tokens`, `response_labels`
- `emotion`：整体情感类别（监督 / 对比用）

Processor 生成（单侧）字段：
| 字段 | 形状 / 类型 | 含义 |
|------|-------------|------|
| `tokens` | List[str] (≤ L) | 截断后的原始词序列（未 Padding） |
| `input_ids` | `[L_max]` | 分词器 ID（填充到 `max_length`） |
| `label_ids` | `[L_max]` | `<noem>`→0, `<em>`→1，Pad 补 0 |
| `attention_mask` | `[L_max]` | 1=有效, 0=Pad |
| `seq_len` | scalar | 实际有效长度 |
| `origin_prompt` | str | 原始拼接文本 |
| `matched_emotion` | str | 对齐的情感标签 |

Batch 后：`input_ids / label_ids / attention_mask` 维度均为 `[B, L]`。

> 编码器 `forward` 建议直接使用 LongTensor `[B,L]` 的 token IDs（与外部 tokenizer 对齐）。

---
## 编码主流程 (EmotionContagionEncoder.forward)
输入：
- `tokens`: `[B, L]`
- `label_ids`: `[B, L]` (em/noem)
- `attention_mask`: `[B, L]`
- `h_tilde`: `[B, L, D_era]`（可选，ERA 推理向量）

步骤：
1. 嵌入组合：
   - `EW = WordEmbedding(tokens)` → `[B,L,D_emb]`
   - 维度对齐（若需要）：`word_projection(EW)` → `[B,L,D]`
   - `EP`（Sinusoidal 或 Learned）→ `[B,L,D]`
   - `ER = ReasonEmbedding(label_ids)` → `[B,L,D]`
   - `EC = EW + EP + ER` → `[B,L,D]`
2. Transformer 编码：多层 `TransformerEncoderLayer` (Pre-LN) → `H [B,L,D]`
3. 与 ERA 交互：
   - 若提供 `h_tilde`：投影 `h_tilde_proj` → `[B,L,D]`
     - `attention_type == cross`：`Z = CrossAttention(H, h_tilde_proj)`
     - `attention_type == gate`：`Z = GateReweight(H, h_tilde_proj, attention_mask)`
   - 否则 `Z = H`
4. Masked Mean Pooling：`Q = mean(Z * mask) / valid_len` → `[B,D]`
5. Contrastive Expert：`P = CONExpert(Q, tokens)` → `[B, emo_dim]`

输出：`{"H": [B,L,D], "Q": [B,D], "P": [B,emo_dim]}`

---
## 关键子模块
### TransformerEncoderLayer
- 结构：Pre-LN 自注意力 + FFN
- 输入：`[B,L,D]`；输出：同维

### CrossAttention / GateReweight
| 模式 | 输入 | 输出 | 核心思想 |
|------|------|------|----------|
| CrossAttention | H (Query), h̃ (Key/Value) | Z `[B,L,D]` | 显式跨注意力对齐情感推理标注 |
| GateReweight | H, h̃ | Z `[B,L,D]` | 由 h̃ 生成逐 Token gate，重加权 H |

### ReasonEmbedding
- 将 em/noem 标签映射为情感显著性偏置向量。

### CONExpert
- 用 VADER 统计批次极性：得到 `v ∈ {nneg, nneu, npos}`
- `v = nneg` → 选负专家；`v = npos` → 选正专家；中性 → 混合：`α Wpos + (1-α) Wneg`
- 输出 `P` 提供额外的情感子空间表征。

### 对比学习 (`EmotionContagionEncoder.loss`)
- 输入：`Q [B,D]`，`labels [B]`
- 计算余弦相似度矩阵 + 温度缩放 → NT-Xent：
  - 同标签为正对，不同标签为负对
  - 提升同类情感聚类紧致度，区分度更强

### 情感分类 (可选)
- `EmotionClassifier(Q)` → `logits [B, num_emotions]`
- `ce_loss_for_emotion` 支持 label smoothing
- 可与对比损失联合训练

### IntentSemanticScorer（可选挂载）
- 维护意图原型 `intent_prototypes [N_intent, D]`
- `psemantic = softmax(cos(Q, prototypes))` → `[B, N_intent]`
- 在 Intent Twice 中与外部分类器分布融合重排：`Intent_first = psemantic + α p_intent`

---
## 形状速览
| 名称 | 形状 | 说明 |
|------|------|------|
| `input_ids` | `[B,L]` | 词 ID 序列 |
| `label_ids` | `[B,L]` | em/noem 标签 0/1 |
| `attention_mask` | `[B,L]` | 1 有效 / 0 填充 |
| `h_tilde` | `[B,L,D_era]` | ERA 推理表示 |
| `EC, H, Z` | `[B,L,D]` | 中间表示 |
| `Q` | `[B,D]` | 全局情感—语境向量 |
| `P` | `[B,emo_dim]` | 对比专家子空间分布 |
| `logits` | `[B,num_emotions]` | 情感分类输出 |
| `psemantic` | `[B,N_intent]` | 在线意图语义分布 |

---
## 配置参数 (EmotionContagionConfig) 速览
| 参数 | 作用 | 默认 |
|------|------|------|
| `word_embedding_dim` | 词向量维度 | 300 |
| `model_dim` | 主隐藏维 | 300 |
| `num_encoder_layers` | Transformer 层数 | 4 |
| `num_attention_heads` | Multi-Head 数 | 8 |
| `feedforward_dim` | FFN 中间层 | 1024 |
| `position_embedding_type` | 位置编码类型 | sinusoidal |
| `num_reason_labels` | em/noem 类别数 | 2 |
| `attention_type` | cross 或 gate | cross |
| `gate_activation` | gate 激活 | sigmoid |
| `era_hidden_dim` | ERA 输入维 | 768 |
| `era_projection_dim` | ERA 投影维 | = model_dim |
| `dropout_rate` | 编码层 dropout | 0.1 |
| `attention_dropout` | 注意力 dropout | 0.1 |

---
## 设计优势
1. 嵌入简单相加，低耦合易复现；
2. 支持两种情感对齐策略（CrossAttention / Gate）；
3. 利用 em/noem 显式引导情感显著性；
4. Contrastive + CE 双目标稳固类别边界；
5. 极性专家缓解批次情感不均衡；
6. 便于挂载 Intent Twice 的在线意图原型检索。

---
## 典型使用示例
```python
from emotion_contagion import EmotionContagionConfig, EmotionContagionEncoder
import torch

config = EmotionContagionConfig(vocab_size=50000)
encoder = EmotionContagionEncoder(config)

B, L = 8, 64
input_ids = torch.randint(0, config.vocab_size, (B, L))
label_ids = torch.randint(0, 2, (B, L))
attention_mask = torch.ones(B, L, dtype=torch.long)
h_tilde = torch.randn(B, L, config.era_hidden_dim)

out = encoder(tokens=input_ids,
              label_ids=label_ids,
              attention_mask=attention_mask,
              h_tilde=h_tilde)
Q = out["Q"]  # [B,D]
P = out["P"]  # [B,emo_dim]
```

---
## 常见问题与排查
| 症状 | 排查建议 |
|------|----------|
| Q 出现 NaN | 检查是否所有行 `attention_mask` 为 0（除法 0），或梯度爆炸（调低学习率） |
| 情感分类不上升 | 确认标签顺序一致；联合训练时权重是否失衡 |
| GPU OOM | 减少 `num_encoder_layers` / 降低 `L_max` / 减少 batch size |
| 极性专家输出接近均匀 | 检查输入文本是否为空、VADER 阈值设置是否过宽 |
| 对比损失为 0 | 批次内同类样本不足（增大 batch 或分布采样） |

---
## 扩展建议
| 方向 | 思路 |
|------|------|
| 更细粒度标签 | 扩展 `num_reason_labels` 并新增嵌入 |
| 高级聚合 | 使用自注意力池化或 CLS Token |
| 硬负例对比 | 按相似度选择难负样本再计算 NT-Xent |
| 多任务联合 | 与意图分类、响应生成共享 `Q` |
| 可视化 | 输出注意力权重（需在 CrossAttention 中返回） |

---
## 快速摘要
- 输入：含 em/noem 标注与可选 ERA 推理表示的对话序列
- 输出：序列表示 H、全局表示 Q、极性专家分布 P
- 训练：CE + NT-Xent +（可选）专家/意图联合
- 目标：提升情感识别与表达可控性，为 Intent Twice 与响应生成奠定基础。

---
