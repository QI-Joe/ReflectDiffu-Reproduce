# intent_twice 模块说明

> 实现论文中 “Intent Twice Exploring – Sampling – Correcting” 机制（与情感扩散 Emofused 表示融合），为响应生成阶段提供情感 + 意图双重校正后的高质量条件上下文。

## 目录
- [总体目标](#总体目标)
- [核心思想与三阶段流程](#核心思想与三阶段流程)
- [文件结构与职责](#文件结构与职责)
- [关键张量与形状对照表](#关键张量与形状对照表)
- [情感→群组→意图 Refer 映射](#情感群组意图-refer-映射)
- [EMU: Diffusion + 双分支CVAE 采样与融合](#emu-diffusion--双分支cvae-采样与融合)
- [IntentPolicy: REINFORCE 策略选择与奖励](#intentpolicy-reinforce-策略选择与奖励)
- [IntentTwiceModule 集成前向流程](#intenttwicemodule-集成前向流程)
- [损失构成与公式映射](#损失构成与公式映射)
- [Response Decoder（下游使用）](#response-decoder下游使用)
- [训练插入点 & 典型数据流](#训练插入点--典型数据流)
- [常见问题与排查](#常见问题与排查)
- [可扩展改进方向](#可扩展改进方向)

---
## 总体目标
1. 基于上游 Emotion Contagion / ERA 模块的情感语义表示 (Q, H, Emopos/Emoneg 初值)；
2. 通过“意图二次”机制（探索 → 采样 → 校正）生成 经情感与奖励对齐 的融合表示 `Emofused`；
3. 为后续响应解码器提供唯一跨注意 (Cross-Attention) 键值（避免信息漂移）；
4. 同时输出/更新：
   - 正向与负向情感意图分布 (Emopos, Emoneg)
   - 融合表示 Emofused
   - 采样选中的意图 id / one-hot 向量
   - KL / 校正 / 策略等损失项

---
## 核心思想与三阶段流程
| 阶段 | 名称 | 论文公式（示意） | 代码落点 | 目的 |
|------|------|------------------|----------|------|
| Exploring | Intentfirst | (5) 初始情感-意图先验估计 | `IntentTwiceModule._build_semantic_intent()` | 结合语义与外部意图预测，得到初始候选权重 |
| Sampling | Diffusion + 双CVAE | (7)(8) 逐步噪声注入/去噪 | `EMU.q_sample`, `EMU.p_sample_step` | 得到情感正/负两条潜变量轨迹 Emopos / Emoneg |
| Correcting | Intent 校正 | (10)(12) 融合 & Loss | `IntentPolicy.forward` + `IntentTwiceModule.forward` | 通过奖励与交叉熵校正意图分布，生成 Emofused |

> 注意：源码中未显式编号公式，但语义对应：初始构造 → 扩散采样 → KL 与意图校正损失 → 总体联合训练。

---
## 文件结构与职责
| 文件 | 作用 |
|------|------|
| `EMU.py` | 定义 `IntentAwareCVAE`（单分支）与 `EMU`（双分支+扩散）：正向/负向 latent 生成与 KL 累积；输出 Emopos/Emoneg/Emofused。|
| `IntentPolicy.py` | 定义 `IntentPolicy`：根据 Refer 候选构造三个意图选项，softmax 采样 + REINFORCE 奖励，产生 `Lpolicy` 与校正CE损失 `Lintent`。|
| `intent_twice_integration.py` | 定义 `EmotionMappings`（情感→群组与 refer 映射）与 `IntentTwiceModule`（统筹：mask 构建、调用 EMU & Policy、聚合损失）。|
| `intent_emotion_capture.py` | 日志辅助：`BatchIntegrator` 周期性收集情感/意图分布方便可视化或调试。|
| `response_decoder.py` | 下游响应生成：单层 Transformer Decoder + Pointer-Generator，将 `Emofused` 作为唯一 KV 融入解码。|
| `__init__.py` | 汇总导出模块入口。|

---
## 关键张量与形状对照表
| 名称 | 形状 | 产生位置 | 说明 |
|------|------|----------|------|
| `H` | `[B, L_ctx, D]` | 上游 Encoder | 对话上下文 token 编码序列 |
| `Q` | `[B, D]` | EmotionContagion mean-pool | 全局情感语义向量（池化）|
| `semantic_intent` | `[B, N_int]` | `_build_semantic_intent` | 基于语义 (p_semantic) + 外部意图 (p_intent) 加权融合 |
| `emotion_probs` | `[B, N_emotion]` | 上游情感分类器 | 每类情感概率 |
| `group_mask` | `[B, N_int]` | `compute_masks_and_groups` | 按情感群组过滤无关意图 |
| `refer_map` | 列表/索引结构 | 同上 | 每条样本对应候选意图 id 列表（Top-3）|
| `intent_candidates` | `[B, 3]` | `IntentPolicy.forward` | 三个候选意图 id（含当前选中及对立/随机）|
| `Emopos` | `[B, D]` | `EMU.forward` | 正向情感潜变量解码后表示 |
| `Emoneg` | `[B, D]` | `EMU.forward` | 负向情感潜变量解码后表示 |
| `Emofused` | `[B, L_fuse, D]` *(通常 L_fuse=2 或拼接后再扩展)* | `EMU.forward` | 将 Emopos/Emoneg 或其注意聚合后的融合序列 |
| `chosen_intent_ids` | `[B]` | `IntentPolicy.forward` | 采样结果（整数索引）|
| `intent_logits` | `[B, 3]` | 同上 | 三候选得分 |
| `Lkl_pos / Lkl_neg` | 标量/列表 | EMU | 每步或累计 KL 项 |
| `Lintent` | 标量 | Policy | 交叉熵校正意图区分损失 |
| `Lpolicy` | 标量 | Policy | REINFORCE 策略梯度项（可选计入总损失）|
| `Ltwice` | 标量 | IntentTwiceModule | KL + Lintent (+ 可选 Lpolicy) 汇总 |
| `Pw` | `[B, T_dec, V_ext]` | ResponseDecoder | 解码输出词分布 |

---
## 情感→群组→意图 Refer 映射
`EmotionMappings` 中：
1. `EMOTIONS`: 基础情感标签枚举。
2. `GROUPS`: 将若干情感聚合成上层群组（降低稀疏 & 提供共享 refer 候选）。
3. `REFER_BY_GROUP`: 每个群组对应一组常见意图 id（Top-K，源码中取前三用于候选）。
4. `polarity`：判断当前情感整体倾向（正/负）以决定奖励方向（选择 Emopos 或 Emoneg 与目标意图的相似性打分）。

构建过程要点：
- 根据 `emotion_probs` 取 argmax 得到主导情感 → 找到群组 → 查 refer 列表；
- 若 refer 数量不足 3，则补随机/默认意图；
- 生成 `refer_map`：后续 `IntentPolicy` 用于构造三候选意图。

---
## EMU: Diffusion + 双分支CVAE 采样与融合
`EMU.py` 结构：
1. `IntentAwareCVAE`：典型 (μ, logσ²) 编码 → 重参数 (z) → 解码得到情感方向向量。保留 `mu`, `logvar`, `epsilon_hat` 用于 KLD。
2. 双分支：`pos_branch`, `neg_branch` 各自独立参数，分别建模积极/消极情绪意图子空间。
3. 扩散接口：
   - `q_sample(x0, t, noise)`: 前向添加噪声（简化：线性 β schedule）。
   - `p_sample_step(xt, t)`: 逆过程一步去噪（内部通过 CVAE 得到期望方向，近似 DDPM 反演）。
4. `forward`：
   - 迭代 t=1..T：分别在正/负分支进行采样或重采；
   - 收集每步 KL (pos/neg)；
   - 最终得到 `Emopos`, `Emoneg`；
   - 融合策略：可以是简单拼接 / 加权注意 / gating（源码当前实现：返回二者及 `Emofused`）。
5. 输出：
```python
{
  'Emopos': Tensor[B, D],
  'Emoneg': Tensor[B, D],
  'Emofused': Tensor[B, L_fuse, D],
  'kl_pos_list': [...], 'kl_neg_list': [...]
}
```

> KL 项在 `IntentTwiceModule.forward` 聚合为标量：`Lkl = Σ_t (KL_pos_t + KL_neg_t)`。

---
## IntentPolicy: REINFORCE 策略选择与奖励
步骤：
1. 构造三候选：`[refer_primary, refer_contrast, refer_random]`；
2. 取 `intent_embeddings`（通常来自上游词/意图嵌入矩阵索引）；
3. 计算评分：对 Emopos / Emoneg 取加权相似（若 polarity=正 → 使用 Emopos 相似度，否则 Emoneg）；
4. `softmax` 得到 `intent_logits` → 采样 `chosen_intent_id`；
5. 奖励：与“目标方向”余弦相似度或点积（源码里以情感方向乘法/聚合形式实现）；
6. 策略梯度：`Lpolicy = - (log π(selected) * (reward - baseline))`；
7. 校正 CE：`Lintent = CrossEntropy(intent_logits, target_candidate_index)`（target 可能是 refer_primary 或工程定义标签）。

> 训练中可选择：`总损失 = Ltwice (+ λ_policy * Lpolicy)`，源码默认 `Ltwice` 聚合 KL + Lintent（`Lpolicy` 监控即可）。

---
## IntentTwiceModule 集成前向流程
伪代码梳理：
```python
def forward(H, Q, emotion_probs, ext_intent_probs, semantic_intent_probs, ...):
    # 1. 构造语义意图融合
    semantic_intent = α * semantic_intent_probs + (1-α) * ext_intent_probs

    # 2. 依据主导情感 -> 群组 -> refer_map (Top-3 意图)
    group_mask, refer_map = compute_masks_and_groups(emotion_probs)

    # 3. EMU 双分支扩散采样 + 融合
    emu_out = EMU(Q or H 或二者组合)
    Emopos, Emoneg, Emofused = emu_out

    # 4. 策略网络 (IntentPolicy)
    policy_out = IntentPolicy(Emopos, Emoneg, refer_map, polarity)

    # 5. 损失聚合
    Lkl = sum(emu_out.kl_pos_list + emu_out.kl_neg_list)
    Ltwice = Lkl + policy_out.Lintent  # (+ 可选 γ * policy_out.Lpolicy)

    return {Emopos, Emoneg, Emofused, chosen_intent_ids, Ltwice, ...}
```

---
## 损失构成与公式映射
| 组件 | 符号 | 公式含义 | 源码变量 | 说明 |
|------|------|----------|----------|------|
| 正向 KL | KL_pos | (7)/(8) 扩散正向CVAE正则 | `kl_pos_list` | 对所有时间步求和 |
| 负向 KL | KL_neg | 同上（负向分支） | `kl_neg_list` | 对所有时间步求和 |
| 意图校正 CE | L_intent | (10) 校正项 | `Lintent` | 区分正确 refer 候选 |
| 策略梯度 | L_policy | 探索奖励 | `Lpolicy` | 可选加入；默认监控 |
| 综合 | L_twice | (12) 子模块总体 | `Ltwice` | 实现中 = KL_pos+KL_neg + Lintent |

> 总体训练（与情感、ERA、解码等）可再加权：`L_total = L_emotion + δ L_contrast + ζ L_ERA + η L_twice + θ L_decoder (+ ...)`。

---
## Response Decoder（下游使用）
`response_decoder.py` 提供符合论文约束的轻量单层解码器：
1. `Emofused` 作为 Cross-Attention 唯一 KV（确保条件集中）；
2. Pointer-Generator：`Pw = p_mix * Pgen + (1-p_mix) * Pcopy`；
3. 可选 Coverage Loss 减少重复注意；
4. 输出：`Pw / Pgen / Pcopy / p_mix / attn_weights / context_vector` 及损失 (`loss`, `ce_loss`, `coverage_loss`)。

### 解码输入/输出简表
| 名称 | 形状 | 说明 |
|------|------|------|
| `trg_input_ids` | `[B, T_dec]` | 解码端输入（带起始符）|
| `Emofused` | `[B, L_fuse, D]` | 上游 Intent Twice 融合表示 |
| `src_token_ids` | `[B, L_src]` | 用于 Pointer Copy（需与 Emofused 对齐或截断 / pad）|
| `Pw` | `[B, T_dec, V_ext]` | 最终扩展词表概率 |
| `p_mix` | `[B, T_dec, 1]` | 生成 vs 复制 混合门 |

---
## 训练插入点 & 典型数据流
```
数据(batch) → 上游Tokenizer/Embedding → EmotionContagion Encoder
    → 得到 H, Q, emotion_probs
    → ERA (可选) 提供 reasoning mask / 额外特征
    → IntentTwiceModule.forward(H, Q, emotion_probs, intent_probs, ...)
        → Emopos / Emoneg / Emofused / Ltwice
    → ResponseDecoder(trg_input_ids, Emofused, src_token_ids)
        → Pw + 解码损失
    → 聚合全部损失反传
```

---
## 常见问题与排查
| 现象 | 可能原因 | 排查建议 |
|------|----------|----------|
| KL → 0 且不收敛 | β schedule 太小或重参数被梯度消失 | 调整 β 上限；确保 `logvar` 未被 clamp 过强 |
| Lintent 不下降 | refer_map 候选无区分度 | 检查情感→群组映射；增加负采样多样性 |
| 采样模式塌陷（总选同一意图） | reward 设计过于单调 | 引入 entropy 正则或温度 τ > 1.0 |
| 解码重复片段 | coverage 未启用或 p_mix 极端 | 打开 `coverage_weight`；监控 p_mix 分布 |
| Emofused 与 Emopos/Emoneg 几乎相等 | 融合策略退化 | 引入可学习 gating 或注意力融合层 |

---
## 可扩展改进方向
1. Diffusion 深化：使用余弦调度 + 学习式噪声预测网络（替换当前简化 p_sample_step）。
2. 意图候选生成：从静态 refer → 动态检索（相似度 Top-K）以提升开放域泛化。
3. 融合层：多头情感-意图交互注意力替代简单拼接，输出更细粒度 `L_fuse > 2` 序列。
4. 策略优化：REINFORCE → PPO / A2C；或引入 critic 网络减少方差。
5. 多任务标签：同时预测意图强度（回归）用于 reward 加权。
6. 统一损失调度：KLD cyclical annealing 防止 posterior collapse。

---
## 术语速查
| 缩写 | 含义 |
|------|------|
| CVAE | Conditional Variational Auto-Encoder |
| KL / KLD | Kullback–Leibler Divergence |
| REINFORCE | Monte-Carlo Policy Gradient 算法 |
| Pointer-Generator | 结合生成与复制机制的混合概率模型 |

---
## 快速使用示例（伪）
```python
# 1. 上游得到 H, Q, emotion_probs, intent_probs
intent_twice = IntentTwiceModule(config)
twice_out = intent_twice(
    H=H,
    Q=Q,
    emotion_probs=emotion_probs,
    ext_intent_probs=ext_intent_probs,
    semantic_intent_probs=semantic_intent_probs,
)

decoder = create_paper_compliant_decoder(vocab_size)
dec_out = decoder(
    trg_input_ids=trg_in,
    emofused=twice_out['Emofused'],
    src_token_ids=src_ids,
    gold_ids=gold_ids,
)

loss = (twice_out['Ltwice'] * eta) + dec_out['loss'] * theta
loss.backward()
```

---
## 结语
`intent_twice` 模块实现了论文中以“情感 + 意图”双向校正为核心的中间层，它以轻量扩散 + 策略梯度结合的方式，在不显著增加解码复杂度的前提下增强对生成语义的约束。理解并正确监控 KL、Lintent、Emofused 质量，是取得预期效果的关键。
