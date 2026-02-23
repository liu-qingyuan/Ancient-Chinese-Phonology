# ACP Full Training Guide

## 问题诊断

### 现状
当前所有模型（DT、CT、GTeT）的评估结果全是 0，原因是**训练没有真正执行**。

### 根因
训练日志显示：
```
total_train_steps=1
saving_steps=1
```

模型只训练了 1 步就保存了，相当于没有训练。

---

## 快速开始

### 1. dry run 测试（推荐先跑）
```bash
cd Ancient-Chinese-Phonology
./train_acp_full.sh dry
```

这会运行 1 步训练，验证代码能正常工作。

### 2. 训练单个 seed
```bash
cd Ancient-Chinese-Phonology
./train_acp_full.sh 42
```

### 3. 训练所有 seeds
```bash
cd Ancient-Chinese-Phonology
./train_acp_full.sh
```

---

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total_train_steps` | 100000 | 总训练步数 |
| `--batch_size` | 128 | 批大小 |
| `--gradient_accumulation_steps` | 8 | 梯度累积（有效 batch = 128 × 8 = 1024） |
| `--learning_rate` | 0.0001 | Adam 学习率 |
| `--saving_steps` | 10000 | 每 N 步保存一次 checkpoint |
| `--logging_steps` | 1000 | 每 N 步打印日志 |

---

## 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | 16GB VRAM | 24GB VRAM |
| 显存 | 12GB | 20GB+ |
| 训练时间 | ~2 小时/epoch | ~1 小时/epoch |

---

## 内存优化

如果遇到 OOM：

1. **减少 batch_size**：
```bash
BATCH_SIZE=64 ./train_acp_full.sh 42
```

2. **增加梯度累积步数**：
```bash
BATCH_SIZE=64 GRAD_ACCUM_STEPS=16 ./train_acp_full.sh 42
```

3. **监控显存**：
```bash
watch -n 1 nvidia-smi
```

---

## 监控

### 查看训练日志
```bash
tail -f repro/seed_runs_native/42/training.log
```

### 查看 GPU 使用
```bash
nvidia-smi -l 1
```

### 查看 loss 曲线
训练过程中 loss 会打印到日志。

---

## 预期结果

### 正确训练后的期望值

| 模型 | 预期 F1 范围 | 说明 |
|------|-------------|------|
| RD (随机) | 0.1-0.3 | 随机基线 |
| MC (多数类) | ~majority rate | 多数类预测 |
| DT (决策树) | 0.3-0.6 | 有监督基线 |
| CT (完全失败) | 0.0 | 预期失败 |
| GTeT (Transformer) | 0.6-0.9 | 主要模型 |

### 错误现象

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 全 0 | 训练步数太少 | 增加 total_train_steps |
| loss 不下降 | 学习率问题 | 调整 learning_rate |
| OOM | 显存不足 | 减少 batch_size |

---

## 训练完成后

训练完成后，运行评估：
```bash
cd Ancient-Chinese-Phonology/repro

# 生成输入
python generate_gtet_stage_matrix.py --seeds 42 --tables table5_random_split --cells T

# 运行评估
python eval_acp_f1.py --input input.json --protocol acp_f1_protocol_v1.json --out eval.json
```

---

## 常见问题

**Q: 训练多长时间？**  
A: 100000 步大约需要 2-4 小时（取决于 GPU）。

**Q: 可以中断后继续吗？**  
A: 可以，checkpoint 已保存。重新运行会继续训练。

**Q: 需要多少 GPU 显存？**  
A: 16GB 最低，24GB 推荐。

**Q: 如何知道训练是否正常？**  
A: 观察 loss 是否下降。正常情况下 loss 应该从 ~5-10 降到 ~1-2。
