# ACP Track-B 训练与推理流程

> 本文档解释 Ancient Chinese Phonology (ACP) Track-B 的完整训练和推理流程。

---

## 目录

1. [整体架构](#1-整体架构)
2. [数据流程](#2-数据流程)
3. [训练流程](#3-训练流程)
4. [推理流程](#4-推理流程)
5. [评估流程](#5-评估流程)
6. [模型说明](#6-模型说明)
7. [快速开始](#7-快速开始)

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ACP Track-B 流程                              │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   数据准备    │ -> │  BERT预训练   │ -> │  下游任务     │
└──────────────┘    └──────────────┘    └──────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    v                      v                      v
              ┌──────────┐          ┌──────────┐          ┌──────────┐
              │    DT     │          │    CT     │          │   GTeT   │
              │Decision   │          │  Cognate  │          │   GT     │
              │  Tree     │          │Transfomer │          │ Enhanced │
              └──────────┘          └──────────┘          └──────────┘
                    │                      │                      │
                    └──────────────────────┼──────────────────────┘
                                           │
                                           v
                                    ┌──────────────┐
                                    │   评估与聚合   │
                                    └──────────────┘
```

### 模型说明

| 模型 | 全称 | 用途 |
|------|------|------|
| RD | Random | 随机基线 |
| MC | Majority Class | 多数类基线 |
| DT | Decision Tree | 决策树下游任务 |
| CT | Cognate Transformer | 词族Transformer下游任务 |
| GTeT | GT-enhanced Transformer | GT增强Transformer下游任务 |

---

## 2. 数据流程

### 2.1 数据文件

```
corpus/
├── train.txt          # 训练集 (~15,333 条)
├── dev.txt           # 验证集 (~1,668 条)
├── train_smoke.txt   # 小规模训练集 (100条，用于测试)
└── dev_smoke.txt    # 小规模验证集 (10条，用于测试)

data/
├── han_seq.json      # 汉字字典
├── config.json       # BERT配置
└── split_manifest.json  # 数据分割清单
```

### 2.2 数据格式

每行是一个JSON对象，包含：
- `id`: 唯一标识
- `text`: 古汉语音义对应文本
- `target_vector`: 目标向量（32维，对应4个stage × 8个slot）

---

## 3. 训练流程

### 3.1 BERT 预训练

**目的**: 学习古汉语字符的表示

```bash
cd Ancient-Chinese-Phonology/train

# 运行训练
python -u train.py \
    --output_dir repro/seed_runs_native/42 \
    --train_dir ../corpus/train.txt \
    --dev_dir ../corpus/dev.txt \
    --char_dict ../data/han_seq.json \
    --name pretrain_native \
    --seed 42 \
    --total_train_steps 100000 \
    --batch_size 128 \
    --saving_steps 10000 \
    --logging_steps 100 \
    --learning_rate 0.0001
```

**参数说明**:
- `total_train_steps`: 总训练步数 (推荐 100000)
- `batch_size`: 批次大小 (推荐 128)
- `gradient_accumulation_steps`: 梯度累积 (默认1，可用8达到 effective batch 1024)
- `saving_steps`: 每多少步保存checkpoint
- `learning_rate`: 学习率 (推荐 0.0001)

### 3.2 下游任务训练

BERT预训练完成后，使用预训练权重初始化下游模型（DT/CT/GTeT）。

**当前仓库中的模型**: 决策树/Transformer下游任务，直接使用BERT的embedding输出

---

## 4. 推理流程

### 4.1 推理脚本

使用 `inference_from_checkpoint.py` 从checkpoint生成预测：

```bash
cd Ancient-Chinese-Phonology/repro

python inference_from_checkpoint.py \
    --checkpoint repro/seed_runs_native/42/model/xxx.ckp \
    --split-manifest repro/splits_seed43_run1/table5_random_split/split_manifest.json \
    --csv dataset/ancient_chinese_phonology.csv \
    --char-dict data/han_seq.json \
    --config data/config.json \
    --out output.json
```

### 4.2 各模型推理矩阵生成

```bash
# DT (Decision Tree)
python generate_dt_stage_matrix.py

# CT (Cognate Transformer)
python generate_ct_stage_matrix.py

# GTeT (GT-enhanced Transformer)
python generate_gtet_stage_matrix.py

# RD (Random) & MC (Majority Class)
python generate_rd_mc_stage_matrix.py
```

每个脚本会：
1. 加载对应seed的checkpoint
2. 对dev集进行推理
3. 生成stage presence矩阵

---

## 5. 评估流程

### 5.1 F1 评估

使用 `eval_acp_f1.py` 计算各模型的F1分数：

```bash
python eval_acp_f1.py \
    --targets data/targets/targets.jsonl \
    --split-manifest data/splits/split_manifest.json \
    --matrix-dir repro/seed_runs_native/
```

### 5.2 聚合多seed结果

```bash
python aggregate_seeds.py \
    --matrices-dir repro/seed_runs_native/ \
    --output-dir repro/aggregated/
```

---

## 6. 模型说明

### 6.1 BERT 预训练模型

```
Net(BertForMaskedLM)
├── BertModel (12层Transformer)
│   ├── Embeddings
│   ├── Encoder (12 x BertLayer)
│   │   └── Self-Attention + FFN
│   └── Pooler
└── MLM Head (预测masked字符)
```

### 6.2 下游任务

仓库中的DT/CT/GTeT是"下游任务"模型，它们：
1. 使用BERT预训练的embedding作为输入特征
2. 训练分类器预测stage presence

### 6.3 预测全0的问题

**原因**: 之前的训练只运行了1步 (total_train_steps=1)，BERT权重≈随机初始化

**解决方案**: 
1. 确保 transformers 版本兼容 (4.30.0)
2. 使用正确的参数运行训练 (100000步)
3. 训练完成后进行推理

---

## 7. 快速开始

### 7.1 环境配置

```bash
# 安装依赖
pip install transformers==4.30.0 torch

# 确保GPU可用
nvidia-smi
```

### 7.2 快速测试训练流程

```bash
cd Ancient-Chinese-Phonology

# 方式1: 使用脚本
./train_acp_smoke.sh smoke    # 10步测试
./train_acp_smoke.sh quick    # 100步测试
./train_acp_smoke.sh full     # 完整训练

# 方式2: 手动运行
cd train
python -u train.py \
    --output_dir ../repro/smoke_test \
    --train_dir ../corpus/train_smoke.txt \
    --dev_dir ../corpus/dev_smoke.txt \
    --char_dict ../data/han_seq.json \
    --name smoke_test \
    --seed 42 \
    --total_train_steps 10 \
    --batch_size 32
```

### 7.3 完整训练流程

```bash
# 1. BERT预训练 (5个seed)
for seed in 42 43 44 45 46; do
    python train.py \
        --output_dir repro/seed_runs_native/$seed \
        --train_dir corpus/train.txt \
        --dev_dir corpus/dev.txt \
        --char_dict data/han_seq.json \
        --name pretrain_native \
        --seed $seed \
        --total_train_steps 100000 \
        --batch_size 128 \
        --saving_steps 10000
done

# 2. 生成各模型推理矩阵
python repro/generate_dt_stage_matrix.py
python repro/generate_ct_stage_matrix.py
python repro/generate_gtet_stage_matrix.py

# 3. 评估
python repro/eval_acp_f1.py --targets data/targets/targets.jsonl

# 4. 聚合
python repro/aggregate_seeds.py
```

---

## 常见问题

### Q: 为什么DT/CT/GTeT预测全0？
A: 之前训练只运行了1步，BERT权重≈随机初始化。现在已修复，可以正常运行100000步训练。

### Q: 如何确认训练正常？
A: 检查日志中loss是否下降（如17→1.7），checkpoint大小是否~1.2GB

### Q: checkpoint保存失败？
A: 可能是磁盘空间不足，清理后重试

---

## 文件索引

| 文件 | 用途 |
|------|------|
| `train/train.py` | BERT预训练脚本 |
| `repro/inference_from_checkpoint.py` | 推理脚本 |
| `repro/generate_dt_stage_matrix.py` | DT推理矩阵生成 |
| `repro/generate_ct_stage_matrix.py` | CT推理矩阵生成 |
| `repro/generate_gtet_stage_matrix.py` | GTeT推理矩阵生成 |
| `repro/eval_acp_f1.py` | F1评估脚本 |
| `repro/aggregate_seeds.py` | 多seed结果聚合 |
| `train_acp_smoke.sh` | 快速训练脚本 |
