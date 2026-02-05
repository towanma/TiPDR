# TiPDR - Tibetan Prosody-Disentangled Representation

基于韵律解耦表征学习的藏语三大方言语音处理模型。

## 核心思想

将语音分解为三个独立的部分，然后可以任意重组：

```
        音频
         │
    ┌────┴────┐
    │  分解器  │
    └────┬────┘
   ┌─────┼─────┐
   ▼     ▼     ▼
 内容   韵律   音色      ← 三个独立表示
 (词)  (声调) (嗓音)
   │     │     │
   └─────┼─────┘
    ┌────┴────┐
    │  合成器  │
    └────┬────┘
         ▼
      新音频
```

**应用场景：**
- 安多话内容 + 拉萨话声调 → 带拉萨口音的语音
- 提取方言无关的内容单元 → 统一ASR

## 安装

```bash
# 1. 克隆项目
git clone https://github.com/towanma/TiPDR.git
cd TiPDR

# 2. 安装依赖
pip install -e .
# 或
pip install -r requirements.txt

# 3. 安装 pyworld (F0提取)
pip install pyworld
```

## 数据准备

### 目录结构

```
data/
└── audios_PDR/
    ├── lhasa/              # 卫藏方言（有声调）
    │   ├── speaker_001/
    │   │   ├── 001.wav
    │   │   └── ...
    │   └── speaker_002/
    ├── kham/               # 康巴方言（有声调）
    │   └── ...
    └── amdo/               # 安多方言（无声调）
        └── ...
```

### 预处理

```bash
# 增加文件句柄限制（避免 Too many open files 错误）
ulimit -n 4096

# 运行预处理
python scripts/preprocess_data.py \
    --data_root ./data/audios_PDR \
    --output_dir ./data \
    --num_workers 1
```

预处理后生成：
```
data/
├── metadata.json      # 文件元信息
├── splits.json        # train/val/test 划分
└── preprocessed/      # 特征文件 (.npz)
```

## 训练

```bash
python train.py --config configs/config.yaml
```

### 从检查点恢复

```bash
python train.py --config configs/config.yaml --resume checkpoints/latest.pt
```

### 多阶段训练策略

| 阶段 | Epoch | 内容 |
|-----|-------|------|
| Stage 1 | 0-50 | 仅重建损失，冻结判别器 |
| Stage 2 | 50-150 | 添加对抗训练 |
| Stage 3 | 150-200 | 添加互信息最小化 |

## 监控训练

### TensorBoard

```bash
tensorboard --logdir=logs --port=6006
# 浏览器打开 http://localhost:6006
```

### 关键指标

| 指标 | 含义 | 期望趋势 |
|-----|------|---------|
| `reconstruction_total` | 重建损失 | ↓ 下降 |
| `vq_total` | VQ量化损失 | 稳定 |
| `codebook_utilization` | 码本使用率 | >50% |
| `content_adv_loss` | 内容对抗损失 | 波动但稳定 |
| `mi_loss` | 互信息损失 | ↓ 下降 |

### 训练输出解读

```
Epoch 0: loss=36.67, recon=36.43
         ↑           ↑
      总损失      重建损失（主要指标）
```

- **recon 持续下降** → 模型在学习
- **recon 不下降** → 检查学习率或数据
- **loss 变 NaN** → 学习率太大

## 评估

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/config.yaml
```

## 项目结构

```
TiPDR/
├── configs/config.yaml       # 配置文件
├── data/
│   ├── dataset.py           # 数据集类
│   └── preprocessor.py      # 数据预处理
├── models/
│   ├── content_encoder.py   # HuBERT + VQ-VAE
│   ├── prosody_encoder.py   # F0/韵律编码
│   ├── speaker_encoder.py   # 说话人编码
│   ├── decoder.py           # 解码器
│   ├── discriminator.py     # 判别器 + GRL
│   └── disentangled_model.py
├── losses/losses.py         # 损失函数
├── utils/
│   ├── audio.py             # 音频处理
│   └── f0_extractor.py      # F0提取
├── scripts/
│   ├── preprocess_data.py
│   ├── convert_dialect.py
│   └── extract_content_units.py
├── train.py
└── evaluate.py
```

## 常见问题

### 1. Too many open files
```bash
ulimit -n 4096
# 或使用 --num_workers 1
```

### 2. 验证集为空 (val=0)
检查 `data/splits.json`，确保 val 列表不为空。可以手动将部分 train 文件移到 val。

### 3. CUDA out of memory
减小 batch_size（在 config.yaml 中修改）：
```yaml
training:
  batch_size: 8  # 默认16，内存不够可改小
```

### 4. 模块导入错误
```bash
pip install -e .
# 或
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 技术细节

### 解耦机制

1. **Instance Normalization** - 在 Content Encoder 中去除全局统计信息（声调）
2. **VQ 离散化** - 将内容量化为离散码本，自然去除连续韵律
3. **对抗训练 (GRL)** - 判别器尝试从内容预测方言，编码器学习欺骗它
4. **互信息最小化** - 强制内容和韵律表示独立

### 模型规模

- 参数量：~121M
- 主要来自 HuBERT (94M)
- 训练显存：~12GB (batch_size=16)

## License

MIT
