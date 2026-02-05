# Prosody-Disentangled Tibetan HuBERT

基于韵律解耦表征学习的藏语三大方言语音处理模型。

## 项目概述

本项目实现了一个**显式韵律解耦表征**模型，专门针对藏语三大方言（卫藏/拉萨话、康巴话、安多话）的语音特点设计：

- **卫藏话(Lhasa)**: 有声调，依靠声调辨义
- **康巴话(Kham)**: 有声调
- **安多话(Amdo)**: 无声调，依靠辅音组合辨义

### 核心创新

1. **三路解耦架构**:
   - Content Encoder: 基于HuBERT + VQ-VAE，提取"说什么"
   - Prosody Encoder: 提取F0/声调/节奏信息
   - Speaker Encoder: 提取说话人音色特征

2. **对抗互信息最小化**: 确保内容编码不包含韵律信息

3. **方言适应**: 自动处理有调/无调方言差异

## 项目结构

```
tibetan_hubert/
├── configs/
│   └── config.yaml          # 训练配置
├── data/
│   ├── dataset.py           # 数据集类
│   └── preprocessor.py      # 数据预处理
├── models/
│   ├── content_encoder.py   # 内容编码器 (HuBERT + VQ)
│   ├── prosody_encoder.py   # 韵律编码器
│   ├── speaker_encoder.py   # 说话人编码器
│   ├── decoder.py           # 解码器
│   ├── discriminator.py     # 判别器 + GRL
│   └── disentangled_model.py # 主模型
├── losses/
│   └── losses.py            # 损失函数
├── utils/
│   ├── audio.py             # 音频处理
│   └── f0_extractor.py      # F0提取
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
└── requirements.txt         # 依赖
```

## 安装

```bash
pip install -r requirements.txt
```

## 数据准备

### 数据格式

```
data/
├── lhasa/              # 卫藏方言
│   ├── speaker_001/
│   │   ├── audio_001.wav
│   │   └── ...
│   └── ...
├── kham/               # 康巴方言
│   └── ...
└── amdo/               # 安多方言
    └── ...
```

### 预处理

```bash
python -c "
from data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(
    sample_rate=16000,
    hop_length=320,
    n_mels=80,
    f0_method='dio'
)

preprocessor.preprocess_dataset(
    data_root='path/to/raw/data',
    metadata_path='data/metadata.json',
    num_workers=4
)
"
```

## 训练

### 基础训练

```bash
python train.py --config configs/config.yaml
```

### 从检查点恢复

```bash
python train.py --config configs/config.yaml --resume checkpoints/latest.pt
```

### 多阶段训练

模型采用三阶段训练策略：

1. **Stage 1 (50 epochs)**: 仅重建损失，冻结判别器
2. **Stage 2 (100 epochs)**: 添加对抗训练
3. **Stage 3 (50 epochs)**: 添加互信息最小化

## 评估

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --config configs/config.yaml
```

### 评估指标

- **重建质量**: Mel MSE, MCD
- **解耦程度**: 
  - Content不应能预测方言 (低准确率 = 好)
  - Prosody应能预测方言 (高准确率 = 好)
- **ABX测试**: 内容不变性
- **说话人验证**: EER

## 应用场景

### 1. 方言转换

```python
model = ProsodyDisentangledModel.load('checkpoint.pt')

# 将安多话内容 + 卫藏话韵律 = 卫藏话
converted_mel = model.convert_dialect(
    content_audio=amdo_audio,
    prosody_audio=lhasa_audio,
    speaker_audio=target_speaker_audio
)
```

### 2. 方言无关ASR

```python
# 只使用内容编码进行ASR
content, indices = model.encode_content(audio)
# indices 是方言无关的离散内容单元
```

### 3. 声调分析

```python
# 使用韵律编码分析声调模式
prosody = model.encode_prosody(f0, energy)
```

## 配置说明

关键配置参数 (`configs/config.yaml`):

```yaml
model:
  content_encoder:
    pretrained_hubert: "facebook/hubert-base-ls960"
    output_size: 256
    use_instance_norm: true  # 关键：去除韵律信息
  
  vq:
    num_embeddings: 512      # 内容码本大小
    commitment_cost: 0.25
  
  prosody_encoder:
    output_size: 64
  
  speaker_encoder:
    output_size: 128

training:
  loss_weights:
    reconstruction: 1.0
    vq_commitment: 0.25
    content_adversarial: 0.1  # GRL权重
    mi_minimization: 0.05     # 互信息最小化
```

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{tibetan_hubert,
  title={Prosody-Disentangled Speech Representation for Tibetan Dialects},
  author={Your Name},
  year={2024}
}
```

## 参考文献

1. HuBERT: Self-Supervised Speech Representation Learning
2. VQ-VAE: Neural Discrete Representation Learning
3. SpeechSplit: Unsupervised Speech Decomposition
4. VQMIVC: Vector Quantization and Mutual Information
