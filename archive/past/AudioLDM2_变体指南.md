# AudioLDM2 多变体使用指南

## 🎯 如何选择不同的 AudioLDM2 变体

在 `main_multi_model.py` 文件中，找到第 46 行：

```python
choice = "1"  # 默认使用标准版，可以改为 "2", "3", "4" 来选择其他变体
```

## 📋 可用变体详情

### 1. AudioLDM2 (标准版) - `choice = "1"`
- **Checkpoint**: `cvssp/audioldm2`
- **任务**: 文本转音频 (通用)
- **UNet 大小**: 350M 参数
- **总模型大小**: 1.1B
- **训练数据**: 1150k小时
- **适用场景**: 通用音频生成、音效、环境声音

### 2. AudioLDM2-Large (大型版) - `choice = "2"`
- **Checkpoint**: `cvssp/audioldm2-large`
- **任务**: 文本转音频 (高质量)
- **UNet 大小**: 750M 参数
- **总模型大小**: 1.5B
- **训练数据**: 1150k小时
- **适用场景**: 高质量音频生成，更好的细节和保真度

### 3. AudioLDM2-Music (音乐专用) - `choice = "3"`
- **Checkpoint**: `cvssp/audioldm2-music`
- **任务**: 文本到音乐
- **UNet 大小**: 350M 参数
- **总模型大小**: 1.1B
- **训练数据**: 665k小时 (音乐专用数据)
- **适用场景**: 音乐创作、乐器合成、节拍和旋律

### 4. AudioLDM2-GigaSpeech (语音) - `choice = "4"`
- **Checkpoint**: `anhnct/audioldm2_gigaspeech`
- **任务**: 文本转语音 (TTS)
- **UNet 大小**: 350M 参数
- **总模型大小**: 1.1B
- **训练数据**: 10k小时 (语音数据)
- **适用场景**: 语音合成、对话生成、配音

## 🔧 使用示例

### 生成音乐 (推荐使用音乐专用模型)
```python
choice = "3"  # 选择音乐专用模型
# 系统会自动使用: "Upbeat electronic dance music with synthesizers and drum beats"
```

### 生成语音 (使用 TTS 模型)
```python
choice = "4"  # 选择语音模型
# 系统会自动使用:
# prompt = "A female speaker with clear pronunciation"
# transcription = "Hello, this is a test of text to speech generation."
```

### 生成高质量音频 (使用大型模型)
```python
choice = "2"  # 选择大型模型
# 更好的质量，但加载和推理时间更长
```

## 📊 性能对比

| 模型 | 加载时间 | 推理速度 | 质量 | 专用性 |
|------|----------|----------|------|--------|
| 标准版 | 快 | 快 | 好 | 通用 |
| 大型版 | 慢 | 慢 | 最好 | 通用 |
| 音乐版 | 快 | 快 | 好 | 音乐 |
| 语音版 | 快 | 快 | 好 | 语音 |

## 🚀 快速切换

1. 编辑 `main_multi_model.py` 文件
2. 修改 `choice = "1"` 为你想要的选项 (1-4)
3. 保存文件
4. 运行: `python main_multi_model.py`

## 💡 提示

- **首次使用**: 每个模型首次加载时需要下载，请确保网络连接良好
- **存储空间**: 每个模型约 1-1.5GB，确保有足够存储空间
- **显存要求**: 大型版本需要更多显存
- **质量 vs 速度**: 根据需求选择合适的模型

---
创建时间: 2025年7月16日
