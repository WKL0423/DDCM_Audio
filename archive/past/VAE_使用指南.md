# AudioLDM2 VAE 测试使用指南

## 快速开始

### 1. 使用简化版本测试脚本

```bash
python simple_vae_test.py [音频文件路径] [可选：音频长度秒数]
```

**示例**:
```bash
# 测试5秒音频
python simple_vae_test.py AudioLDM2_Music_output.wav 5

# 测试10秒音频
python simple_vae_test.py techno.wav 10

# 如果不指定文件，脚本会列出当前目录的音频文件供选择
python simple_vae_test.py
```

### 2. 使用完整版本测试脚本

```bash
python test_vae_reconstruction.py
```

完整版本提供更详细的分析和多模型支持。

## 脚本功能

### `simple_vae_test.py` - 快速测试
- **默认模型**: AudioLDM2-Music
- **功能**: 
  - 加载音频文件
  - VAE 编码/解码测试
  - 计算基本质量指标
  - 保存原始和重建音频

### `test_vae_reconstruction.py` - 完整测试
- **支持多个模型变体**:
  - AudioLDM2 标准版
  - AudioLDM2-Large
  - AudioLDM2-Music
  - AudioLDM2-GigaSpeech
- **功能**:
  - 详细的测试报告
  - 更多质量指标
  - 自动模型选择
  - 生成测试报告文件

## 测试流程

1. **音频加载**: 加载音频文件，可选择长度限制
2. **Mel-spectrogram转换**: 将音频转换为mel频谱图
3. **VAE编码**: 使用AudioLDM2 VAE编码到潜在空间
4. **VAE解码**: 从潜在空间解码回mel频谱图
5. **音频重建**: 将mel频谱图转换回音频
6. **质量评估**: 计算MSE、SNR、相关系数等指标

## 输出文件

所有测试结果保存在：
- `vae_quick_test/` - 简化版本输出
- `vae_test_results/` - 完整版本输出

输出文件包括：
- `*_original_*.wav` - 处理后的原始音频
- `*_reconstructed_*.wav` - VAE重建音频
- `test_report_*.txt` - 详细测试报告（完整版本）

## 关键指标说明

- **MSE**: 均方误差，越小越好
- **SNR**: 信噪比（dB），越高越好
- **相关系数**: 与原始音频的相关性，范围[-1,1]，越接近1越好
- **压缩比**: 原始数据与压缩数据的比值
- **处理时间**: 编码和解码所需时间

## 音频压缩研究应用

这个VAE测试为音频压缩研究提供了基础：

1. **压缩效果分析**: 评估不同压缩比下的音频质量
2. **模型比较**: 测试不同AudioLDM2变体的压缩性能
3. **参数优化**: 调整mel-spectrogram参数和VAE设置
4. **质量评估**: 建立音频重建质量的评估指标

## 注意事项

- 确保GPU内存足够（建议8GB+）
- 音频文件格式支持: WAV, MP3, FLAC, M4A
- 测试时间取决于音频长度和GPU性能
- VAE重建质量受原始音频特性影响

## 故障排除

1. **内存不足**: 减少音频长度或使用CPU模式
2. **数据类型错误**: 脚本已自动处理float16/float32转换
3. **音频格式不支持**: 转换为WAV格式
4. **模型加载失败**: 检查网络连接，模型会自动下载

---

更多详细信息请参考 `VAE_测试总结.md`
