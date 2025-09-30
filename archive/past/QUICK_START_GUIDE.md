# 快速启动指南 (Quick Start Guide)

## 🚀 一键运行推荐脚本

### 1. 快速 VAE 音频重建 (推荐新手)
```bash
# 最稳定的VAE重建，适合快速测试
python vae_hifigan_ultimate_fix.py
```
**输出**: `AudioLDM2_Music_output.wav` (重建音频)
**耗时**: ~30秒
**特点**: 稳定、快速、基础质量保证

### 2. 高频修复 VAE 重建 (推荐高质量需求)
```bash
# 高频保持最佳的VAE版本
python v5_high_freq_fix.py
```
**输出**: `AudioLDM2_Music_output.wav` (高频优化音频)
**耗时**: ~45秒
**特点**: 高频保持、细节丰富

### 3. 创新 DDCM 输入相关生成 ⭐⭐ (项目核心成果)
```bash
# 基于输入音频的DDCM生成，输出与输入相关
python audioldm2_ddcm_input_based_fixed.py
```
**输出**: `ddcm_input_based_output/` 目录下的音频文件
**耗时**: ~2-3分钟
**特点**: 输入相关、创新算法、高质量

### 4. 改进版 DDCM 生成 (软量化优化)
```bash
# 使用软量化策略的改进DDCM
python audioldm2_ddcm_improved.py
```
**输出**: `improved_ddcm_output/` 目录下的音频文件
**耗时**: ~2-3分钟
**特点**: 软量化、混合策略、质量提升

### 5. 完整 Diffusion 管道 (高质量但慢)
```bash
# 完整的AudioLDM2扩散管道
python audioldm2_full_diffusion_fixed.py
```
**输出**: `AudioLDM2_output.wav`
**耗时**: ~5-10分钟
**特点**: 最高质量、完整管道、计算密集

## 📊 质量验证与分析

### 6. DDCM 相关性验证分析
```bash
# 验证DDCM输出与输入的相关性
python verify_ddcm_correlation.py
```
**输出**: 控制台分析报告 + 相关性图表
**功能**: 分析输入输出相关性、质量指标

### 7. 综合方法对比分析
```bash
# 对比所有方法的性能和质量
python ddcm_final_summary.py
```
**输出**: 详细的对比分析报告
**功能**: 全面评估、方法对比、推荐建议

### 8. VAE vs Diffusion 深度对比
```bash
# 深入分析VAE和Diffusion的差异
python diffusion_vs_vae_analysis.py
```
**输出**: VAE vs Diffusion 对比报告
**功能**: 技术对比、性能分析、适用场景

## 🎯 按需求选择脚本

### 需求 1: 快速音频重建 (≤ 1分钟)
```bash
python vae_hifigan_ultimate_fix.py
```

### 需求 2: 高质量音频重建 (1-2分钟)
```bash
python v5_high_freq_fix.py
```

### 需求 3: 输入相关的音频变换 (2-3分钟)
```bash
python audioldm2_ddcm_input_based_fixed.py
```

### 需求 4: 最高质量生成 (5-10分钟)
```bash
python audioldm2_full_diffusion_fixed.py
```

### 需求 5: 创新算法研究 (2-3分钟)
```bash
python audioldm2_ddcm_improved.py
```

## 📁 输出文件说明

### 主要输出文件:
- `AudioLDM2_Music_output.wav` - VAE重建音频
- `AudioLDM2_output.wav` - Diffusion生成音频
- `ddcm_input_based_output/reconstructed_*.wav` - DDCM输入相关音频
- `improved_ddcm_output/improved_*.wav` - 改进DDCM音频

### 分析报告:
- 控制台输出包含详细的质量指标
- SNR (信噪比)、相关性、MAE、高频保持率等

## ⚡ 性能与时间预估

| 脚本 | 预估时间 | 质量等级 | 适用场景 |
|------|----------|----------|----------|
| VAE Ultimate Fix | 30秒 | ⭐⭐⭐ | 快速测试 |
| V5 High Freq | 45秒 | ⭐⭐⭐⭐ | 高频需求 |
| DDCM Input Based | 2-3分钟 | ⭐⭐⭐⭐⭐ | 输入相关 |
| DDCM Improved | 2-3分钟 | ⭐⭐⭐⭐⭐ | 创新研究 |
| Full Diffusion | 5-10分钟 | ⭐⭐⭐⭐⭐ | 最高质量 |

## 🔧 环境要求

### 必要依赖:
```bash
# 确保已安装必要的包
pip install torch torchaudio transformers diffusers
pip install librosa soundfile matplotlib numpy
```

### GPU 推荐:
- 最低: 6GB VRAM (GTX 1060 6GB)
- 推荐: 8GB+ VRAM (RTX 3070+)
- 最佳: 12GB+ VRAM (RTX 3080+)

## 🎵 测试音频

### 默认测试文件:
- `techno.wav` - 电子音乐测试
- `test_audio_reference.wav` - 标准参考音频

### 自定义音频:
将您的音频文件放在项目根目录，脚本会自动检测和处理。

## 🆘 故障排除

### 常见问题:
1. **CUDA 内存不足**: 重启 Python，或使用更小的批处理大小
2. **音频格式问题**: 确保音频为 WAV 格式，采样率 16kHz
3. **依赖缺失**: 运行 `pip install -r requirements.txt`

### 调试脚本:
```bash
# 检查环境和依赖
python debug_latent_dims.py

# 诊断问题文件
python diagnose_problem_files.py
```

---

**开始建议**: 新用户建议从 `vae_hifigan_ultimate_fix.py` 开始，熟悉后尝试 `audioldm2_ddcm_input_based_fixed.py` 体验创新功能。

**高级用户**: 直接使用 DDCM 相关脚本和分析工具进行深度研究。
