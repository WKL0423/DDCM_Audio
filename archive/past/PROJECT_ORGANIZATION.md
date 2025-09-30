# AudioLDM2 VAE 项目脚本整理
## 项目结构整理与脚本分类

### 📂 当前脚本状态分析

根据我们的工作历程，当前有以下几个主要脚本类别：

## 🎯 **推荐使用的核心脚本**

### 1. **simple_vae_test.py** ⭐⭐⭐⭐⭐
**状态**: ✅ 最新版本，已修复兼容性
**功能**: 
- AudioLDM2 VAE 快速重建测试
- 使用soundfile (PCM_16) 保存，兼容性最佳
- 集成了多种vocoder方法 (HiFiGAN + Griffin-Lim)
- 完整的性能指标计算

**使用方法**:
```bash
python simple_vae_test.py techno.wav 5
```

### 2. **vae_final_noise_fix.py** ⭐⭐⭐⭐
**状态**: ✅ 已更新兼容性保存
**功能**:
- 专门解决HiFiGAN噪音问题
- 高级降噪和后处理
- 现已修复音频兼容性问题
- 使用soundfile保存PCM_16格式

**使用方法**:
```bash
python vae_final_noise_fix.py techno.wav 3
```

### 3. **diagnose_problem_files.py** ⭐⭐⭐
**状态**: ✅ 新创建的工具
**功能**:
- 诊断音频文件兼容性问题
- 自动修复格式问题
- 批量处理问题文件

**使用方法**:
```bash
python diagnose_problem_files.py
```

## 🧪 **实验性脚本**

### 4. **vae_hifigan_ultimate.py** ⭐⭐⭐
**状态**: ⚠️ 当前查看的文件，需要更新
**功能**:
- HiFiGAN集成的终极尝试
- 精确的维度匹配
- 需要添加兼容性保存

### 5. **vocoder_analysis.py** ⭐⭐
**状态**: ✅ 分析工具
**功能**:
- 深入分析vocoder内部结构
- 帮助理解HiFiGAN参数需求

## 📁 **需要整理的脚本分类**

### 🗑️ **可以删除的过时脚本**
这些是开发过程中的中间版本，已被更好的版本替代：

1. `vae_noise_fix_test.py` - 被`vae_final_noise_fix.py`替代
2. `vae_noise_fix_v2.py` - 被`vae_final_noise_fix.py`替代  
3. `vae_quality_fixer.py` - 功能已整合到其他脚本
4. `vae_quick_improver.py` - 被`simple_vae_test.py`替代
5. `test_vae_reconstruction.py` - 功能重复
6. `ultimate_vae_test.py` - 功能重复
7. `ultimate_vae_reconstruction.py` - 功能重复
8. `stable_vae_test.py` - 被`simple_vae_test.py`替代
9. `simple_stable_vae_test.py` - 功能重复
10. `vae_comparison_test.py` - 实验性，不再需要

### 📚 **保留的工具脚本**
1. `audio_fix.py` - 通用音频诊断工具
2. `test_playback_compatibility.py` - 播放兼容性测试
3. `windows_audio_fix.py` - Windows特定修复

### 🏗️ **主要应用脚本**
1. `main.py` - 主要的AudioLDM2应用
2. `main_enhanced_fixed.py` - 增强版本
3. `main_multi_model.py` - 多模型版本
4. `New_pipeline_audioldm2.py` - 新管道实现

## 🎯 **推荐的工作流程**

### 日常VAE测试:
```bash
python simple_vae_test.py [audio_file] [duration]
```

### 噪音问题修复:
```bash
python vae_final_noise_fix.py [audio_file] [duration]
```

### 文件兼容性问题:
```bash
python diagnose_problem_files.py
```

### 深度分析:
```bash
python vocoder_analysis.py
```

## 📊 **输出目录说明**

- `vae_quick_test/` - simple_vae_test.py的输出
- `vae_final_noise_fix/` - vae_final_noise_fix.py的输出
- `vae_final_noise_fix_repaired/` - 修复后的兼容文件
- `vae_hifigan_final_test/` - vae_hifigan_ultimate.py的输出

## 🔄 **待完成的整理任务**

1. ✅ 修复vae_final_noise_fix.py的兼容性 (已完成)
2. ⏳ 更新vae_hifigan_ultimate.py使用兼容性保存
3. ⏳ 删除过时的脚本文件
4. ⏳ 创建统一的配置文件
5. ⏳ 整理输出目录结构

---

**建议**: 当前最稳定和实用的是 `simple_vae_test.py` 和 `vae_final_noise_fix.py`。
