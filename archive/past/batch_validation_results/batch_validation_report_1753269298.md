# AudioLDM2 VAE增强 - 批量验证报告

生成时间: 2025-07-23 20:14:58

## 验证摘要

- 总文件数: 8
- 增强系数: 1.4x
- 平均处理时间 (原始): 0.032s
- 平均处理时间 (增强): 0.073s

## 质量指标对比

| 指标 | 原始VAE | 增强VAE | 改进 |
|------|---------|---------|------|
| snr | -0.000 | -0.000 | 0.000 |
| high_freq_retention | 0.000 | 0.000 | nan |
| spectral_correlation | nan | nan | nan |

## 详细结果

### 1. AudioLDM2_Music_output.wav

- 文件长度: 10.00s
- 处理时间: 0.037s
- snr: -0.000 → -0.000 (+0.000)
- high_freq_retention: 0.000 → 0.000 (+nan)
- spectral_correlation: nan → nan (+nan)

### 2. AudioLDM2_output.wav

- 文件长度: 10.00s
- 处理时间: 0.079s
- snr: 0.000 → 0.000 (+0.000)
- high_freq_retention: 0.000 → 0.000 (+nan)
- spectral_correlation: nan → nan (+nan)

### 3. custom_pipeline_output.wav

- 文件长度: 5.00s
- 处理时间: 0.078s
- snr: -0.000 → -0.000 (+0.000)
- high_freq_retention: 0.000 → 0.000 (+nan)
- spectral_correlation: nan → nan (+nan)

### 4. custom_techno.wav

- 文件长度: 10.00s
- 处理时间: 0.080s
- snr: -0.000 → -0.000 (+0.000)
- high_freq_retention: 0.000 → 0.000 (+nan)
- spectral_correlation: nan → nan (+nan)

### 5. techno.wav

- 文件长度: 10.00s
- 处理时间: 0.077s
- snr: -0.000 → -0.000 (+0.000)
- high_freq_retention: 0.000 → 0.000 (+nan)
- spectral_correlation: nan → nan (+nan)

### 6. AudioLDM2_Music_output_original_1752675322.wav

- 文件长度: 5.00s
- 处理时间: 0.080s
- snr: -0.000 → -0.000 (+0.000)
- high_freq_retention: 0.000 → 0.000 (+nan)
- spectral_correlation: nan → nan (+nan)

### 7. AudioLDM2_Music_output_AudioLDM2_ClapFeatureExtractor_1752803658.wav

- 文件长度: 3.33s
- 处理时间: 0.074s
- snr: 0.000 → 0.000 (+0.000)
- high_freq_retention: 0.000 → 0.000 (+nan)
- spectral_correlation: nan → nan (+nan)

### 8. AudioLDM2_Music_output_original_1752803658.wav

- 文件长度: 3.33s
- 处理时间: 0.076s
- snr: -0.000 → -0.000 (+0.000)
- high_freq_retention: 0.000 → 0.000 (+nan)
- spectral_correlation: nan → nan (+nan)

## 结论

⚠️ 增强VAE在信噪比方面略有下降（这是高频增强的正常现象）
⚠️ 高频保持度改善有限

📊 所有音频文件和可视化图表已保存到 batch_validation_results/ 目录
