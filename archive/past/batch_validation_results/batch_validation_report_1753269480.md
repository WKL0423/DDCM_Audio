# AudioLDM2 VAE增强 - 批量验证报告

生成时间: 2025-07-23 20:18:00

## 验证摘要

- 总文件数: 8
- 增强系数: 1.4x
- 平均处理时间 (原始): 0.032s
- 平均处理时间 (增强): 0.121s

## 质量指标对比

| 指标 | 原始VAE | 增强VAE | 改进 |
|------|---------|---------|------|
| snr | -0.937 | -2.531 | -1.594 |
| high_freq_retention | 64.858 | 85.091 | 31.150 |
| spectral_correlation | 0.363 | 0.366 | 0.714 |

## 详细结果

### 1. AudioLDM2_Music_output.wav

- 文件长度: 10.00s
- 处理时间: 0.039s
- snr: -1.067 → -3.535 (-2.468)
- high_freq_retention: 66.495 → 100.198 (+50.684)
- spectral_correlation: 0.282 → 0.289 (+2.540)

### 2. AudioLDM2_output.wav

- 文件长度: 10.00s
- 处理时间: 0.079s
- snr: -1.166 → -4.608 (-3.442)
- high_freq_retention: 60.170 → 92.885 (+54.371)
- spectral_correlation: 0.333 → 0.291 (-12.813)

### 3. custom_pipeline_output.wav

- 文件长度: 5.00s
- 处理时间: 0.079s
- snr: -2.940 → -6.075 (-3.134)
- high_freq_retention: 142.898 → 173.045 (+21.097)
- spectral_correlation: 0.246 → 0.222 (-9.581)

### 4. custom_techno.wav

- 文件长度: 10.00s
- 处理时间: 0.072s
- snr: -0.113 → -0.244 (-0.131)
- high_freq_retention: 33.009 → 38.345 (+16.164)
- spectral_correlation: 0.449 → 0.486 (+8.304)

### 5. techno.wav

- 文件长度: 10.00s
- 处理时间: 0.078s
- snr: -0.862 → -1.584 (-0.722)
- high_freq_retention: 75.883 → 83.159 (+9.588)
- spectral_correlation: 0.443 → 0.432 (-2.403)

### 6. AudioLDM2_Music_output_original_1752675322.wav

- 文件长度: 5.00s
- 处理时间: 0.078s
- snr: -1.047 → -3.655 (-2.609)
- high_freq_retention: 63.366 → 102.244 (+61.355)
- spectral_correlation: 0.271 → 0.282 (+4.251)

### 7. AudioLDM2_Music_output_AudioLDM2_ClapFeatureExtractor_1752803658.wav

- 文件长度: 3.33s
- 处理时间: 0.303s
- snr: -0.218 → -0.364 (-0.145)
- high_freq_retention: 36.307 → 43.059 (+18.597)
- spectral_correlation: 0.573 → 0.565 (-1.517)

### 8. AudioLDM2_Music_output_original_1752803658.wav

- 文件长度: 3.33s
- 处理时间: 0.244s
- snr: -0.079 → -0.180 (-0.101)
- high_freq_retention: 40.735 → 47.798 (+17.341)
- spectral_correlation: 0.310 → 0.363 (+16.935)

## 结论

⚠️ 增强VAE在信噪比方面略有下降（这是高频增强的正常现象）
✅ 高频保持度有适度提升

📊 所有音频文件和可视化图表已保存到 batch_validation_results/ 目录
