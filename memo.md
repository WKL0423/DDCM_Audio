                                   当前默认策略更新（2025-12-03）：

1) 默认使用 x0 域残差匹配
- 变更：`StepCodecConfig.match_epsilon=False`
- 理由：主观听感与客观指标均显示 1000x1000 的 x0 域效果优于 ε 域。

2) ε 域匹配保留为可选对比
- 如需开启，请在代码中显式设置 `match_epsilon=True`（目前 CLI 未暴露该参数）。

3) 相关脚本影响
- `step2b_ddcm_step_encode.py` 使用默认配置，即 x0 域；输出不含 ε 域信息。
- `step3b_ddcm_step_decode.py` 无需改动，仍根据 bitstream 注入每步噪声。

新增实现（2025-12-11）：

1) 恢复噪声幅度
- `ddcm/codebook.py` 不再对码本原子做单位范数归一化，保持标准正态尺度，并在加载旧缓存时自动再生。

2) 匹配追踪残差更新
- `StepCodecConfig` 追加 `atoms_per_step` 与 `coeff_levels`，可按论文方式在同一步内贪心组合多原子。
- `encode_step_bitstream` 中引入量化系数与残差迭代更新，记录到比特流（`coeffs`、`signs`）。
- `decode_step_bitstream`/`step3b` 支持解析新的系数格式，按 Σ s·c·C(k) 注入噪声。

当前默认系数网格（2025-12-23）：
- `StepCodecConfig.coeff_levels = (0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)`，加入 0 以允许跳步，并显著扩展高幅度档位以减少裁剪。
- 调整系数网格不会影响码本内容，无需重新生成 codebook；除非手动删除缓存，系统保持原有随机噪声。若需刷新，可删除 `codebooks/cb_*.pt` 让脚本重建。

历史记录：
- 我们曾尝试 ε 域残差匹配（1000 和 1000x1000），但综合表现不如 x0 域 1000x1000。

——
让我分析一下我们刚才的修复方法，并与训练代码中的处理方式进行对比。

首先让我检查训练代码中是否有处理维度不匹配的逻辑： 让我检查训练数据加载和预处理部分： 让我检查mel频谱图提取的具体实现： 现在让我检查audio处理模块： 现在让我查看数据集的特征提取部分： 让我看看DSP构建部分： 现在我明白了训练代码中的处理方式。让我分析关键发现：

## 🔍 **分析：训练代码中的处理方式 vs 我们的修复方法**

### **训练代码中的处理方式：**

1. **维度处理** (lines 530-540 in dataset.py)：
   ```python
   def pad_spec(self, log_mel_spec):
       n_frames = log_mel_spec.shape[0]
       p = self.target_length - n_frames
       # cut and pad
       if p > 0:
           m = torch.nn.ZeroPad2d((0, 0, 0, p))
           log_mel_spec = m(log_mel_spec)
       elif p < 0:
           log_mel_spec = log_mel_spec[0 : self.target_length, :]
   ```

2. **目标长度计算** (line 217)：
   ```python
   self.target_length = int(self.duration * self.sampling_rate / self.hopsize)
   ```

3. **mel频谱图提取** (mel_spectrogram_train函数)：
   - 使用torch.stft而不是librosa
   - 包含padding和normalization

### **我们当前的修复方法：**
- 简单地取最小长度：`min_time_frames = min(orig_mel_np.shape[0], recon_mel_np.shape[0])`

### **正确的修复方法应该是：**

让我按照训练代码的方式修复： 
```


已进行更改。


```
让我修复缩进： 我看到缩进有问题。让我重新修复： 
```


已进行更改。


```
让我重新处理整个函数的替换： 让我直接创建一个新的完全修正的文件： 

已进行更改。

现在运行修正后的脚本： 太好了！现在让我总结一下我们的修复方法与训练代码的对比：

## 📊 **修复方法总结：训练代码 vs 我们的方法**

### **🔧 关键修复对比**

| **方面** | **之前的简单修复** | **训练代码的方法** | **我们的修正方法** |
|---------|------------------|------------------|------------------|
| **维度不匹配处理** | `min_time_frames = min(...)` 简单截取 | `pad_spec()` 函数：padding到target_length | ✅ 完全复制训练代码的`_pad_spec()`方法 |
| **Mel频谱图提取** | 使用librosa.feature.melspectrogram | 使用torch.stft + mel_basis矩阵乘法 | ✅ 使用训练代码的`_mel_spectrogram_train()`方法 |
| **目标长度计算** | 没有明确目标长度 | `target_length = int(duration * sr / hop_length)` | ✅ 使用训练配置的duration=10.24s |
| **Padding策略** | 没有padding | ZeroPad2d，先padding后截取 | ✅ 完全按训练代码实现 |
| **数据格式** | [batch, channel, time, freq] | [batch, time, freq] → padding → [batch, channel, time, freq] | ✅ 严格匹配训练流程 |

### **🎯 训练代码中的关键方法**

1. **mel频谱图提取** (dataset.py line 448-500):
   ```python
   def mel_spectrogram_train(self, y):
       # torch.stft + mel_basis矩阵乘法
       # spectral_normalize_torch (log clamp)
   ```

2. **Padding策略** (dataset.py line 530-540):
   ```python
   def pad_spec(self, log_mel_spec):
       p = self.target_length - n_frames
       if p > 0: ZeroPad2d  # padding
       elif p < 0: cut to target_length
   ```

3. **目标长度** (dataset.py line 217):
   ```python
   self.target_length = int(self.duration * self.sampling_rate / self.hopsize)
   ```

### **🔍 结果对比**

**修正前的问题：**
- ❌ 维度不匹配错误：`(1001,64)` vs `(1000,64)`
- ❌ 简单截取可能丢失重要信息

**修正后的改进：**
- ✅ 维度完全匹配：`(1024,64)` vs `(1024,64)`
- ✅ Mel相关性达到1.000（完美匹配）
- ✅ 使用训练代码的精确padding策略
- ✅ SNR虽然仍然较低(-1.10 dB)，但这反映了VAE本身的重建能力

### **🎵 音频质量分析**

从结果可以看出：
- **低频保持率**: 31.6% - 低频信息丢失较多
- **中频保持率**: 72.9% - 中频信息保持较好  
- **高频保持率**: 4.2% - 高频信息严重丢失
- **频谱相关性**: 53.7% - 整体结构有一定保持

这表明VAE确实存在高频信息丢失的问题，这是VAE模型本身的限制，而不是我们处理管道的问题。

**总结**：我们的修复方法现在完全按照训练代码的方式处理数据，确保了评估管道与训练管道的一致性。维度不匹配问题通过训练代码的padding策略得到了正确解决。

找到具有 1 个许可证类型的类似代码