"""
检查 AudioLDM2Pipeline 的实际组件和属性
"""

import torch
from diffusers import AudioLDM2Pipeline
import warnings
warnings.filterwarnings("ignore")

def inspect_audioldm2_pipeline():
    """检查 AudioLDM2Pipeline 的组件"""
    print("🔍 检查 AudioLDM2Pipeline 组件...")
    
    # 加载 pipeline
    pipeline = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-music",
        torch_dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📋 Pipeline 类型: {type(pipeline)}")
    print(f"📋 Pipeline 属性:")
    
    # 列出所有属性
    for attr_name in dir(pipeline):
        if not attr_name.startswith('_'):
            attr_value = getattr(pipeline, attr_name)
            if not callable(attr_value):
                print(f"   {attr_name}: {type(attr_value)}")
    
    print(f"\n🧩 Pipeline 组件:")
    # 检查关键组件
    key_components = ['vae', 'unet', 'scheduler', 'text_encoder', 'tokenizer', 'feature_extractor']
    
    for component_name in key_components:
        if hasattr(pipeline, component_name):
            component = getattr(pipeline, component_name)
            print(f"   ✅ {component_name}: {type(component)}")
            
            # 对于某些组件，显示更多信息
            if component_name == 'vae':
                print(f"      VAE config: {component.config}")
            elif component_name == 'feature_extractor':
                print(f"      Feature extractor type: {type(component)}")
                if hasattr(component, 'sampling_rate'):
                    print(f"      Sampling rate: {component.sampling_rate}")
        else:
            print(f"   ❌ {component_name}: 未找到")
    
    # 检查 pipeline 的 __call__ 方法签名
    print(f"\n🔧 Pipeline 方法:")
    methods = [method for method in dir(pipeline) if callable(getattr(pipeline, method)) and not method.startswith('_')]
    key_methods = ['__call__', 'encode_prompt', 'decode_latents']
    
    for method_name in key_methods:
        if method_name in methods:
            print(f"   ✅ {method_name}: 可用")
        else:
            print(f"   ❌ {method_name}: 未找到")
    
    # 尝试生成一个简单的音频
    print(f"\n🎵 测试音频生成...")
    try:
        with torch.no_grad():
            audio = pipeline("piano music", num_inference_steps=2, audio_length_in_s=2.0)
        print(f"   ✅ 生成成功: {type(audio)}")
        if hasattr(audio, 'audios'):
            print(f"      音频形状: {audio.audios.shape}")
        elif hasattr(audio, 'audio'):
            print(f"      音频形状: {audio.audio.shape}")
        else:
            print(f"      返回类型: {type(audio)}")
            if hasattr(audio, 'shape'):
                print(f"      形状: {audio.shape}")
    except Exception as e:
        print(f"   ❌ 生成失败: {e}")

if __name__ == "__main__":
    inspect_audioldm2_pipeline()
