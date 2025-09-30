#!/usr/bin/env python3
"""
AudioLDM2 项目脚本清理和整理工具
"""

import os
import shutil
from pathlib import Path
import sys

# 定义过时的脚本（可以安全删除）
OUTDATED_SCRIPTS = [
    'vae_noise_fix_test.py',
    'vae_noise_fix_v2.py',
    'vae_quality_fixer.py',
    'vae_quick_improver.py',
    'test_vae_reconstruction.py',
    'ultimate_vae_test.py',
    'ultimate_vae_reconstruction.py',
    'stable_vae_test.py',
    'simple_stable_vae_test.py',
    'vae_comparison_test.py',
    'vae_hifigan_fixed.py'  # 被 vae_hifigan_ultimate.py 替代
]

# 定义核心脚本（重要，不要删除）
CORE_SCRIPTS = [
    'simple_vae_test.py',
    'vae_final_noise_fix.py',
    'vae_hifigan_ultimate.py',
    'guided_diffusion_reconstruction.py',  # 创新的引导式diffusion重建
    'diagnose_problem_files.py',
    'vocoder_analysis.py',
    'audio_fix.py',
    'test_playback_compatibility.py'
]

# 定义主应用脚本
MAIN_SCRIPTS = [
    'main.py',
    'main_enhanced_fixed.py',
    'main_multi_model.py',
    'New_pipeline_audioldm2.py'
]

def create_backup():
    """创建备份文件夹"""
    backup_dir = Path("backup_scripts")
    backup_dir.mkdir(exist_ok=True)
    return backup_dir

def list_current_scripts():
    """列出当前目录中的所有Python脚本"""
    current_dir = Path(".")
    py_files = list(current_dir.glob("*.py"))
    return [f.name for f in py_files]

def categorize_scripts():
    """分类现有脚本"""
    current_scripts = list_current_scripts()
    
    categories = {
        'core': [],
        'main': [],
        'outdated': [],
        'tools': [],
        'unknown': []
    }
    
    for script in current_scripts:
        if script in CORE_SCRIPTS:
            categories['core'].append(script)
        elif script in MAIN_SCRIPTS:
            categories['main'].append(script)
        elif script in OUTDATED_SCRIPTS:
            categories['outdated'].append(script)
        elif script.startswith(('audio_', 'test_', 'diagnose_', 'windows_')):
            categories['tools'].append(script)
        else:
            categories['unknown'].append(script)
    
    return categories

def display_categories(categories):
    """显示脚本分类"""
    print("📊 脚本分类结果:")
    print("=" * 60)
    
    print("\n🎯 核心脚本 (重要，建议保留):")
    for script in categories['core']:
        print(f"  ✅ {script}")
    
    print("\n🏗️ 主应用脚本:")
    for script in categories['main']:
        print(f"  📱 {script}")
    
    print("\n🛠️ 工具脚本:")
    for script in categories['tools']:
        print(f"  🔧 {script}")
    
    print("\n🗑️ 过时脚本 (可以删除):")
    for script in categories['outdated']:
        print(f"  ❌ {script}")
    
    print("\n❓ 未分类脚本 (需要手动检查):")
    for script in categories['unknown']:
        print(f"  ❓ {script}")

def clean_outdated_scripts(categories, backup_dir, dry_run=True):
    """清理过时脚本"""
    outdated = categories['outdated']
    
    if not outdated:
        print("\n✅ 没有找到过时的脚本")
        return
    
    print(f"\n🗑️ 准备处理 {len(outdated)} 个过时脚本:")
    
    for script in outdated:
        script_path = Path(script)
        if script_path.exists():
            if dry_run:
                print(f"  🔍 [模拟] 将移动: {script} -> {backup_dir}/{script}")
            else:
                try:
                    backup_path = backup_dir / script
                    shutil.move(str(script_path), str(backup_path))
                    print(f"  ✅ 已移动: {script} -> {backup_dir}/{script}")
                except Exception as e:
                    print(f"  ❌ 移动失败 {script}: {e}")
        else:
            print(f"  ⚠️ 文件不存在: {script}")

def analyze_diffusion_capabilities():
    """分析项目中的diffusion能力"""
    print(f"\n🔬 AudioLDM2 Diffusion能力分析:")
    print("=" * 60)
    
    # 检查完整diffusion pipeline
    full_diffusion_scripts = []
    vae_only_scripts = []
    
    # 主要diffusion脚本
    diffusion_files = [
        'main.py',
        'main_enhanced_fixed.py', 
        'main_multi_model.py',
        'New_pipeline_audioldm2.py'
    ]
    
    # VAE专用脚本
    vae_files = [
        'simple_vae_test.py',
        'vae_final_noise_fix.py',
        'vae_hifigan_ultimate.py'
    ]
    
    print("\n🎯 完整Diffusion Pipeline (Text → Audio):")
    for script in diffusion_files:
        if os.path.exists(script):
            print(f"  ✅ {script} - 完整text-to-audio生成")
            full_diffusion_scripts.append(script)
        else:
            print(f"  ❌ {script} - 文件不存在")
    
    print("\n🔧 VAE重建测试 (Audio → Latent → Audio):")
    for script in vae_files:
        if os.path.exists(script):
            print(f"  ✅ {script} - VAE编码/解码测试")
            vae_only_scripts.append(script)
        else:
            print(f"  ❌ {script} - 文件不存在")
    
    # 分析diffusion组件
    print(f"\n📊 Diffusion组件分析:")
    if os.path.exists('New_pipeline_audioldm2.py'):
        print(f"  ✅ 自定义AudioLDM2Pipeline实现")
        print(f"  ✅ UNet2D扩散模型")
        print(f"  ✅ Scheduler (噪音调度器)")
        print(f"  ✅ VAE (变分自编码器)")
        print(f"  ✅ HiFiGAN Vocoder")
        print(f"  ✅ CLAP文本编码器")
        print(f"  ✅ T5文本编码器")
        print(f"  ✅ GPT2语言模型")
    else:
        print(f"  ⚠️ 缺少自定义pipeline实现")
    
    # 功能对比
    print(f"\n🎭 功能对比:")
    print(f"┌─────────────────────────┬─────────────┬─────────────┐")
    print(f"│ 脚本类型                │ Diffusion   │ 生成类型    │")
    print(f"├─────────────────────────┼─────────────┼─────────────┤")
    print(f"│ main.py等               │ ✅ 完整      │ Text→Audio  │")
    print(f"│ vae_*.py                │ ❌ 仅VAE     │ Audio→Audio │")
    print(f"│ New_pipeline_audioldm2  │ ✅ 框架      │ 框架定义    │")
    print(f"└─────────────────────────┴─────────────┴─────────────┘")
    
    return {
        'full_diffusion': full_diffusion_scripts,
        'vae_only': vae_only_scripts,
        'has_diffusion': len(full_diffusion_scripts) > 0
    }


def organize_output_directories():
    """整理输出目录"""
    current_dir = Path(".")
    output_dirs = [d for d in current_dir.iterdir() if d.is_dir() and ('vae_' in d.name or 'test_' in d.name)]
    
    if not output_dirs:
        print("\n✅ 没有找到需要整理的输出目录")
        return
    
    print(f"\n📁 找到 {len(output_dirs)} 个输出目录:")
    
    # 创建outputs主目录
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    for dir_path in output_dirs:
        print(f"  📂 {dir_path.name}")
        
        # 检查目录大小
        total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        
        file_count = len(list(dir_path.rglob('*.wav')))
        print(f"     💾 {total_size_mb:.1f}MB, {file_count}个WAV文件")
        
        # 检查是否有最近的文件
        recent_files = []
        for f in dir_path.rglob('*.wav'):
            age_hours = (time.time() - f.stat().st_mtime) / 3600
            if age_hours < 24:  # 24小时内
                recent_files.append(f)
        
        if recent_files:
            print(f"     🕐 有 {len(recent_files)} 个最近文件 (24小时内)")
        else:
            print(f"     🕐 没有最近文件")

def main():
    """主函数"""
    print("🧹 AudioLDM2 项目脚本清理工具")
    print("=" * 50)
    
    # 分析当前脚本
    categories = categorize_scripts()
    display_categories(categories)
      # 显示清理选项
    print(f"\n🛠️ 清理选项:")
    print("1. 模拟清理 (安全查看)")
    print("2. 执行清理 (移动过时脚本到backup)")
    print("3. 仅显示分类")
    print("4. 整理输出目录")
    print("5. 分析Diffusion能力")
    print("6. 退出")
      try:
        choice = input("\n请选择操作 (1-6): ").strip()
        
        if choice == "1":
            backup_dir = create_backup()
            print(f"\n🔍 模拟清理模式 (备份目录: {backup_dir})")
            clean_outdated_scripts(categories, backup_dir, dry_run=True)
            
        elif choice == "2":
            confirm = input("\n⚠️ 确认执行清理? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                backup_dir = create_backup()
                print(f"\n🗑️ 执行清理 (备份目录: {backup_dir})")
                clean_outdated_scripts(categories, backup_dir, dry_run=False)
                print(f"\n✅ 清理完成！过时脚本已移动到 {backup_dir}")
            else:
                print("❌ 取消清理")
                
        elif choice == "3":
            print("\n✅ 分类显示完成")
            
        elif choice == "4":
            organize_output_directories()
            
        elif choice == "5":
            diffusion_analysis = analyze_diffusion_capabilities()
            
        elif choice == "6":
            print("👋 退出")
            
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n👋 用户取消")
    except Exception as e:
        print(f"\n❌ 错误: {e}")

if __name__ == "__main__":
    import time
    main()
