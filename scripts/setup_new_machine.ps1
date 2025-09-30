# PowerShell setup script for new Windows machine
# Purpose: Recreate environment, set HF cache, and verify pipeline runs
# Usage: Right-click "Run with PowerShell" or run in an elevated PowerShell

param(
    [string]$EnvName = "audioldm_env",
    [string]$PythonVersion = "3.10",
    [string]$HfHome = "F:\\Kasai_Lab\\hf_cache\\huggingface",
    [string]$RepoRoot = "F:\\Kasai_Lab\\Code\\AudioLDM_2_diffuser"
)

Write-Host "[1/6] Ensuring conda is available..."
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "conda not found in PATH. Please install Anaconda/Miniconda and reopen PowerShell."
    exit 1
}

Write-Host "[2/6] Creating/Updating conda env: $EnvName (Python $PythonVersion)"
conda env list | Select-String "\b$EnvName\b" | Out-Null
if ($LASTEXITCODE -ne 0) {
    conda create -y -n $EnvName python=$PythonVersion
}

Write-Host "[3/6] Activating env and installing dependencies (pip)"
conda activate $EnvName
# Basic deps; you can expand or pin versions as needed
pip install --upgrade pip
pip install torch==2.2.1+cpu torchvision==0.17.1+cpu torchaudio==2.2.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install diffusers==0.25.0 transformers==4.30.2 huggingface-hub==0.23.2 tokenizers==0.13.3 accelerate==0.23.0
pip install librosa==0.9.2 soundfile==0.13.1 matplotlib==3.8.1 pandas==2.1.3 einops tqdm
# Optional/used in project
pip install pytorch-lightning==2.1.1 taming-transformers-rom1504 h5py==3.10.0 webdataset wandb kornia

Write-Host "[4/6] Setting Hugging Face cache path to $HfHome"
New-Item -ItemType Directory -Force -Path $HfHome | Out-Null
[Environment]::SetEnvironmentVariable('HF_HOME', $HfHome, 'User')
# Also set hub subdir for compatibility
$Hub = Join-Path $HfHome 'hub'
[Environment]::SetEnvironmentVariable('HUGGINGFACE_HUB_CACHE', $Hub, 'User')

Write-Host "[5/6] Verifying environment variables"
Write-Host "HF_HOME=$([Environment]::GetEnvironmentVariable('HF_HOME','User'))"
Write-Host "HUGGINGFACE_HUB_CACHE=$([Environment]::GetEnvironmentVariable('HUGGINGFACE_HUB_CACHE','User'))"

Write-Host "[6/6] Running smoke test (may download model to cache if not present)"
Set-Location $RepoRoot
conda activate $EnvName
python -X faulthandler -u .\smoke_test_sampler.py

Write-Host "Done. If VS Code is open, reload the window and select the $EnvName interpreter."
