# The BC pipeline on Small, sized for the Windows PC (Ryzen 7 7700: 8 cores / 16 threads, 32 GB, RTX 5070).
# Native Windows, PowerShell, from the repository folder:
#
#     powershell -ExecutionPolicy Bypass -File experiments\bc\run_small_pc.ps1           # CPU training
#     powershell -ExecutionPolicy Bypass -File experiments\bc\run_small_pc.ps1 -Cuda     # train on the GPU
#     powershell -ExecutionPolicy Bypass -File experiments\bc\run_small_pc.ps1 "--stages=select,confirm,publish"   # quote comma lists
#
# Time: collecting the teacher's data 2 h (14 workers, about 800 episodes); training 5 variants about 15 min with
# -Cuda, or 2.5 to 3 h on the CPU (then pass "--variants=plain_u4,lp_u4,lp_u8", quoted, to halve it); choosing and confirming
# about 1 h, most of it computing the reference costs of new episode sets (once; cached afterwards).
# So about 3.5 h with -Cuda (recommended), 5 to 6 h without.
# Run it again after a crash, a reboot or Ctrl+C: finished chunks and trained candidates are kept
# (outputs\bc\small_pc\). The result: agents\bc_small_best\ and outputs\bc\small_pc\REPORT.md.
# sbf check needs Linux: copy agents\bc_small_best\ to the laptop and run `uv run sbf check bc_small_best --task=small`.
param([switch]$Cuda)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

if (-not (Test-Path ".venv")) { uv sync }  # once; later runs keep the environment (and a CUDA PyTorch) as it is

if ($Cuda) {
    # PyPI's PyTorch for Windows has no CUDA. The RTX 5070 (Blackwell) needs a CUDA 12.8 build: install it once.
    $ok = uv run --no-sync python -c "import torch; print(torch.cuda.is_available())"
    if ($ok -ne "True") {
        uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
    }
}

# --no-sync: do not put the locked (CPU) PyTorch back over the CUDA one
uv run --no-sync python experiments\bc\pipeline.py `
    --task=small --run=small_pc --publish=bc_small_best `
    --episodes=1000 --collect_hours=2 --workers=14 `
    @args
