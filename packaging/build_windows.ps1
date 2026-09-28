# Build DRISHTI-3D for Windows (x64), with NVIDIA CUDA acceleration.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -Cuda cu130
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -CpuOnly
#
# Produces dist\DRISHTI-3D\DRISHTI-3D.exe and dist\DRISHTI-3D-windows-x64.zip.
# Needs: uv (https://docs.astral.sh/uv/), git, and for CUDA an NVIDIA driver
# new enough for the chosen CUDA build (cu126: driver >= 560; cu130: >= 580).
# PyTorch publishes torch 2.14.0 for Windows as cu126, cu130 and cu132 only.
#
# PyPI's Windows torch wheels are CPU-only, so after the locked environment
# is synced the same torch/torchvision versions are reinstalled from
# PyTorch's CUDA index. On a machine without an NVIDIA GPU the CUDA build
# still runs (device selection falls back to CPU).

param(
    [string]$Cuda = "cu126",
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/"
}

$py = ".venv\Scripts\python.exe"

Write-Host "==> syncing the locked environment (gui + ml + semantics + reference)"
uv sync --locked --extra gui --extra ml --extra semantics --extra reference
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

if (-not $CpuOnly) {
    $torch = & $py -c "import torch; print(torch.__version__.split('+')[0])"
    if ($LASTEXITCODE -ne 0) { throw "torch import failed" }
    $vision = & $py -c "import torchvision; print(torchvision.__version__.split('+')[0])"
    if ($LASTEXITCODE -ne 0) { throw "torchvision import failed" }
    Write-Host "==> installing CUDA ($Cuda) builds of torch $torch / torchvision $vision"
    uv pip install --python $py --reinstall "torch==$torch" "torchvision==$vision" --index-url "https://download.pytorch.org/whl/$Cuda"
    if ($LASTEXITCODE -ne 0) { throw "CUDA torch install failed (try -Cuda cu126, or -CpuOnly)" }
}

& $py -c "import torch; print('  torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "torch import failed" }

Write-Host "==> rendering the app icon"
$env:QT_QPA_PLATFORM = "offscreen"
& $py packaging\make_icon.py --ico packaging\DRISHTI-3D.ico
if ($LASTEXITCODE -ne 0) { throw "icon rendering failed" }
Remove-Item Env:QT_QPA_PLATFORM

Write-Host "==> building"
& $py -m PyInstaller packaging\drishti3d.spec --noconfirm --distpath dist --workpath build\pyi
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Write-Host "==> smoke test: bundled imports"
& $py packaging\smoke_test.py dist\DRISHTI-3D\DRISHTI-3D.exe
if ($LASTEXITCODE -ne 0) { throw "bundled application smoke test failed" }
Write-Host "==> packaging"
Compress-Archive -Path dist\DRISHTI-3D -DestinationPath dist\DRISHTI-3D-windows-x64.zip -Force
Write-Host "Built: dist\DRISHTI-3D\DRISHTI-3D.exe  (zip: dist\DRISHTI-3D-windows-x64.zip)"
Write-Host "CLI:   dist\DRISHTI-3D\DRISHTI-3D.exe run VIDEO --telemetry LOG --out DIR"
