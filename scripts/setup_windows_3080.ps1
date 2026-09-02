param(
    [string]$EnvName = "gaussianobject-py311",
    [switch]$CleanCondaCache,
    [switch]$UseTunaMirror,
    [switch]$SkipSubmodules,
    [switch]$BuildCroco
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:CONDA_NO_PLUGINS = "true"
$env:TORCH_CUDA_ARCH_LIST = "8.6"
$env:MAX_JOBS = "4"
$LocalCondaPkgs = Join-Path $Root ".conda-pkgs"
New-Item -ItemType Directory -Force -Path $LocalCondaPkgs | Out-Null
$env:CONDA_PKGS_DIRS = $LocalCondaPkgs
Write-Host "Using local conda package cache: $LocalCondaPkgs"

if ($CleanCondaCache) {
    Write-Host "Cleaning conda index cache"
    conda clean -i -y
}

if ($UseTunaMirror) {
    Write-Host "Using TUNA conda mirrors for this command"
    $CreateChannels = @(
        "-c", "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main",
        "-c", "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r",
        "-c", "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2",
        "--override-channels"
    )
} else {
    $CreateChannels = @()
}

if (-not $SkipSubmodules) {
    Write-Host "Checking submodules"
    try {
        & (Join-Path $Root "scripts\sync_submodules_windows.ps1")
    } catch {
        throw "Submodule sync failed. If GitHub is slow, try: .\scripts\sync_submodules_windows.ps1 -UseGhProxy"
    }
}

$EnvList = conda env list
if (-not ($EnvList -match "(^|\s)$EnvName(\s|$)")) {
    Write-Host "Creating conda env: $EnvName"
    conda --no-plugins create --solver=classic -n $EnvName python=3.11 -y @CreateChannels
    if ($LASTEXITCODE -ne 0) {
        throw "Conda env creation failed. Try: .\scripts\setup_windows_3080.ps1 -CleanCondaCache -UseTunaMirror"
    }
} else {
    Write-Host "Conda env already exists: $EnvName"
}

Write-Host "Installing PyTorch CUDA 11.8 wheels"
if ($UseTunaMirror) {
    conda run -n $EnvName python -m pip install --upgrade pip wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
} else {
    conda run -n $EnvName python -m pip install --upgrade pip wheel
}
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
if ($UseTunaMirror) {
    conda run -n $EnvName python -m pip install --force-reinstall setuptools==69.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
} else {
    conda run -n $EnvName python -m pip install --force-reinstall setuptools==69.5.1
}
if ($LASTEXITCODE -ne 0) { throw "setuptools pin installation failed." }
conda run -n $EnvName python -m pip install --force-reinstall torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
conda run -n $EnvName python -m pip install --force-reinstall numpy==1.26.4
if ($LASTEXITCODE -ne 0) { throw "NumPy pin installation failed." }

Write-Host "Installing PyPI requirements"
if ($UseTunaMirror) {
    conda run -n $EnvName python -m pip install -r requirements-windows-pypi.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
} else {
    conda run -n $EnvName python -m pip install -r requirements-windows-pypi.txt
}
if ($LASTEXITCODE -ne 0) { throw "PyPI requirements installation failed." }

Write-Host "Re-pinning build-critical packages"
if ($UseTunaMirror) {
    conda run -n $EnvName python -m pip install --force-reinstall numpy==1.26.4 setuptools==69.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
} else {
    conda run -n $EnvName python -m pip install --force-reinstall numpy==1.26.4 setuptools==69.5.1
}
if ($LASTEXITCODE -ne 0) { throw "Build-critical package pinning failed." }

Write-Host "Installing local CUDA/source packages without build isolation"
$LocalRequirements = Get-Content (Join-Path $Root "requirements-windows-local.txt") | Where-Object { $_.Trim().Length -gt 0 -and -not $_.Trim().StartsWith("#") }
foreach ($LocalPackage in $LocalRequirements) {
    Write-Host "Installing $LocalPackage"
    $LocalPackagePath = Join-Path $Root ($LocalPackage -replace "^\./", "")
    if ((Test-Path (Join-Path $LocalPackagePath "setup.py")) -and ($LocalPackage -like "./submodules/*")) {
        Push-Location $LocalPackagePath
        conda run -n $EnvName python setup.py install
        $InstallExitCode = $LASTEXITCODE
        Pop-Location
    } else {
        conda run -n $EnvName python -m pip install --no-build-isolation --no-deps --no-use-pep517 $LocalPackage
        $InstallExitCode = $LASTEXITCODE
    }
    if ($InstallExitCode -ne 0) { throw "Local CUDA/source package installation failed at $LocalPackage." }
}

if ($BuildCroco) {
    Write-Host "Building optional CroCo CUDA extension"
    Push-Location "submodules\croco\models\curope"
    conda run -n $EnvName python setup.py build_ext --inplace
    Pop-Location
}

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "conda activate $EnvName"
Write-Host ""
Write-Host "Next required downloads:"
Write-Host "  Push-Location models"
Write-Host "  conda run -n $EnvName python download_hf_models.py"
Write-Host "  Pop-Location"
Write-Host "  Download SAM/DUSt3R weights into models\ if you use COLMAP-free mode."
