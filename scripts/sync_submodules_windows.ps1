param(
    [switch]$UseGhProxy,
    [switch]$RepairIncomplete
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Resolve-Url([string]$Url) {
    if ($UseGhProxy -and $Url.StartsWith("https://github.com/")) {
        return "https://gh-proxy.com/$Url"
    }
    return $Url
}

$Submodules = @(
    @{ Path = "submodules\diff-gaussian-rasterization"; Url = "https://github.com/ashawkey/diff-gaussian-rasterization"; Marker = "setup.py" },
    @{ Path = "submodules\diff-gaussian-rasterization-w-pose"; Url = "https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose"; Marker = "setup.py" },
    @{ Path = "submodules\simple-knn"; Url = "https://gitlab.inria.fr/bkerbl/simple-knn.git"; Marker = "setup.py" },
    @{ Path = "submodules\pytorch3d"; Url = "https://github.com/facebookresearch/pytorch3d"; Marker = "setup.py" },
    @{ Path = "submodules\minLoRA"; Url = "https://github.com/cccntu/minLoRA"; Marker = "setup.py" },
    @{ Path = "submodules\CLIP"; Url = "https://github.com/openai/CLIP"; Marker = "setup.py" },
    @{ Path = "submodules\segment-anything"; Url = "https://github.com/facebookresearch/segment-anything"; Marker = "setup.py" },
    @{ Path = "submodules\croco"; Url = "https://github.com/naver/croco"; Marker = "models\curope\setup.py" }
)

$NestedSubmodules = @(
    @{ Path = "submodules\diff-gaussian-rasterization\third_party\glm"; Url = "https://github.com/g-truc/glm.git"; Marker = "glm\glm.hpp" },
    @{ Path = "submodules\diff-gaussian-rasterization-w-pose\third_party\glm"; Url = "https://github.com/g-truc/glm.git"; Marker = "glm\glm.hpp" }
)

foreach ($Submodule in $Submodules) {
    $Path = Join-Path $Root $Submodule.Path
    $MarkerPath = Join-Path $Path $Submodule.Marker

    if (Test-Path $MarkerPath) {
        Write-Host "Submodule ready: $($Submodule.Path)"
        continue
    }

    $HasFiles = (Test-Path $Path) -and ((Get-ChildItem $Path -Force | Select-Object -First 1) -ne $null)
    if ($HasFiles) {
        if ($RepairIncomplete) {
            Write-Host "Removing incomplete submodule directory: $($Submodule.Path)"
            Remove-Item -LiteralPath $Path -Recurse -Force
        } else {
            throw "Incomplete submodule directory '$($Submodule.Path)' exists but '$($Submodule.Marker)' is missing. Rerun with -RepairIncomplete."
        }
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    $Url = Resolve-Url $Submodule.Url
    Write-Host "Cloning $Url -> $($Submodule.Path)"
    git clone --depth 1 $Url $Path
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone $($Submodule.Path). You can rerun this script after the network recovers."
    }
}

foreach ($Submodule in $NestedSubmodules) {
    $Path = Join-Path $Root $Submodule.Path
    $MarkerPath = Join-Path $Path $Submodule.Marker

    if (Test-Path $MarkerPath) {
        Write-Host "Nested submodule ready: $($Submodule.Path)"
        continue
    }

    $PrimaryGlm = Join-Path $Root "submodules\diff-gaussian-rasterization\third_party\glm"
    $PrimaryGlmMarker = Join-Path $PrimaryGlm "glm\glm.hpp"
    if (($Submodule.Path -eq "submodules\diff-gaussian-rasterization-w-pose\third_party\glm") -and (Test-Path $PrimaryGlmMarker)) {
        if (Test-Path $Path) {
            Write-Host "Replacing incomplete nested submodule directory from primary GLM: $($Submodule.Path)"
            Remove-Item -LiteralPath $Path -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
        Copy-Item -LiteralPath $PrimaryGlm -Destination $Path -Recurse
        Write-Host "Copied primary GLM -> $($Submodule.Path)"
        continue
    }

    $HasFiles = (Test-Path $Path) -and ((Get-ChildItem $Path -Force | Select-Object -First 1) -ne $null)
    if ($HasFiles) {
        if ($RepairIncomplete) {
            Write-Host "Removing incomplete nested submodule directory: $($Submodule.Path)"
            Remove-Item -LiteralPath $Path -Recurse -Force
        } else {
            throw "Incomplete nested submodule directory '$($Submodule.Path)' exists but '$($Submodule.Marker)' is missing. Rerun with -RepairIncomplete."
        }
    } elseif (Test-Path $Path) {
        Remove-Item -LiteralPath $Path -Force
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    $Url = Resolve-Url $Submodule.Url
    Write-Host "Cloning $Url -> $($Submodule.Path)"
    git clone --depth 1 $Url $Path
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone $($Submodule.Path). You can rerun this script after the network recovers."
    }
}

Write-Host "Submodule sync finished."
