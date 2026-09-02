param(
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$DeleteTargets = @(
    # Local caches / temporary folders
    ".conda-pkgs",
    ".venv",
    "segment_anything.ipynb",

    # Large historical training outputs not used by the final course report
    "output\gaussian_object\pikaqiu",
    "output\gaussian_object\pikaqiu_video16_mixed",
    "output\gaussian_object\pikaqiu_video16_mixed_sharp",
    "output\gaussian_object\rabbit",
    "output\controlnet_finetune\pikaqiu",
    "output\controlnet_finetune\pikaqiu_video16_mixed",
    "output\controlnet_finetune\rabbit",

    # Old inspection / annotation scratch outputs
    "output\video_inspection",
    "output\tail_annotation",
    "output\right_ear_video_scan",
    "output\back_annotation",
    "output\body_annotation",
    "output\presentation_review",
    "output\ear_tip_annotation",
    "output\figures",
    "output\keyframe_selection_frames",
    "output\tail_enhance_pikaqiu78",

    # Old data variants not used by the final report; keep pikaqiu_video54_colmap_undistorted
    "data\realcap\pikaqiu_video",
    "data\realcap\ustc",
    "data\realcap\pikaqiu_video40_clean",
    "data\realcap\ustc_front48",
    "data\realcap\pikaqiu_video78_rightear_registered_undistorted",
    "data\realcap\pikaqiu_video_test",
    "data\realcap\pikaqiu_video30_mixed",
    "data\realcap\pikaqiu",
    "data\realcap\pikaqiu_video16_mixed",
    "data\realcap\pikaqiu_video54_colmap",
    "data\realcap\pikaqiu_video78_rightear_registered",
    "data\realcap\pikaqiu_video30_top6new",
    "data\realcap\ustc_colmap_undistorted",
    "data\realcap\ustc_turntable48_hq",
    "data\realcap\ustc_turntable48_calibrated",
    "data\realcap\pikaqiu_top_rotation14",
    "data\realcap\pikaqiu_video30_topnew",
    "data\realcap\pikaqiu_turntable",
    "data\realcap\ustc_turntable24_corrected",
    "data\realcap\ustc_turntable24",
    "data\realcap\pikaqiu_right_ear_ring48",
    "data\realcap\pikaqiu_right_ear_ring24_undistorted",
    "data\realcap\pikaqiu_right_ear_ring24_curated",
    "data\realcap\rabbit",
    "data\realcap\ustc_front24",
    "data\realcap\pikaqiu_top_candidates",
    "data\realcap\pikaqiu_side_top_candidates",
    "data\realcap\pikaqiu_side_top6_candidates",
    "data\realcap\pikaqiu_top6_candidates"
)

function Get-TargetSize([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    $Item = Get-Item -LiteralPath $Path
    if ($Item.PSIsContainer) {
        return (Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    }
    return $Item.Length
}

$ResolvedRoot = (Resolve-Path -LiteralPath $Root).Path
$Total = 0
foreach ($Target in $DeleteTargets) {
    $FullPath = Join-Path $Root $Target
    if (-not (Test-Path -LiteralPath $FullPath)) {
        continue
    }
    $Resolved = (Resolve-Path -LiteralPath $FullPath).Path
    if (-not $Resolved.StartsWith($ResolvedRoot)) {
        throw "Refusing to delete outside workspace: $Resolved"
    }
    $Size = Get-TargetSize $Resolved
    $Total += $Size
    "{0,8:N2} MB  {1}" -f ($Size / 1MB), $Target
}

"Estimated removable size: {0:N2} GB" -f ($Total / 1GB)

if (-not $Execute) {
    ""
    "Dry run only. Re-run with -Execute to delete these files."
    exit 0
}

foreach ($Target in $DeleteTargets) {
    $FullPath = Join-Path $Root $Target
    if (Test-Path -LiteralPath $FullPath) {
        $Resolved = (Resolve-Path -LiteralPath $FullPath).Path
        if (-not $Resolved.StartsWith($ResolvedRoot)) {
            throw "Refusing to delete outside workspace: $Resolved"
        }
        Remove-Item -LiteralPath $Resolved -Recurse -Force
        "Deleted $Target"
    }
}
