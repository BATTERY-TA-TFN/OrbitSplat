param(
    [string]$Python = "python",
    [string]$Gpu = "0",
    [string]$SceneName = "rabbit",
    [string]$DataDir = "data\realcap\rabbit",
    [int]$SparseNum = 4,
    [int]$PoseIterations = 300,
    [int]$Resolution = 8,
    [int]$LoraSteps = 1800,
    [double]$SilhouetteWeight = 0.01,
    [double]$PruneOpacityThreshold = 0.02,
    [string]$RepairTag = "",
    [ValidateSet("dust3r", "mast3r")]
    [string]$PoseEstimator = "dust3r",
    [switch]$MaskedPoseInput,
    [string]$PoseJson = "",
    [string]$InitPcdName = "",
    [switch]$LockPose,
    [string]$StartAt = "downsample",
    [string]$StopAfter = "render_final"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:CUDA_VISIBLE_DEVICES = $Gpu
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

$steps = @(
    "downsample",
    "pred_poses",
    "train_gs",
    "render_coarse",
    "loo_stage1",
    "loo_stage2",
    "train_lora",
    "repair",
    "render_final"
)

function Step-Index([string]$Name) {
    $idx = [array]::IndexOf($steps, $Name)
    if ($idx -lt 0) {
        throw "Unknown step '$Name'. Valid steps: $($steps -join ', ')"
    }
    return $idx
}

$startIndex = Step-Index $StartAt
$stopIndex = Step-Index $StopAfter
if ($startIndex -gt $stopIndex) {
    throw "StartAt must be before or equal to StopAfter."
}

function Should-Run([string]$Name) {
    $idx = Step-Index $Name
    return ($idx -ge $startIndex -and $idx -le $stopIndex)
}

function Run-Step([string]$Name, [scriptblock]$Body) {
    if (Should-Run $Name) {
        Write-Host ""
        Write-Host "===== $Name ====="
        & $Body
        if ($LASTEXITCODE -ne 0) {
            throw "Step '$Name' failed with exit code $LASTEXITCODE."
        }
    }
}

function Require-Path([string]$Path, [string]$Hint) {
    if (-not (Test-Path $Path)) {
        throw "Missing '$Path'. $Hint"
    }
}

function Resolve-ModelPath([string]$FileName) {
    $localPath = Join-Path "models" $FileName
    if (Test-Path $localPath) {
        return $localPath
    }

    if ($env:GAUSSIANOBJECT_MODEL_DIR) {
        $externalPath = Join-Path $env:GAUSSIANOBJECT_MODEL_DIR $FileName
        if (Test-Path $externalPath) {
            return $externalPath
        }
    }

    return $localPath
}

function Require-Model([string]$FileName, [string]$Hint) {
    $path = Resolve-ModelPath $FileName
    if (-not (Test-Path $path)) {
        throw "Missing '$FileName'. $Hint"
    }
    Write-Host "Model ready: $path"
}

Require-Path $DataDir "Check DataDir."
Require-Path (Join-Path $DataDir "images") "Put the four input images under images\."
Require-Path (Join-Path $DataDir "masks") "Generate masks first, for example with segment_anything.ipynb."
if ($PoseEstimator -eq "mast3r") {
    Require-Model "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" "Run models\download_preprocess_models_windows.ps1 -DownloadMast3r."
} else {
    Require-Model "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" "Download DUSt3R weights into models\ or GAUSSIANOBJECT_MODEL_DIR."
}
Require-Model "v1-5-pruned.ckpt" "Run: python models\download_hf_models.py, or put it in GAUSSIANOBJECT_MODEL_DIR."
Require-Model "control_v11f1e_sd15_tile.pth" "Run: python models\download_hf_models.py"

$GsDir = "output\gs_init\$SceneName"
$LooDir = "output\gs_init\$($SceneName)_loo"
$LoraExp = "controlnet_finetune/$SceneName"
$LoraDir = "output\controlnet_finetune\$SceneName"
$EffectiveRepairTag = if ($RepairTag) { $RepairTag } else { $SceneName }
$RepairPly = "output\gaussian_object\$EffectiveRepairTag\save\last.ply"
$LoraCheckpointName = "lora-step=$($LoraSteps - 1).ckpt"
$RefinedJson = Join-Path $GsDir "refined_cams.json"
if (-not $InitPcdName) {
    $InitPcdName = "dust3r_$SparseNum"
}

Run-Step "downsample" {
    & $Python preprocess\downsample.py -s $DataDir
}

Run-Step "pred_poses" {
    $poseArgs = @("-s", $DataDir, "--sparse_num", $SparseNum, "--niter", $PoseIterations)
    if ($MaskedPoseInput) {
        $poseArgs += "--masked_input"
    }
    if ($PoseEstimator -eq "mast3r") {
        & $Python pred_poses_mast3r.py @poseArgs
    } else {
        & $Python pred_poses.py @poseArgs
    }
}

Run-Step "train_gs" {
    $poseJsonArgs = @()
    if ($PoseJson) {
        $poseJsonArgs = @("--dust3r_json", $PoseJson)
    }
    $poseLockArgs = @()
    if ($LockPose) {
        $poseLockArgs = @("--pose_iterations", "0")
    }
    & $Python train_gs.py -s $DataDir `
        -m $GsDir `
        -r $Resolution --sparse_view_num $SparseNum --sh_degree 2 `
        --init_pcd_name $InitPcdName `
        @poseJsonArgs `
        @poseLockArgs `
        --white_background --random_background --use_dust3r `
        --max_num_splats 800000 `
        --densification_interval 200 `
        --densify_grad_threshold 0.0005 `
        --opacity_reset_interval 1000
}

Run-Step "render_coarse" {
    Require-Path $RefinedJson "Run train_gs first."
    & $Python render.py `
        -m $GsDir `
        --sparse_view_num $SparseNum --sh_degree 2 `
        --init_pcd_name $InitPcdName `
        --dust3r_json $RefinedJson `
        --white_background --render_path --use_dust3r `
        --not_generate_video
}

Run-Step "loo_stage1" {
    Require-Path $RefinedJson "Run train_gs first."
    & $Python leave_one_out_stage1.py -s $DataDir `
        -m $LooDir `
        -r $Resolution --sparse_view_num $SparseNum --sh_degree 2 `
        --init_pcd_name $InitPcdName `
        --dust3r_json $RefinedJson `
        --white_background --random_background --use_dust3r `
        --max_num_splats 800000 `
        --densification_interval 200 `
        --densify_grad_threshold 0.0005 `
        --opacity_reset_interval 1000
}

Run-Step "loo_stage2" {
    Require-Path $RefinedJson "Run train_gs first."
    & $Python leave_one_out_stage2.py -s $DataDir `
        -m $LooDir `
        -r $Resolution --sparse_view_num $SparseNum --sh_degree 2 `
        --init_pcd_name $InitPcdName `
        --dust3r_json $RefinedJson `
        --white_background --random_background --use_dust3r `
        --max_num_splats 800000 `
        --densification_interval 200 `
        --densify_grad_threshold 0.0005 `
        --opacity_reset_interval 1000
}

Run-Step "train_lora" {
    & $Python train_lora.py --exp_name $LoraExp `
        --prompt xxy5syt00 --sh_degree 2 --resolution $Resolution --sparse_num $SparseNum `
        --data_dir $DataDir `
        --gs_dir $GsDir `
        --loo_dir $LooDir `
        --bg_white --sd_locked --train_lora --use_prompt_list `
        --add_diffusion_lora --add_control_lora --add_clip_lora --use_dust3r `
        --batch_size 1 --image_size 512 --max_steps $LoraSteps `
        --callbacks_every_n_train_steps $LoraSteps
}

Run-Step "repair" {
    Require-Path (Join-Path $LoraDir "ckpts-lora\$LoraCheckpointName") "Run train_lora first with -LoraSteps $LoraSteps."
    & $Python train_repair.py `
        --config configs\gaussian-object-colmap-free-3080.yaml `
        --train --gpu $Gpu `
        tag="$EffectiveRepairTag" `
        system.init_dreamer="$GsDir" `
        system.exp_name="$LoraDir" `
        system.lora_name="$LoraCheckpointName" `
        system.refresh_size=6 `
        system.gaussian_opt_params.lambda_silhouette=$SilhouetteWeight `
        system.gaussian_opt_params.prune_opacity_threshold=$PruneOpacityThreshold `
        data.data_dir="$DataDir" `
        data.resolution=$Resolution `
        data.sparse_num=$SparseNum `
        data.prompt="a photo of a xxy5syt00" `
        data.json_path="$RefinedJson" `
        data.refresh_size=6 `
        system.sh_degree=2 `
        checkpoint.save_last=false `
        checkpoint.save_top_k=0
}

Run-Step "render_final" {
    Require-Path $RepairPly "Run repair first."
    & $Python render.py `
        -m $GsDir `
        --sparse_view_num $SparseNum --sh_degree 2 `
        --init_pcd_name $InitPcdName `
        --white_background --render_path --use_dust3r `
        --load_ply $RepairPly `
        --not_generate_video
}

Write-Host ""
Write-Host "Finished selected steps."
