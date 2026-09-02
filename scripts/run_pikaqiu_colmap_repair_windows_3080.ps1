param(
    [string]$Python = "D:\Anaconda\envs\gaussianobject-py311\python.exe",
    [string]$Gpu = "0",
    [string]$DataDir = "data\realcap\pikaqiu_video54_colmap_undistorted",
    [string]$GsDir = "output\gs_init\pikaqiu_video54_colmap_hq20k",
    [string]$Tag = "pikaqiu_video54_colmap_repair_validation",
    [int]$SparseNum = 8,
    [int]$LooCheckpointIteration = 1600,
    [int]$LooIterations = 1800,
    [int]$LoraSteps = 32,
    [int]$RepairSteps = 50,
    [int]$RepairRefreshSize = 1,
    [int]$Resolution = 8,
    [string]$StartAt = "prepare",
    [string]$StopAfter = "render_final"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:CUDA_VISIBLE_DEVICES = $Gpu
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
if (-not $env:GAUSSIANOBJECT_MODEL_DIR) {
    $env:GAUSSIANOBJECT_MODEL_DIR = "D:\GaussianObject-models"
}

$Steps = @("prepare", "loo_stage1", "loo_stage2", "train_lora", "repair", "render_final")

function Step-Index([string]$Name) {
    $Index = [array]::IndexOf($Steps, $Name)
    if ($Index -lt 0) {
        throw "Unknown step '$Name'. Valid steps: $($Steps -join ', ')"
    }
    return $Index
}

$StartIndex = Step-Index $StartAt
$StopIndex = Step-Index $StopAfter
if ($StartIndex -gt $StopIndex) {
    throw "StartAt must be before or equal to StopAfter."
}

function Run-Step([string]$Name, [scriptblock]$Body) {
    $Index = Step-Index $Name
    if ($Index -ge $StartIndex -and $Index -le $StopIndex) {
        Write-Host ""
        Write-Host "===== $Name ====="
        $global:LASTEXITCODE = 0
        & $Body
        if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            throw "Step '$Name' failed with exit code $LASTEXITCODE."
        }
    }
}

function Require-Path([string]$Path, [string]$Hint) {
    if (-not (Test-Path $Path)) {
        throw "Missing '$Path'. $Hint"
    }
}

$LooDir = "output\gs_init\$($Tag)_loo"
$LoraExp = "controlnet_finetune/$Tag"
$LoraDir = "output\controlnet_finetune\$Tag"
$LoraCheckpointName = "lora-step=$($LoraSteps - 1).ckpt"
$RepairPly = "output\gaussian_object\$Tag\save\last.ply"

Run-Step "prepare" {
    Require-Path $DataDir "COLMAP dataset is required."
    Require-Path (Join-Path $DataDir "sparse\0\images.bin") "Run COLMAP first."
    Require-Path (Join-Path $DataDir "colmap_object.ply") "Run point-cloud mask filtering first."
    Require-Path (Join-Path $GsDir "point_cloud\iteration_20000\point_cloud.ply") "Run HQ20k training first."

    # Zero-based camera indices: six around-object views plus two top-ring views.
    $SparseIds = @(0, 7, 14, 21, 28, 35, 44, 51)
    if ($SparseIds.Count -ne $SparseNum) {
        throw "This validation preset expects SparseNum=$($SparseIds.Count)."
    }
    $SparseIds | Set-Content (Join-Path $DataDir "sparse_$SparseNum.txt")
    @(1..53 | Where-Object { $_ -notin $SparseIds }) | Set-Content (Join-Path $DataDir "sparse_test.txt")
    Write-Host "Prepared sparse views: $($SparseIds -join ', ')"
}

Run-Step "loo_stage1" {
    & $Python leave_one_out_stage1.py -s $DataDir -m $LooDir `
        -r $Resolution --sparse_view_num $SparseNum --sh_degree 2 `
        --init_pcd_name colmap_object --white_background --random_background `
        --iterations $LooIterations --position_lr_max_steps $LooIterations `
        --densify_until_iter $LooCheckpointIteration `
        --checkpoint_iterations $LooCheckpointIteration `
        --max_num_splats 200000 --densification_interval 200 `
        --densify_grad_threshold 0.0005 --opacity_reset_interval 800
}

Run-Step "loo_stage2" {
    & $Python leave_one_out_stage2.py -s $DataDir -m $LooDir `
        -r $Resolution --sparse_view_num $SparseNum --sh_degree 2 `
        --init_pcd_name colmap_object --white_background --random_background `
        --iterations $LooIterations --position_lr_max_steps $LooIterations `
        --loo_checkpoint_iteration $LooCheckpointIteration `
        --densify_until_iter $LooCheckpointIteration `
        --max_num_splats 200000 --densification_interval 200 `
        --densify_grad_threshold 0.0005 --opacity_reset_interval 800
}

Run-Step "train_lora" {
    & $Python train_lora.py --exp_name $LoraExp `
        --prompt xxy5syt00 --sh_degree 2 --resolution $Resolution --sparse_num $SparseNum `
        --data_dir $DataDir --gs_dir $GsDir --loo_dir $LooDir `
        --bg_white --sd_locked --train_lora --use_prompt_list `
        --add_diffusion_lora --add_control_lora --add_clip_lora `
        --batch_size 1 --image_size 512 --precision 16-mixed --disable_image_logger --max_steps $LoraSteps `
        --callbacks_every_n_train_steps $LoraSteps
}

Run-Step "repair" {
    Require-Path (Join-Path $LoraDir "ckpts-lora\$LoraCheckpointName") "Run train_lora first."
    & $Python train_repair.py `
        --config configs\gaussian-object-colmap-3080-validation.yaml `
        --train --gpu $Gpu `
        tag="$Tag" `
        system.init_dreamer="$GsDir" `
        system.exp_name="$LoraDir" `
        system.lora_name="$LoraCheckpointName" `
        system.refresh_size=$RepairRefreshSize `
        system.around_gt_steps=$RepairSteps `
        system.ctrl_steps=$RepairSteps `
        system.gaussian_opt_params.iterations=$RepairSteps `
        system.gaussian_opt_params.position_lr_max_steps=$RepairSteps `
        system.gaussian_opt_params.densify_until_iter=0 `
        data.data_dir="$DataDir" `
        data.resolution=$Resolution `
        data.sparse_num=$SparseNum `
        data.prompt="a photo of a xxy5syt00" `
        data.length=$RepairSteps `
        data.around_gt_steps=$RepairSteps `
        data.refresh_size=$RepairRefreshSize `
        trainer.max_steps=$RepairSteps `
        checkpoint.every_n_train_steps=$RepairSteps `
        checkpoint.save_last=false `
        checkpoint.save_top_k=0
}

Run-Step "render_final" {
    Require-Path $RepairPly "Run repair first."
    & $Python render.py -m $GsDir --sh_degree 2 `
        --white_background --render_path --load_ply $RepairPly --not_generate_video
}

Write-Host ""
Write-Host "Finished selected COLMAP repair steps."
