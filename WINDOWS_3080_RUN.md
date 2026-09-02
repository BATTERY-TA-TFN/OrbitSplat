# Windows + RTX 3080 reproduction notes

This repo can be run on Windows, but Linux/WSL2 is still easier for CUDA/C++ extensions.
The scripts here keep the original code intact and add a conservative RTX 3080 path.

## 1. Create the environment

Open "Anaconda PowerShell Prompt" in the repo root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows_3080.ps1
conda activate gaussianobject-py311
```

If conda times out or reports a broken cache, retry with:

```powershell
.\scripts\setup_windows_3080.ps1 -CleanCondaCache -UseTunaMirror
conda activate gaussianobject-py311
```

The setup script uses a local package cache at `.conda-pkgs` to avoid broken or locked files under `D:\Anaconda\pkgs`.
On Windows it installs `requirements-windows-pypi.txt` first, then installs local CUDA/source packages from `requirements-windows-local.txt` with `--no-build-isolation` so the build can see the already installed PyTorch.
The Windows setup pins PyTorch to `torch==2.1.2+cu118`, `torchvision==0.16.2+cu118`, and `numpy==1.26.4`; this is more compatible with the 3DGS CUDA extensions than newer PyTorch 2.7 wheels.
PyTorch3D is skipped on Windows by default. The project only uses it for KNN outlier pruning, and `scene/gaussian_model.py` has a chunked fallback for the Windows reproduction path.

If the repo was downloaded as a zip, fill the empty submodule folders first:

```powershell
.\scripts\sync_submodules_windows.ps1
```

If GitHub is slow:

```powershell
.\scripts\sync_submodules_windows.ps1 -UseGhProxy
```

If a previous clone failed halfway:

```powershell
.\scripts\sync_submodules_windows.ps1 -UseGhProxy -RepairIncomplete
```

If CroCo CUDA ops are needed:

```powershell
.\scripts\setup_windows_3080.ps1 -BuildCroco
```

## 2. Download required weights

```powershell
Push-Location models
python download_hf_models.py
Pop-Location
```

Also place these COLMAP-free weights in `models\`:

- `sam_vit_h_4b8939.pth`
- `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`
- `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` if using MASt3R

## 3. Prepare `rabbit`

The bundled rabbit sample has images only. Generate masks first:

```text
data\realcap\rabbit\masks\001.png
data\realcap\rabbit\masks\002.png
data\realcap\rabbit\masks\003.png
data\realcap\rabbit\masks\004.png
```

Use `segment_anything.ipynb` or any mask tool. White/object = foreground, black = background.

## 4. Run the RTX 3080 pipeline

```powershell
.\scripts\run_rabbit_windows_3080.ps1
```

Resume from a step:

```powershell
.\scripts\run_rabbit_windows_3080.ps1 -StartAt train_lora
```

Run only one step:

```powershell
.\scripts\run_rabbit_windows_3080.ps1 -StartAt pred_poses -StopAfter pred_poses
```

Valid steps:

```text
downsample, pred_poses, train_gs, render_coarse, loo_stage1, loo_stage2, train_lora, repair, render_final
```

## 5. RTX 3080 choices

The 3080 config uses:

- `resolution=8`
- `batch_size=1`
- `max_num_splats=800000`
- slower densification interval
- final repair steps reduced from 4000 to 3000

If you have a 12 GB RTX 3080 and memory is stable, you can raise `max_num_splats` to `1200000`.
If you have a 10 GB RTX 3080 and hit OOM, lower it to `500000` in:

- `configs\gaussian-object-colmap-free-3080.yaml`
- `scripts\run_rabbit_windows_3080.ps1`

## 6. Expected outputs

```text
output\gs_init\rabbit\refined_cams.json
output\controlnet_finetune\rabbit\ckpts-lora\lora-step=1799.ckpt
output\gaussian_object\rabbit\save\last.ply
output\gs_init\rabbit\render\ours_None\
```
