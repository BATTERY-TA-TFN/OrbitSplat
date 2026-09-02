# OrbitSplat: Real-World Object Reconstruction

> Reconstruct a real-world object from a handheld orbiting video through foreground extraction, camera pose estimation, point-cloud filtering, 3D Gaussian Splatting, and novel-view rendering.

**OrbitSplat** is a course project for reconstructing real-world objects from ordinary smartphone videos. Built on GaussianObject and 3D Gaussian Splatting, it addresses practical challenges such as blurry frames, redundant viewpoints, background contamination, and limited GPU memory. The repository provides an end-to-end workflow designed to run on **Windows with an NVIDIA RTX 3080**.

Rather than simply reproducing the original method, this project improves keyframe selection, foreground-mask generation, and COLMAP point-cloud filtering for real captured data. It also evaluates different initialization methods, the effects of view count and spatial coverage, and the engineering feasibility of Gaussian Repair.

## Highlights

- **Accessible data capture:** uses an orbiting object video recorded with an ordinary smartphone.
- **Automated preprocessing:** extracts sharp and representative keyframes while reducing blur and redundancy.
- **Foreground-aware reconstruction:** generates object masks using color priors, GrabCut/SAM, and morphological processing.
- **Point-cloud filtering:** applies multi-view mask constraints to remove background points from a sparse COLMAP reconstruction.
- **End-to-end evaluation:** covers 3DGS training, Gaussian Repair, novel-view rendering, and quantitative evaluation.
- **Consumer GPU support:** includes configurations and scripts tailored for Windows and the RTX 3080.

## Pipeline

```text
Handheld Orbiting Video
    ↓
Keyframe Extraction and Sharpness Filtering
    ↓
Foreground Segmentation and Mask Refinement
    ↓
Camera Pose Estimation (COLMAP / DUSt3R / MASt3R)
    ↓
Multi-View Mask-Constrained Point-Cloud Filtering
    ↓
3D Gaussian Splatting Initialization and Training
    ↓
Gaussian Repair
    ↓
Novel-View Rendering and PSNR / SSIM / LPIPS Evaluation
```

## Environment

| Component | Recommended Configuration |
|---|---|
| Operating system | Windows |
| GPU | NVIDIA RTX 3080 |
| Python | 3.11 |
| CUDA | 11.8 |
| PyTorch | 2.1.2 |

See [WINDOWS_3080_RUN.md](WINDOWS_3080_RUN.md) for complete environment setup, model-weight preparation, and step-by-step execution instructions.

## Quick Start

Open an Anaconda PowerShell Prompt in the repository root and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows_3080.ps1
conda activate gaussianobject-py311
```

After preparing the model weights and object masks, run the included `rabbit` example:

```powershell
.\scripts\run_rabbit_windows_3080.ps1
```

The pipeline can also resume from a specific stage:

```powershell
.\scripts\run_rabbit_windows_3080.ps1 -StartAt train_lora
```

Available stages:

```text
downsample, pred_poses, train_gs, render_coarse,
loo_stage1, loo_stage2, train_lora, repair, render_final
```

## Experimental Results

### Initialization Ablation

| Initialization Method | Initial Points | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| Original COLMAP | 5,727 | 26.596 | 0.96273 | 0.04459 |
| Mask-Filtered COLMAP | 285 | 26.519 | 0.96276 | 0.04515 |
| Visual Hull | 7,787 | 24.713 | 0.95745 | 0.05733 |

Multi-view mask filtering reduces the initial point count from 5,727 to 285, removing approximately **95.02%** of background points. The PSNR decreases by only 0.077 dB, indicating that the method substantially suppresses background contamination while preserving reconstruction quality.

### View-Coverage Experiment

| Training Views | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| 8 views, including several elevated views | 17.958 | 0.92673 | 0.10038 |
| 40 views, primarily captured along a horizontal orbit | 17.414 | 0.93085 | 0.09961 |

The results show that the number of training images is not the only determining factor. Spatial viewpoint coverage also has a significant effect on rendering quality at unseen views.

## Key Files

| Path | Purpose |
|---|---|
| `train_gs.py` | Initial 3D Gaussian Splatting training |
| `train_lora.py` | LoRA fine-tuning |
| `train_repair.py` | Gaussian Repair training |
| `pred_poses.py` / `pred_poses_mast3r.py` | Camera pose estimation |
| `visual_hull.py` | Visual Hull initialization experiment |
| `render.py` | Novel-view rendering |
| `scripts/run_rabbit_windows_3080.ps1` | RTX 3080 example pipeline |
| `scripts/evaluate_foreground_metrics.py` | Foreground-region metric evaluation |
| `scripts/summarize_evaluation.py` | Experimental result aggregation |
| `output/evaluation/` | Metrics, plots, and qualitative comparisons |
| `output/course_report/` | Generated course reports |

## Example Outputs

The complete pipeline produces the following primary outputs:

```text
output/gs_init/rabbit/refined_cams.json
output/controlnet_finetune/rabbit/ckpts-lora/lora-step=1799.ckpt
output/gaussian_object/rabbit/save/last.ply
output/gs_init/rabbit/render/ours_None/
```

## Intended Use

OrbitSplat can serve as a reference implementation for real-world object reconstruction, hands-on 3D Gaussian Splatting experiments, and computer-vision coursework. The repository retains the base code and dependency directories required by GaussianObject. For a course-only submission, the most relevant materials are located in `scripts/`, `output/evaluation/`, `output/course_report/`, and the final dataset used for the experiments.
