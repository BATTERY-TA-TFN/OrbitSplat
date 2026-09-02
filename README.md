# OrbitSplat: Real-World Object Reconstruction

> Reconstruct a real-world object from a handheld orbiting video through foreground extraction, camera pose estimation, point-cloud filtering, 3D Gaussian Splatting, and novel-view rendering.

**OrbitSplat** is a course project for reconstructing real-world objects from ordinary smartphone videos. Built on GaussianObject and 3D Gaussian Splatting, it addresses practical challenges such as blurry frames, redundant viewpoints, background contamination, and limited GPU memory. The repository provides an end-to-end workflow designed to run on **Windows with an NVIDIA RTX 3080**.

Rather than simply reproducing the original method, this project improves keyframe selection, foreground-mask generation, and COLMAP point-cloud filtering for real captured data. It also evaluates different initialization methods, the effects of view count and spatial coverage, and the engineering feasibility of Gaussian Repair.

## Demo Videos

The following videos show novel-view renderings produced by the reconstructed 3D Gaussian models. Click a preview link to play or download the corresponding MP4 file.

| Scene | Description | Video |
|---|---|---|
| USTC object | A camera trajectory rendered around the reconstructed object, demonstrating overall geometry, appearance consistency, and view interpolation. | [Watch `ustc.mp4`](assets/videos/ustc.mp4) |
| Pikachu | A novel-view rendering of the reconstructed Pikachu object, used to inspect foreground boundaries, texture preservation, and reconstruction stability around the object. | [Watch `pikaqiu.mp4`](assets/videos/pikaqiu.mp4) |

> GitHub may display MP4 files on a separate page instead of playing them directly inside the README. Both files are kept under `assets/videos/` so that they are included when the repository is cloned.

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

### 1. Video and Keyframe Preparation

The input is a handheld or turntable-style video that observes the target object from multiple directions. Frames are sampled from the video and filtered according to sharpness and visual redundancy. This step avoids spending training time on nearly identical images and reduces artifacts caused by motion blur.

For best results, the captured sequence should:

- keep the entire object visible throughout the video;
- provide a full horizontal orbit and several elevated or downward-looking views;
- use stable exposure and focus;
- avoid strong shadows, reflective surfaces, and moving backgrounds;
- preserve sufficient overlap between neighboring views.

### 2. Foreground Segmentation

Foreground masks are generated with color priors, GrabCut or Segment Anything (SAM), followed by morphological refinement. White pixels represent the object and black pixels represent the background. These masks are used during preprocessing, point-cloud filtering, and foreground-only evaluation.

### 3. Camera Pose and Geometry Initialization

Camera poses can be estimated with COLMAP, DUSt3R, or MASt3R. The resulting sparse geometry is projected into multiple object masks to reject points that are inconsistent with the foreground observations. A Visual Hull initialization is also included as an experimental alternative.

### 4. Gaussian Training and Repair

The filtered geometry initializes the 3D Gaussian representation. The project first trains a coarse Gaussian model, renders leave-one-out views, fine-tunes the repair model with LoRA, and finally refines the Gaussian representation using repaired supervision. The supplied RTX 3080 configuration reduces memory-intensive settings while retaining the complete workflow.

### 5. Rendering and Evaluation

The final model can be rendered along a custom camera path to produce images or videos such as the two demos above. Reconstruction quality is evaluated with PSNR, SSIM, and LPIPS. Foreground-only metrics are also available to reduce the influence of large, uniform background regions.

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

### Required Model Weights

Run the included download helper where applicable:

```powershell
Push-Location models
python download_hf_models.py
Pop-Location
```

Additional SAM, DUSt3R, and MASt3R weights should be placed in `models/` as described in [WINDOWS_3080_RUN.md](WINDOWS_3080_RUN.md). Model weights are intentionally excluded from Git because they are large and may have separate distribution terms.

### Expected Dataset Layout

A prepared object sequence should contain images, masks, and the camera or sparse-reconstruction files required by the selected pose-estimation route. A simplified layout is shown below:

```text
data/realcap/<object_name>/
├── images/
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── masks/
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── sparse/ or pose-estimation outputs
```

Image and mask filenames must correspond. Incorrect masks or inconsistent numbering can lead to poor point filtering, broken camera alignment, or evaluation errors.

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

The files serve different purposes:

- `refined_cams.json` stores the refined camera parameters used by later stages.
- `lora-step=1799.ckpt` is an intermediate LoRA checkpoint for the repair process.
- `last.ply` is the final reconstructed Gaussian model.
- `render/ours_None/` contains rendered frames that can be evaluated or assembled into an MP4 video.

Generated checkpoints, full render sequences, datasets, and temporary training files are not intended to be committed to Git. Only compact demonstration media and selected evaluation figures should be kept in `assets/`.

## Repository Structure

```text
OrbitSplat/
├── arguments/              # Command-line and configuration arguments
├── assets/                 # README figures and compact demo videos
├── configs/                # Training configurations, including RTX 3080 presets
├── gaussian_renderer/      # Gaussian rasterization and rendering interface
├── preprocess/             # Input preprocessing utilities
├── scene/                  # Cameras, datasets, and Gaussian model definitions
├── scripts/                # Setup, execution, evaluation, and report scripts
├── utils/                  # Shared utility functions
├── models/                 # Download scripts and local model weights
├── train_gs.py             # Initial Gaussian training
├── train_lora.py           # LoRA fine-tuning
├── train_repair.py         # Gaussian Repair stage
└── render.py               # Image and novel-view rendering
```

## Limitations

- Reconstruction quality depends strongly on mask accuracy and camera-pose quality.
- Thin structures, transparent objects, highly reflective materials, and textureless surfaces remain challenging.
- A purely horizontal capture may leave the top and bottom of the object underconstrained, even when many frames are used.
- Gaussian Repair requires additional training time and model weights; it may not improve every scene equally.
- The Windows path is optimized for an RTX 3080, but memory usage still varies with image resolution, point count, and densification settings.

## Intended Use

OrbitSplat can serve as a reference implementation for real-world object reconstruction, hands-on 3D Gaussian Splatting experiments, and computer-vision coursework. The repository retains the base code and dependency directories required by GaussianObject. For a course-only submission, the most relevant materials are located in `scripts/`, `output/evaluation/`, `output/course_report/`, and the final dataset used for the experiments.
