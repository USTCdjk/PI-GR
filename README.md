# PI-GR: Microscopy Super-Resolution Reconstruction and Denoising

This project provides two independent Python inference scripts:

- **`super_resolution.py`** performs super-resolution (SR) reconstruction from wide-field (WF) observations and nine-frame structured illumination microscopy (SIM) data using diffusion sampling guided by physical forward models.
- **`denoise.py`** performs diffusion-based fluorescence microscopy denoising with an image formation model and a Poisson–Gaussian noise-consistency constraint.

> This README describes the supplied versions of the scripts. Neither script is fully standalone: compatible pretrained weights and project modules such as `models/` and `guided_diffusion/` are required. SR additionally requires `degraded.py`.

## 1. Project Structure

Place the scripts in the root of the corresponding project, alongside their dependencies:

```text
project/
├── README_EN.md
├── super_resolution.py
├── denoise.py
├── degraded.py                 # Defines SimpleStableSIM
├── models/
│   └── unet_2d_condition.py
├── guided_diffusion/
│   └── script_util_x0.py
├── ...                         # Other guided_diffusion modules
└── checkpoint-285000/
    └── unet_mix/               # Layout must match the source code
```

## 2. Requirements and Setup

A separate Python environment (approximately Python 3.10) and a compatible NVIDIA GPU are recommended. Install the Python packages directly imported by these scripts:

```bash
pip install numpy opencv-python tifffile Pillow transformers
```

Install an appropriate **PyTorch** build for your operating system and CUDA environment. Local imports (`models`, `guided_diffusion`, and `degraded`) are project-specific and are not provided by the command above.

Check the environment and available command-line options:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python super_resolution.py --help
python denoise.py --help
```

Both scripts require local CLIP tokenizer and text-encoder resources, in addition to compatible diffusion/UNet weights. The original absolute paths in the source files are machine-specific; set the appropriate CLI arguments or edit their `create_argparser()` defaults before running.

## 3. Super-Resolution Reconstruction (`super_resolution.py`)

### 3.1 Overview

The SR pipeline uses a WF image and nine raw SIM frames of the same sample as observations. A pretrained UNet and CLIP text encoder are used during diffusion sampling, while WF degradation consistency and a SIM forward model provide physics-based guidance. The SIM forward model is implemented by `degraded.SimpleStableSIM`.

The script also supports overlapping tiled inference, reflection padding, Hanning-window blending, stepwise evaluation, and Top-K result selection.

### 3.2 Input Data

WF image stems are used to match corresponding SIM observations and ground-truth (GT) images:

```text
dataset/
├── WF/
│   ├── 001.tif
│   └── 002.tif
├── SIM/
│   ├── 001.tif                 # Multi-page TIFF containing nine frames
│   └── 002.tif
└── GT/
    ├── 001.tif
    └── 002.tif
```

Alternatively, one sample can be represented by **exactly nine** `.tif`/`.tiff` images in a subfolder such as `SIM/001/`. The nine-frame stack must be readable in a compatible arrangement such as `[9, H, W]`. WF and GT images should be single-channel 2D images.

**Important:** In the supplied script, a matching GT image is required for a sample to be processed. Samples with missing SIM data or GT are skipped; the script fails if no valid matches are found.

### 3.3 Example Command

```bash
python super_resolution.py \
  --wf_root /path/to/dataset/WF \
  --sim_root /path/to/dataset/SIM \
  --gt_root /path/to/dataset/GT \
  --unet_model_path /path/to/checkpoint-285000 \
  --text_encoder_path /path/to/text_encoder \
  --tokenizer_path /path/to/tokenizer \
  --save_dir ./output_sr \
  --num_cases 10 \
  --device 0
```

The UNet checkpoint path must provide the expected `unet_mix` subdirectory. `--device 0` denotes the configured GPU identifier.

### 3.4 Key Default Parameters

| Argument | Default in supplied script | Description |
| --- | --- | --- |
| `wf_root` / `sim_root` / `gt_root` | `/data3/ddt/BIOSR-SIM/数据集/1014/1`, `/2`, `/3` | Input dataset directories (machine-specific) |
| `save_dir` | `./output_1014` | Output directory |
| `prompt` | `Microtubules of COS-7` | Text conditioning prompt |
| `denoise_steps` | `1000` | Diffusion sampling-step setting |
| `tile_size_lr` / `tile_stride_lr` | `128` / `96` | Low-resolution tile size / stride |
| `hr_size` | `256` | High-resolution tile setting |
| `num_cases` | `100` | Maximum number of matched cases |
| `wf_guidance_weight` / `sim_guidance_weight` | `1.0` / `1.0` | Relative weights of the two guidance gradients |
| `img_guidance_scale` / `sim_img_guidance_scale` | `120.0` / `120.0` | WF and SIM guidance strengths |
| `topk` / `topk_metric` | `20` / `best_full_psnr` | Best-case selection configuration |

`main()` explicitly sets `args.diffusion_steps = 3000`; therefore, values overridden inside `main()` take precedence over parser defaults. The WF PSF, gain, noise, and SIM model parameters are experiment-dependent and should be checked before modification. Some settings in the `SimpleStableSIM` initialization, including `image_size=256`, are fixed in the supplied code rather than taken entirely from the `sim_*` CLI arguments.

### 3.5 Main Outputs

The script produces per-case directories and dataset summaries. Representative output layout:

```text
output_sr/
├── 001/
│   ├── result_eval_final.png
│   ├── result_eval_final_center_200x200.png
│   ├── result_sample.png
│   ├── gt.png
│   ├── wf_obs_raw.png
│   ├── sim_mean.png
│   ├── final_metrics.json
│   ├── per_step_metrics.json
│   └── best_step_results/
│       ├── best_full_psnr.png
│       └── ...
├── dataset_summary.json
├── dataset_summary.csv
└── top20_best_full_psnr/       # Name/content depend on the ranking settings
```

Reference-based SR metrics include **MSE, PSNR, and SSIM**, evaluated on both the full image and the central `200 × 200` crop. Additional diagnostic images, heatmaps, and logs depend on enabled options and execution paths. Some helper functions save normalized results as **8-bit PNGs**; these should not be interpreted as preserved 16-bit fluorescence intensities.

## 4. Fluorescence Denoising (`denoise.py`)

### 4.1 Overview

The denoising pipeline supports optional preprocessing, physics-guided diffusion sampling, overlapping tile blending, and image/metric export. Guidance is based on a fluorescence image formation model (PSF blur, camera gain and background) and approximate Poisson–Gaussian observation likelihood. The source also contains optional random masking and TV/background/stripe-related regularization.

Compatible **pretrained model weights are required**; this inference script does not train a model from scratch.

### 4.2 Input Data

```text
dataset/
├── noisy/
│   ├── 001.tif
│   └── 002.png
└── GT/                        # Optional, used for reference-based metrics
    ├── 001.tif
    └── 002.png
```

Supported image extensions include `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, and `.bmp`. A GT image is matched by filename stem, so extensions may differ. With `require_gt=False`, samples without GT may still be denoised, but reference-dependent quality metrics cannot be computed.

Processing primarily operates on grayscale 2D images. Some TIFF stack layouts receive specific handling; inspect input array shapes before processing multi-frame data.

### 4.3 Example Command

```bash
python denoise.py \
  --noisy_root /path/to/dataset/noisy \
  --gt_root /path/to/dataset/GT \
  --unet_model_path /path/to/checkpoint-285000 \
  --text_encoder_path /path/to/text_encoder \
  --tokenizer_path /path/to/tokenizer \
  --save_dir ./output_denoise \
  --num_cases 10 \
  --device 0
```

For inference without GT, an empty `--gt_root` can be used together with `require_gt=False`. Check `python denoise.py --help` for the accepted syntax of boolean arguments in your code version.

### 4.4 Key Default Parameters

| Argument | Default in supplied script | Description |
| --- | --- | --- |
| `noisy_root` | `/data3/ddt/BIOSR-SIM/数据集/1014/1` | Noisy input directory |
| `gt_root` | `/data3/ddt/BIOSR-SIM/数据集/1014/3` | Optional reference directory |
| `require_gt` | `False` | Whether GT is mandatory |
| `save_dir` | `./output_10141014` | Output directory |
| `prompt` | `mitochondria of BPAE` | Text conditioning prompt |
| `denoise_steps` | `1000` | Number of denoising steps |
| `tile_size` / `tile_stride` | `256` / `192` | Tile size / stride |
| `init_mode` | `noisy_plus_noise` | Diffusion initialization method |
| `init_noise_sigma` | `0.15` | Additional initial noise strength |
| `output_source` | `pred_xstart` | Output selection; `sample` is another option |
| `img_guidance_scale` | `89.96860923080581` | Physics-guidance strength |
| `use_self2self_mask` | `True` | Enable random observation masks |
| `self2self_mask_ratio` | `0.4` | Mask ratio |
| `self2self_mc_samples` | `4` | Number of Monte Carlo mask samples |
| `use_preprocess` | `False` | Enable conventional preprocessing |
| `topk` / `topk_metric` | `20` / `psnr` | Best-case ranking; reference-based ranking needs GT |

Additional physics-related arguments include `fluor_psf_sigma`, `fluor_gain`, `fluor_photon_scale`, and `fluor_read_noise_sigma`. The program synchronizes `image_size` with `tile_size` in `main()` and requires `0 < tile_stride <= tile_size` and `denoise_steps <= diffusion_steps`.

### 4.5 Main Outputs

```text
output_denoise/
├── 001/
│   ├── noisy_raw.png
│   ├── noisy.png
│   ├── noisy.tif
│   ├── denoised_final.png
│   ├── denoised_final.tif
│   ├── denoised_padded.png
│   ├── weight_map.png
│   ├── gt.png                   # When GT is available
│   └── metrics.json            # When GT is available
├── dataset_summary.json
├── dataset_summary.csv
└── metrics_all_cases.csv
```

Optional tile images and Top-K selections may also be saved. When GT is available, metrics include **MSE, PSNR, SSIM, NRMSE, HFEN, and GMS**; without GT, relevant entries may be empty. In the supplied version, `denoised_final.tif` is a **single-channel 8-bit TIFF**, not a preserved 16-bit intensity image.

## 5. Script Comparison

| Aspect | `super_resolution.py` | `denoise.py` |
| --- | --- | --- |
| Goal | Super-resolution reconstruction | Fluorescence image denoising |
| Primary inputs | WF + 9-frame SIM + matching GT | Noisy image; GT optional |
| Physics guidance | WF degradation + SIM forward consistency | Fluorescence PSF/camera response + Poisson–Gaussian noise consistency |
| Tiling | Overlapping low-resolution tiles and weighted blending | Overlapping image tiles and weighted blending |
| Metrics | MSE, PSNR, SSIM | MSE, PSNR, SSIM, NRMSE, HFEN, GMS |
| Main image | `result_eval_final.png` | `denoised_final.png` / `denoised_final.tif` |

## 6. Troubleshooting and Important Notes

1. **Missing models or imports:** Run from the correct project root and verify `models`, `guided_diffusion`, pretrained weights, and `degraded.py` (SR). The SR checkpoint is expected to contain `unet_mix`.
2. **No valid SR cases found:** Verify matching WF/SIM/GT filename stems and the nine-frame SIM organization.
3. **CUDA out of memory:** Consider smaller tiles or sampling settings, but ensure the UNet input and fixed `SimpleStableSIM` dimensions remain compatible.
4. **Slow inference:** Long diffusion trajectories, physical guidance and overlapping tiles are computationally expensive. Start with `--num_cases 1`.
5. **Apparent loss of resolution:** Examine fine structures, intensity profiles and reference-based metrics; use an independent resolution estimate when possible. A visually smoother result alone does not demonstrate improved spatial resolution.
6. **Quantization and fluorescence intensity:** Some exported images are 8-bit. For quantitative intensity analysis, implement an explicitly documented high-precision float or 16-bit export with an appropriate normalization policy.
7. **GT-dependent model selection:** SR best-step selection uses GT-based evaluation and should be interpreted as offline reference-based assessment, not a selection procedure available for GT-free deployment.
8. **Repeated experiments:** Use a distinct `save_dir` for each configuration to avoid overwriting earlier results.

## 7. Recommended Validation Workflow

1. Prepare one test sample and verify the input files and dimensions.
2. Confirm CUDA availability, checkpoints, CLIP resources and local project modules.
3. Inspect `--help` for both programs.
4. Run each script with `--num_cases 1` and a new `save_dir`.
5. Check the output image dimensions, channel count, bit depth, logs and metrics.
6. Scale up to the full dataset only after the single-case test succeeds.

---

**Scope:** This README is based on the supplied source code. It does not certify successful execution on any particular machine. Validate dependency versions, checkpoints, paths and hardware before running experiments.
