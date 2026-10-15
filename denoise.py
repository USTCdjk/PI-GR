import argparse
import os
import re
import csv
import json
import time
import glob
import shutil
from typing import Optional, Tuple, List

import cv2
import numpy as np
import tifffile as tiff
from PIL import Image

import torch
import torch as th
import torch.nn.functional as F

from transformers import CLIPTokenizer, CLIPTextModel
from models.unet_2d_condition import UNet2DConditionModel

from guided_diffusion import logger
from guided_diffusion.script_util_x0 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


# =========================================================
# 1. 基础工具函数
# =========================================================
def natural_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def list_image_files(folder):
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
    files = []
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if os.path.isfile(p) and f.lower().endswith(exts):
            files.append(p)
    files.sort(key=lambda x: natural_key(os.path.basename(x)))
    return files


def normalize_to_float01(img, percentile_norm=False, p_low=0.1, p_high=99.9):
    img = np.asarray(img).astype(np.float32)

    if percentile_norm:
        lo, hi = np.percentile(img, [p_low, p_high])
        img = np.clip(img, lo, hi)
        return (img - lo) / (hi - lo + 1e-8)

    mn = float(img.min())
    mx = float(img.max())
    return (img - mn) / (mx - mn + 1e-8)


def read_gray_image_float01(path, percentile_norm=False):
    try:
        img = tiff.imread(path)
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    img = np.asarray(img)
    img = np.squeeze(img)

    if img.ndim == 2:
        return normalize_to_float01(img, percentile_norm=percentile_norm)

    if img.ndim == 3:
        # RGB / BGR 图像转灰度
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
            return normalize_to_float01(img, percentile_norm=percentile_norm)

        # [C,H,W] 且 C=1
        if img.shape[0] == 1:
            return normalize_to_float01(img[0], percentile_norm=percentile_norm)

        # [H,W,1]
        if img.shape[-1] == 1:
            return normalize_to_float01(img[..., 0], percentile_norm=percentile_norm)

        # 多帧 tiff 默认取均值
        if img.shape[0] < 10:
            img = img.astype(np.float32).mean(axis=0)
            return normalize_to_float01(img, percentile_norm=percentile_norm)

    raise ValueError(f"Unsupported image shape: {img.shape}, path={path}")


def save_float01_image(x, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    x_u8 = (x * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(x_u8, mode="L").save(path)


def save_float01_tif_8bit(x, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    x_u8 = (x * 255.0 + 0.5).astype(np.uint8)
    tiff.imwrite(path, x_u8)


def save_heatmap(pred, gt, save_path):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

    diff = np.abs(pred - gt)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    diff_8u = (diff * 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff_8u, cv2.COLORMAP_JET)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, heat)


def resolve_same_name(root, stem):
    if root is None or root == "":
        return None

    candidates = []
    for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"]:
        candidates.append(os.path.join(root, stem + ext))

    for p in candidates:
        if os.path.isfile(p):
            return p

    recursive = []
    for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"]:
        recursive.extend(glob.glob(os.path.join(root, "**", stem + ext), recursive=True))

    recursive = sorted(recursive, key=natural_key)
    if len(recursive) > 0:
        return recursive[0]

    return None


def safe_copy(src, dst):
    if src is None:
        return
    if not os.path.isfile(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


# =========================================================
# 2. 分块与融合
# =========================================================
def get_tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_hanning_weight(h, w, eps=1e-3):
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    weight = np.outer(wy, wx).astype(np.float32)
    weight = np.maximum(weight, eps)
    return weight


def pad_2d_reflect_to_min_size_and_multiple(img, tile_size):
    """
    保证图像尺寸 >= tile_size，并且 pad 到 tile_size 的整数倍附近。
    """
    h, w = img.shape

    target_h = max(tile_size, int(np.ceil(h / tile_size) * tile_size))
    target_w = max(tile_size, int(np.ceil(w / tile_size) * tile_size))

    pad_h = target_h - h
    pad_w = target_w - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = np.pad(img, ((top, bottom), (left, right)), mode="reflect")
    return padded.astype(np.float32), (top, bottom, left, right)


def crop_by_pad_info(img, pad_info):
    top, bottom, left, right = pad_info
    h_end = img.shape[0] - bottom if bottom > 0 else img.shape[0]
    w_end = img.shape[1] - right if right > 0 else img.shape[1]
    return img[top:h_end, left:w_end]


# =========================================================
# 3. 指标
# =========================================================
def compute_psnr(pred, target):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mse = np.mean((pred - target) ** 2)
    psnr = -10.0 * np.log10(max(mse, 1e-12))
    return float(psnr), float(mse)


def compute_ssim_np(pred, target, C1=0.01 ** 2, C2=0.03 ** 2):
    pred_t = torch.from_numpy(pred.astype(np.float32))[None, None]
    target_t = torch.from_numpy(target.astype(np.float32))[None, None]

    mu_x = F.avg_pool2d(pred_t, 3, 1, 1)
    mu_y = F.avg_pool2d(target_t, 3, 1, 1)

    sigma_x = F.avg_pool2d(pred_t * pred_t, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target_t * target_t, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred_t * target_t, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-12
    )
    return float(ssim_map.mean().item())


def compute_nrmse(pred, target):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    denom = target.max() - target.min() + 1e-8
    return float(rmse / denom)


def compute_hfen(pred, target, sigma=1.5):
    """
    HFEN: High-Frequency Error Norm.
    越低越好。
    这里使用 LoG 近似：先 GaussianBlur，再 Laplacian，避免额外依赖 scipy。
    输入默认是 [0,1] float 图像。
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    pred_blur = cv2.GaussianBlur(pred, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    target_blur = cv2.GaussianBlur(target, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))

    pred_log = cv2.Laplacian(pred_blur, cv2.CV_32F, ksize=3)
    target_log = cv2.Laplacian(target_blur, cv2.CV_32F, ksize=3)

    numerator = np.linalg.norm(pred_log - target_log)
    denominator = np.linalg.norm(target_log) + 1e-8

    return float(numerator / denominator)


def compute_gms(pred, target):
    """
    GMS: Gradient Magnitude Similarity.
    越高越好。
    衡量输出图像与 GT 在边缘/梯度结构上的一致性。
    输入默认是 [0,1] float 图像。
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    pred_gx = cv2.Sobel(pred, cv2.CV_32F, 1, 0, ksize=3)
    pred_gy = cv2.Sobel(pred, cv2.CV_32F, 0, 1, ksize=3)
    target_gx = cv2.Sobel(target, cv2.CV_32F, 1, 0, ksize=3)
    target_gy = cv2.Sobel(target, cv2.CV_32F, 0, 1, ksize=3)

    pred_grad = np.sqrt(pred_gx ** 2 + pred_gy ** 2)
    target_grad = np.sqrt(target_gx ** 2 + target_gy ** 2)

    # 图像范围为 [0,1] 时，0.0026 是常用稳定项量级。
    c = 0.0026
    gms_map = (2.0 * pred_grad * target_grad + c) / (
        pred_grad ** 2 + target_grad ** 2 + c
    )

    return float(np.mean(gms_map))


def compute_metrics(pred, target):
    if pred.shape != target.shape:
        pred = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)

    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)

    psnr, mse = compute_psnr(pred, target)
    ssim = compute_ssim_np(pred, target)
    nrmse = compute_nrmse(pred, target)
    hfen = compute_hfen(pred, target)
    gms = compute_gms(pred, target)

    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "nrmse": float(nrmse),
        "hfen": float(hfen),
        "gms": float(gms),
    }


# =========================================================
# 4. Text condition
# =========================================================
def load_text_model(args, device):
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path)
    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder_path).to(device)
    text_encoder.eval()

    for p in text_encoder.parameters():
        p.requires_grad = False

    return tokenizer, text_encoder


@torch.no_grad()
def build_text_condition(tokenizer, text_encoder, prompt, device):
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_input.input_ids.to(device)
    encoder_hidden_states = text_encoder(input_ids=input_ids)[0]
    return encoder_hidden_states


# =========================================================
# 5. 可微滤波 / 损失
# =========================================================
def gaussian_kernel2d(kernel_size, sigma, device, dtype, channels=1):
    if kernel_size <= 1 or sigma <= 0:
        return None

    if kernel_size % 2 == 0:
        kernel_size += 1

    ax = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / (kernel.sum() + 1e-12)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


def gaussian_blur_torch(x, kernel_size=5, sigma=1.0):
    if sigma <= 0 or kernel_size <= 1:
        return x

    c = x.shape[1]
    kernel = gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype, channels=c)
    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=c)


def sobel_edges_torch(x):
    c = x.shape[1]
    kx = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)

    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)

    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def ssim_loss_torch(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-12
    )

    return 1.0 - ssim_map.mean()


def tv_loss_torch(x):
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w


def column_stripe_loss_torch(x):
    """
    抑制竖向 column fixed-pattern noise。
    x: [B,C,H,W], [0,1]
    """
    col_profile = x.mean(dim=2, keepdim=True)  # [B,C,1,W]
    col_profile = col_profile - col_profile.mean(dim=3, keepdim=True)
    return torch.abs(col_profile).mean()


def background_loss_torch(x, kernel_size=31, sigma=8.0):
    """
    抑制暗背景中的大尺度灰雾和低频背景。
    对细胞核这类亮目标图像，背景应尽量接近 0。
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    bg = gaussian_blur_torch(x, kernel_size=kernel_size, sigma=sigma)
    return torch.abs(bg).mean()


def preprocess_noisy_np(img, enable=True, denoise_h=8.0, bg_sigma=20.0, bg_strength=0.5):
    """
    可选的传统预处理：去列噪声 + 背景扣除 + NLM 去噪。
    这是为了处理显微图中常见的 fixed-pattern noise / 背景颗粒。
    """
    img = np.asarray(img, dtype=np.float32)
    if not enable:
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    # 1) 去列噪声：减去每列的中值偏移
    col_profile = np.median(img, axis=0, keepdims=True)
    col_profile = col_profile - np.median(col_profile)
    img = img - col_profile

    # 2) 背景扣除
    if bg_sigma > 0 and bg_strength > 0:
        bg = cv2.GaussianBlur(img, (0, 0), sigmaX=float(bg_sigma), sigmaY=float(bg_sigma))
        img = img - float(bg_strength) * bg

    # 3) 重新归一化
    img = img - img.min()
    img = img / (img.max() + 1e-8)

    # 4) NLM 去噪
    if denoise_h > 0:
        img_u8 = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        img_u8 = cv2.fastNlMeansDenoising(
            img_u8,
            None,
            h=float(denoise_h),
            templateWindowSize=7,
            searchWindowSize=21,
        )
        img = img_u8.astype(np.float32) / 255.0

    return np.clip(img, 0.0, 1.0).astype(np.float32)


# =========================================================
# 5.5 荧光显微可微退化 / 噪声模型
# =========================================================
def fluorescence_forward_torch(
    x01,
    psf_kernel_size=7,
    psf_sigma=1.0,
    gain=1.0,
    background=0.0,
):
    """
    可微荧光显微前向模型的确定性部分。

    clean fluorescence x -> PSF blur -> camera gain/background -> expected observation.

    注意：这里不直接 torch.poisson() 采样，因为随机 Poisson 采样不可微，
    GDP/cond_fn 需要对 x_t 求梯度。因此使用 Poisson-Gaussian 的期望与方差形式
    来构造负对数似然。
    """
    x01 = torch.clamp(x01, 0.0, 1.0)

    if psf_kernel_size % 2 == 0:
        psf_kernel_size += 1

    # 光学模糊/有限NA等效为高斯PSF近似
    blurred = gaussian_blur_torch(
        x01,
        kernel_size=int(psf_kernel_size),
        sigma=float(psf_sigma),
    )

    # 归一化强度域下的相机线性响应
    pred_mean01 = float(gain) * blurred + float(background)
    pred_mean01 = torch.clamp(pred_mean01, 0.0, 1.0)
    return pred_mean01


def poisson_gaussian_nll_loss_torch(
    pred_mean01,
    noisy01,
    photon_scale=80.0,
    read_noise_sigma=0.03,
    mask=None,
    eps=1e-8,
):
    """
    可微 Poisson-Gaussian 噪声一致性。

    观测模型近似：
        y_counts ~ Poisson(mu_counts) + Gaussian(0, sigma_read_counts^2)

    由于真实采样不可微，这里采用异方差 Gaussian 近似的 NLL：
        ((y-mu)^2 / var + log(var))
    其中 var ≈ mu + sigma_read_counts^2。

    pred_mean01/noisy01 均为 [0,1]。photon_scale 越大，表示光子数越高、shot noise 越弱。
    read_noise_sigma 是归一化强度域读出噪声标准差。
    """
    pred_mean01 = torch.clamp(pred_mean01, 0.0, 1.0)
    noisy01 = torch.clamp(noisy01, 0.0, 1.0)

    photon_scale = float(photon_scale)
    read_noise_sigma = float(read_noise_sigma)

    mu = pred_mean01 * photon_scale
    y = noisy01 * photon_scale

    read_sigma_counts = read_noise_sigma * photon_scale
    var = torch.clamp(mu + read_sigma_counts ** 2, min=eps)

    nll = (y - mu) ** 2 / (var + eps) + torch.log(var + eps)

    if mask is not None:
        nll = nll * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(float(nll.numel()), device=nll.device, dtype=nll.dtype)

    return nll.sum() / denom

def sobel_gradient_loss_torch(pred_clean, target_noisy, mask=None, eps=1e-8):
    """
    计算 Sobel 梯度幅值的差异 loss。
    pred_clean: [B,C,H,W] 预测的 clean 图像 (0~1)
    target_noisy: [B,C,H,W] 观测噪声图像 (0~1)
    """
    pred_grad = sobel_edges_torch(pred_clean)      # 复用已有的 sobel_edges_torch
    target_grad = sobel_edges_torch(target_noisy)
    diff = torch.abs(pred_grad - target_grad)
    if mask is not None:
        diff = diff * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(float(diff.numel()), device=diff.device, dtype=diff.dtype)
    return diff.sum() / denom

def log_loss_torch(pred_clean, target_noisy, sigma=1.5, mask=None, eps=1e-8):
    """
    LoG (Laplacian of Gaussian) 高频误差 loss，类似 HFEN。
    sigma: 高斯模糊 sigma，用于控制频率范围。
    """
    # 对 pred_clean 和 target_noisy 分别做高斯模糊
    pred_blur = gaussian_blur_torch(pred_clean, kernel_size=0, sigma=sigma)   # 注意 kernel_size=0 会在函数内部自动计算
    target_blur = gaussian_blur_torch(target_noisy, kernel_size=0, sigma=sigma)
    # Laplacian
    pred_log = laplacian_torch(pred_blur)       # 需要实现一个简单的拉普拉斯卷积
    target_log = laplacian_torch(target_blur)
    diff = torch.abs(pred_log - target_log)
    if mask is not None:
        diff = diff * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(float(diff.numel()), device=diff.device, dtype=diff.dtype)
    return diff.sum() / denom

def laplacian_torch(x):
    """3x3 拉普拉斯核，用于 LoG loss"""
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    c = x.shape[1]
    kernel = kernel.repeat(c, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=c)

def multiscale_laplacian_loss_torch(pred_clean, target_noisy, scales=[1,2,4], mask=None):
    """
    多尺度 Laplacian loss: 对图像进行下采样，计算不同尺度的拉普拉斯差异。
    scales: 下采样倍数列表，例如 [1,2,4] 表示原尺度、1/2尺度、1/4尺度。
    """
    total_loss = 0.0
    for s in scales:
        if s == 1:
            pred_s = pred_clean
            target_s = target_noisy
        else:
            # 使用双线性下采样
            pred_s = F.interpolate(pred_clean, scale_factor=1.0/s, mode='bilinear', align_corners=False)
            target_s = F.interpolate(target_noisy, scale_factor=1.0/s, mode='bilinear', align_corners=False)
        # 拉普拉斯滤波（不先做高斯模糊，直接用拉普拉斯核）
        lap_pred = laplacian_torch(pred_s)
        lap_target = laplacian_torch(target_s)
        diff = torch.abs(lap_pred - lap_target)
        if mask is not None:
            # 注意 mask 也需要同步下采样
            mask_s = F.interpolate(mask, scale_factor=1.0/s, mode='nearest')
            diff = diff * mask_s
            denom = mask_s.sum().clamp_min(1.0)
        else:
            denom = torch.tensor(float(diff.numel()), device=diff.device, dtype=diff.dtype)
        total_loss += diff.sum() / denom
    return total_loss / len(scales)

def charbonnier_loss_masked(x, y, mask=None, eps=1e-3):
    """辅助的鲁棒一致性项，可用于弱保真约束。"""
    residual = x - y
    if mask is not None:
        residual = residual * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(float(residual.numel()), device=residual.device, dtype=residual.dtype)
    return torch.sqrt(residual ** 2 + float(eps)).sum() / denom


# =========================================================
# 6. Self2Self-style 扩散去噪 cond_fn
# =========================================================
def denoise_cond_fn(x, t, noisy=None, args=None, **kwargs):
    """
    荧光显微物理噪声模型引导的 diffusion cond_fn。

    当前版本的核心逻辑：
    1. 当前扩散状态 x_t -> x01，视作当前伪 clean fluorescence image；
    2. x01 经过可微荧光前向模型 H：PSF blur + camera gain/background；
    3. 用 Poisson-Gaussian 近似 NLL 比较 H(x01) 与 noisy observation；
    4. 对 x_t 求梯度，并返回给 guided_diffusion 采样器；
    5. 去除原来较多的 edge/SSIM 等直接 noisy 匹配项，仅保留少量 TV/背景/条纹正则。
    """
    assert noisy is not None
    assert args is not None

    with th.enable_grad():
        x_t = x.detach().requires_grad_(True)
        x_t = torch.clamp(x_t, -1.0, 1.0)

        noisy = noisy.to(device=x_t.device, dtype=x_t.dtype)

        # 当前扩散变量映射到 [0,1]，作为伪 clean fluorescence image
        x01 = (x_t + 1.0) / 2.0
        noisy01 = ((noisy + 1.0) / 2.0).detach()

        # -----------------------------------------------------
        # Self2Self-style blind mask：只在随机位置做观测一致性
        # 这样可降低直接复制 noisy 高频噪声的风险。
        # -----------------------------------------------------
        if bool(args.use_self2self_mask):
            mask_ratio = float(args.self2self_mask_ratio)
            mask_ratio = max(0.0, min(mask_ratio, 0.95))
            loss_mask = (torch.rand_like(noisy01) < mask_ratio).to(dtype=x_t.dtype)
        else:
            loss_mask = torch.ones_like(noisy01)

        # -----------------------------------------------------
        # 荧光显微可微前向模型：clean -> degraded/noisy expectation
        # H(x) = gain * (x (*) PSF) + background
        # -----------------------------------------------------
        pred_noisy_mean01 = fluorescence_forward_torch(
            x01,
            psf_kernel_size=args.fluor_psf_kernel_size,
            psf_sigma=args.fluor_psf_sigma,
            gain=args.fluor_gain,
            background=args.fluor_background,
        )

        # -----------------------------------------------------
        # Poisson-Gaussian 噪声模型一致性
        # 使用 NLL 的可微近似，而不是不可微的 Poisson 随机采样。
        # -----------------------------------------------------
        pg_loss = poisson_gaussian_nll_loss_torch(
            pred_noisy_mean01,
            noisy01,
            photon_scale=args.fluor_photon_scale,
            read_noise_sigma=args.fluor_read_noise_sigma,
            mask=loss_mask,
            eps=args.charbonnier_eps,
        )
        
        # 2. 新增 Sobel 梯度 loss
        if args.sobel_weight > 0:
            sobel_loss = sobel_gradient_loss_torch(
                x01, noisy01, mask=loss_mask, eps=args.charbonnier_eps
            )
        else:
            sobel_loss = torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)

        # 3. 新增 LoG (HFEN) loss
        if args.log_weight > 0:
            log_loss_val = log_loss_torch(
                x01, noisy01, sigma=args.log_sigma, mask=loss_mask, eps=args.charbonnier_eps
            )
        else:
            log_loss_val = torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)

        # 4. 新增多尺度 Laplacian loss
        if args.mslap_weight > 0:
            mslap_loss = multiscale_laplacian_loss_torch(
                x01, noisy01, scales=args.mslap_scales, mask=loss_mask
            )
        else:
            mslap_loss = torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        # 可选：加入一个很弱的低频 Charbonnier 一致性，稳定亮度/背景。
        if args.lowfreq_consistency_weight > 0:
            pred_low = gaussian_blur_torch(
                pred_noisy_mean01,
                kernel_size=args.consistency_blur_size,
                sigma=args.consistency_blur_sigma,
            )
            noisy_low = gaussian_blur_torch(
                noisy01,
                kernel_size=args.consistency_blur_size,
                sigma=args.consistency_blur_sigma,
            )
            lowfreq_loss = charbonnier_loss_masked(
                pred_low,
                noisy_low,
                mask=loss_mask,
                eps=args.charbonnier_eps,
            )
        else:
            lowfreq_loss = torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)

        # 少量正则：去噪任务中避免过多 loss 把结构拉回 noisy
        if args.tv_weight > 0:
            tv_l = tv_loss_torch(x01)
        else:
            tv_l = torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)

        if args.bg_weight > 0:
            bg_l = background_loss_torch(
                x01,
                kernel_size=args.bg_kernel_size,
                sigma=args.bg_sigma,
            )
        else:
            bg_l = torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)

        if args.stripe_weight > 0:
            stripe_l = column_stripe_loss_torch(x01)
        else:
            stripe_l = torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)

        t_val = int(t[0].item())
        if t_val > 900:
            anneal = args.anneal_high
        elif t_val > 700:
            anneal = args.anneal_mid_high
        elif t_val > 400:
            anneal = args.anneal_mid
        else:
            anneal = args.anneal_low

        total = (
                args.img_guidance_scale * pg_loss                     # 主 loss，权重 ~90
                + args.sobel_weight * sobel_loss                      # 建议 0.5 ~ 2.0
                + args.log_weight * log_loss_val                      # 建议 0.5 ~ 2.0
                + args.mslap_weight * mslap_loss                      # 建议 0.2 ~ 1.0
                + args.lowfreq_consistency_weight * lowfreq_loss
                + args.tv_weight * tv_l
                + args.bg_weight * bg_l
                + args.stripe_weight * stripe_l
            )

        # guided_diffusion 的 cond_fn 返回加到采样方向上的梯度。
        # 为最小化 total，这里最大化 -total。
        loss = -total * anneal
        grad = th.autograd.grad(loss, x_t)[0]

        if args.cond_log_interval > 0 and (t_val % args.cond_log_interval == 0 or t_val == 0):
            print(
                f"[t={t_val:04d}] "
                f"pg={pg_loss.item():.6f}, "
                f"lowfreq={lowfreq_loss.item():.6f}, "
                f"tv={tv_l.item():.6f}, "
                f"bg={bg_l.item():.6f}, "
                f"stripe={stripe_l.item():.6f}, "
                f"mask={float(loss_mask.mean().detach().cpu().item()):.3f}, "
                f"psf_sigma={float(args.fluor_psf_sigma):.3f}, "
                f"photons={float(args.fluor_photon_scale):.1f}, "
                f"read={float(args.fluor_read_noise_sigma):.4f}, "
                f"anneal={anneal:.2f}, "
                f"grad_norm={grad.norm().item():.6f}"
            )

        return grad


# =========================================================
# 7. 模型加载
# =========================================================
def create_your_unet_and_diffusion(args, device):
    if not os.path.isdir(args.unet_model_path):
        raise FileNotFoundError(f"unet_model_path does not exist: {args.unet_model_path}")

    unet_subdir = os.path.join(args.unet_model_path, "unet")
    if not os.path.isdir(unet_subdir):
        raise FileNotFoundError(f"Missing unet subfolder: {unet_subdir}")

    unet = UNet2DConditionModel.from_pretrained(
        args.unet_model_path,
        subfolder="unet",
    ).to(device)

    logger.log(f"加载 UNet 成功: {unet_subdir}")

    if args.use_fp16:
        unet = unet.half()

    # 默认 eval；如果模型内部有 Dropout，可通过 use_model_dropout=True 在采样时开启。
    # GroupNorm 不依赖 running statistics，因此对 diffusers UNet 通常是安全的。
    if getattr(args, "use_model_dropout", False):
        unet.train()
    else:
        unet.eval()

    for p in unet.parameters():
        p.requires_grad = False

    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    return unet, diffusion


# =========================================================
# 8. 单 tile 去噪
# =========================================================
def denoise_one_tile(
    noisy_tile,
    model,
    diffusion,
    encoder_hidden_states,
    device,
    args,
    seed=1234,
):
    """
    noisy_tile: [H,W], float32, [0,1]
        这里的 noisy_tile 只作为 Raw/观测图像引导项，不作为扩散采样起点。
    return: denoised_tile [H,W], float32, [0,1]
    """
    h, w = noisy_tile.shape

    if h != args.tile_size or w != args.tile_size:
        noisy_tile = cv2.resize(
            noisy_tile,
            (args.tile_size, args.tile_size),
            interpolation=cv2.INTER_CUBIC,
        )

    noisy_tensor = torch.from_numpy(
        (noisy_tile * 2.0 - 1.0)[None, None].astype(np.float32)
    ).to(device)
    if args.use_fp16:
        noisy_tensor = noisy_tensor.half()

    shape = (1, 1, args.tile_size, args.tile_size)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # =========================================================
    # 初始化方式：
    # gaussian        : 纯高斯生成，适合生成，不适合保真去噪；
    # noisy           : 直接从 noisy refinement，结构最稳；
    # noisy_plus_noise: noisy + 少量高斯扰动，推荐普通去噪。
    # =========================================================
    if args.init_mode == "gaussian":
        noise = torch.randn(
            shape,
            generator=gen,
            device=device,
            dtype=torch.float16 if args.use_fp16 else torch.float32,
        )
    elif args.init_mode == "noisy":
        noise = noisy_tensor.clone()
    elif args.init_mode == "noisy_plus_noise":
        eps = torch.randn(
            shape,
            generator=gen,
            device=device,
            dtype=torch.float16 if args.use_fp16 else torch.float32,
        )
        noise = torch.clamp(noisy_tensor + float(args.init_noise_sigma) * eps, -1.0, 1.0)
    else:
        raise ValueError(f"Unsupported init_mode: {args.init_mode}")
    # 关键修复：当 args.use_fp16=True 时，UNet 权重是 float16，
    # 但 guided_diffusion 传入的 x_in 可能是 float32。
    # 所以进入 UNet 前强制转换到模型 dtype，返回时再转回 float32 给 diffusion。
    model_dtype = next(model.parameters()).dtype
    encoder_hidden_states_model = encoder_hidden_states.to(device=device, dtype=model_dtype)
    model_kwargs = {"encoder_hidden_states": encoder_hidden_states_model}

    def model_forward(x_in, t, **kwargs):
        out = model(
            x_in.to(dtype=model_dtype),
            t,
            encoder_hidden_states=encoder_hidden_states_model,
        ).sample
        return out.float()

    progressive_samples = diffusion.p_sample_loop_progressive(
        model_forward,
        shape,
        noise=noise,
        model_kwargs=model_kwargs,
        cond_fn=lambda x_in, t, y=None, **kwargs: denoise_cond_fn(
            x_in,
            t,
            noisy=noisy_tensor,
            args=args,
        ),
        device=device,
        denoise_steps=args.denoise_steps,
        clip_denoised=args.clip_denoised,
    )

    final = None
    final_x0 = None

    for out in progressive_samples:
        if isinstance(out, dict):
            final = out["sample"] if "sample" in out else out
            final_x0 = out["pred_xstart"] if "pred_xstart" in out else final
        else:
            final = out
            final_x0 = out

    if args.output_source == "sample":
        out_tensor = final
    else:
        out_tensor = final_x0

    out01 = ((out_tensor + 1.0) / 2.0).clamp(0, 1)
    out_np = out01.squeeze().detach().float().cpu().numpy().astype(np.float32)

    if out_np.shape != (h, w):
        out_np = cv2.resize(out_np, (w, h), interpolation=cv2.INTER_CUBIC)

    return np.clip(out_np, 0.0, 1.0).astype(np.float32)


# =========================================================
# 9. 单图去噪：整图分块 + Hanning 融合
# =========================================================
def denoise_one_image(
    case_name,
    noisy_path,
    gt_path,
    case_save_dir,
    model,
    diffusion,
    encoder_hidden_states,
    device,
    args,
):
    os.makedirs(case_save_dir, exist_ok=True)

    t0 = time.time()

    noisy = read_gray_image_float01(
        noisy_path,
        percentile_norm=args.percentile_norm,
    ).astype(np.float32)

    save_float01_image(noisy, os.path.join(case_save_dir, "noisy_raw.png"))

    # 可选预处理：先去掉一部分显式背景/列噪声，再进入 diffusion refinement
    noisy = preprocess_noisy_np(
        noisy,
        enable=args.use_preprocess,
        denoise_h=args.preprocess_nlm_h,
        bg_sigma=args.preprocess_bg_sigma,
        bg_strength=args.preprocess_bg_strength,
    )

    save_float01_image(noisy, os.path.join(case_save_dir, "noisy.png"))
    save_float01_tif_8bit(noisy, os.path.join(case_save_dir, "noisy.tif"))

    gt = None
    if gt_path is not None:
        gt = read_gray_image_float01(
            gt_path,
            percentile_norm=args.percentile_norm,
        ).astype(np.float32)

        if gt.shape != noisy.shape:
            gt = cv2.resize(gt, (noisy.shape[1], noisy.shape[0]), interpolation=cv2.INTER_CUBIC)

        save_float01_image(gt, os.path.join(case_save_dir, "gt.png"))
        save_float01_tif_8bit(gt, os.path.join(case_save_dir, "gt.tif"))

    tile_size = int(args.tile_size)
    stride = int(args.tile_stride)

    noisy_pad, pad_info = pad_2d_reflect_to_min_size_and_multiple(noisy, tile_size)
    H, W = noisy_pad.shape

    ys = get_tile_starts(H, tile_size, stride)
    xs = get_tile_starts(W, tile_size, stride)

    merged = np.zeros_like(noisy_pad, dtype=np.float32)
    weight_map = np.zeros_like(noisy_pad, dtype=np.float32)
    weight = make_hanning_weight(tile_size, tile_size, eps=1e-3)

    logger.log(
        f"[{case_name}] noisy={noisy.shape}, padded={noisy_pad.shape}, "
        f"tile={tile_size}, stride={stride}, num_tiles={len(ys)}x{len(xs)}"
    )

    tile_id = 0
    for y in ys:
        for x in xs:
            noisy_tile = noisy_pad[y:y + tile_size, x:x + tile_size]

            seed = int(args.tile_noise_seed) + tile_id

            # Self2Self-style MC ensemble:
            # 同一个 tile 多次随机 mask / 随机初始化 / 可选 dropout，最后平均以降低方差。
            mc_tiles = []
            mc_samples = max(1, int(args.self2self_mc_samples))
            for mc_id in range(mc_samples):
                den_tile_mc = denoise_one_tile(
                    noisy_tile=noisy_tile,
                    model=model,
                    diffusion=diffusion,
                    encoder_hidden_states=encoder_hidden_states,
                    device=device,
                    args=args,
                    seed=seed + mc_id * 10000,
                )
                mc_tiles.append(den_tile_mc)

            den_tile = np.mean(mc_tiles, axis=0).astype(np.float32)

            merged[y:y + tile_size, x:x + tile_size] += den_tile * weight
            weight_map[y:y + tile_size, x:x + tile_size] += weight

            if args.save_tiles:
                tile_dir = os.path.join(case_save_dir, "tiles")
                os.makedirs(tile_dir, exist_ok=True)
                save_float01_image(noisy_tile, os.path.join(tile_dir, f"tile_{tile_id:04d}_noisy.png"))
                save_float01_image(den_tile, os.path.join(tile_dir, f"tile_{tile_id:04d}_denoised.png"))

            logger.log(
                f"[{case_name}] tile {tile_id + 1:03d}/{len(ys) * len(xs):03d} done"
            )
            tile_id += 1

    den_padded = merged / (weight_map + 1e-8)
    den_padded = np.clip(den_padded, 0.0, 1.0).astype(np.float32)
    den = crop_by_pad_info(den_padded, pad_info)
    den = np.clip(den, 0.0, 1.0).astype(np.float32)

    save_float01_image(den_padded, os.path.join(case_save_dir, "denoised_padded.png"))
    save_float01_image(weight_map / (weight_map.max() + 1e-8), os.path.join(case_save_dir, "weight_map.png"))

    save_float01_image(den, os.path.join(case_save_dir, "denoised_final.png"))
    save_float01_tif_8bit(den, os.path.join(case_save_dir, "denoised_final.tif"))

    runtime = time.time() - t0

    summary = {
        "case_name": case_name,
        "noisy_path": noisy_path,
        "gt_path": gt_path,
        "case_save_dir": case_save_dir,
        "runtime_sec": float(runtime),
        "image_shape": list(map(int, noisy.shape)),
        "tile_size": int(tile_size),
        "tile_stride": int(stride),
        "num_tiles": int(len(ys) * len(xs)),
        "metrics": None,
    }

    if gt is not None:
        metrics = compute_metrics(den, gt)
        summary["metrics"] = metrics

        save_heatmap(
            den,
            gt,
            os.path.join(case_save_dir, "diff_heatmap.png"),
        )

        logger.log(
            f"[{case_name}] PSNR={metrics['psnr']:.4f}, "
            f"SSIM={metrics['ssim']:.4f}, "
            f"NRMSE={metrics['nrmse']:.6f}, "
            f"HFEN={metrics['hfen']:.4f}, "
            f"GMS={metrics['gms']:.4f}, "
            f"time={runtime:.2f}s"
        )
    else:
        logger.log(f"[{case_name}] done, time={runtime:.2f}s")

    with open(os.path.join(case_save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# =========================================================
# 10. 样本列表与汇总
# =========================================================
def build_case_list(args):
    noisy_files = list_image_files(args.noisy_root)

    if len(noisy_files) == 0:
        raise FileNotFoundError(f"No image files found in noisy_root: {args.noisy_root}")

    cases = []
    for noisy_path in noisy_files:
        stem = os.path.splitext(os.path.basename(noisy_path))[0]
        gt_path = resolve_same_name(args.gt_root, stem) if args.gt_root else None

        if args.require_gt and gt_path is None:
            logger.log(f"跳过 {stem}: 未找到 GT")
            continue

        cases.append(
            {
                "case_name": stem,
                "noisy_path": noisy_path,
                "gt_path": gt_path,
            }
        )

        if len(cases) >= args.num_cases:
            break

    if len(cases) == 0:
        raise RuntimeError("No valid cases found.")

    return cases


def save_dataset_summary(all_summaries, save_dir):
    """
    保存数据集汇总结果。

    输出文件：
    1) dataset_summary.json
       - 完整嵌套结构，包含每个样本的 metrics。

    2) dataset_summary.csv
       - 每个样本一行，包含 PSNR / SSIM / NRMSE / MSE / HFEN / GMS。
       - 使用 utf-8-sig，方便 Excel 正常打开。

    3) metrics_all_cases.csv
       - 更直接的指标总表，只保留核心路径和指标列。
       - 推荐后续论文统计和 Excel 排序优先看这个文件。
    """
    os.makedirs(save_dir, exist_ok=True)

    valid_metrics = [x["metrics"] for x in all_summaries if x["metrics"] is not None]

    dataset_summary = {
        "num_cases": len(all_summaries),
        "num_cases_with_gt": len(valid_metrics),
        "average_runtime_sec": float(np.mean([x["runtime_sec"] for x in all_summaries])) if len(all_summaries) > 0 else 0.0,
        "average_metrics": None,
        "per_case": all_summaries,
    }

    if len(valid_metrics) > 0:
        dataset_summary["average_metrics"] = {
            "psnr": float(np.mean([m["psnr"] for m in valid_metrics])),
            "ssim": float(np.mean([m["ssim"] for m in valid_metrics])),
            "nrmse": float(np.mean([m["nrmse"] for m in valid_metrics])),
            "mse": float(np.mean([m["mse"] for m in valid_metrics])),
            "hfen": float(np.mean([m["hfen"] for m in valid_metrics])),
            "gms": float(np.mean([m["gms"] for m in valid_metrics])),
        }

    with open(os.path.join(save_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)

    header = [
        "case_name",
        "runtime_sec",
        "has_gt",
        "psnr",
        "ssim",
        "nrmse",
        "mse",
        "hfen",
        "gms",
        "noisy_path",
        "gt_path",
        "case_save_dir",
    ]

    rows = []
    for x in all_summaries:
        m = x["metrics"]
        rows.append([
            x["case_name"],
            x["runtime_sec"],
            m is not None,
            "" if m is None else m["psnr"],
            "" if m is None else m["ssim"],
            "" if m is None else m["nrmse"],
            "" if m is None else m["mse"],
            "" if m is None else m["hfen"],
            "" if m is None else m["gms"],
            x["noisy_path"],
            x["gt_path"],
            x["case_save_dir"],
        ])

    # 1) 原汇总 CSV：包含新增 HFEN/GMS
    csv_path = os.path.join(save_dir, "dataset_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # 2) 更直观的指标总表：推荐打开这个文件
    metrics_csv_path = os.path.join(save_dir, "metrics_all_cases.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_name",
            "psnr",
            "ssim",
            "nrmse",
            "mse",
            "hfen",
            "gms",
            "runtime_sec",
            "noisy_path",
            "gt_path",
            "case_save_dir",
        ])

        for x in all_summaries:
            m = x["metrics"]
            writer.writerow([
                x["case_name"],
                "" if m is None else m["psnr"],
                "" if m is None else m["ssim"],
                "" if m is None else m["nrmse"],
                "" if m is None else m["mse"],
                "" if m is None else m["hfen"],
                "" if m is None else m["gms"],
                x["runtime_sec"],
                x["noisy_path"],
                x["gt_path"],
                x["case_save_dir"],
            ])

    return dataset_summary



def save_topk(all_summaries, save_dir, topk=20, metric="psnr"):
    valid = [x for x in all_summaries if x["metrics"] is not None]
    if len(valid) == 0:
        return None

    lower_is_better = metric.lower() in ["mse", "nrmse", "hfen"]
    valid.sort(key=lambda x: x["metrics"][metric], reverse=not lower_is_better)
    valid = valid[:min(topk, len(valid))]

    top_dir = os.path.join(save_dir, f"top{len(valid)}_{metric}")
    os.makedirs(top_dir, exist_ok=True)

    rows = []

    for rank, item in enumerate(valid, start=1):
        case_dir = item["case_save_dir"]
        out_dir = os.path.join(top_dir, f"rank_{rank:02d}_{item['case_name']}")
        os.makedirs(out_dir, exist_ok=True)

        safe_copy(os.path.join(case_dir, "noisy.png"), os.path.join(out_dir, "noisy.png"))
        safe_copy(os.path.join(case_dir, "denoised_final.png"), os.path.join(out_dir, "denoised_final.png"))
        safe_copy(os.path.join(case_dir, "gt.png"), os.path.join(out_dir, "gt.png"))
        safe_copy(os.path.join(case_dir, "diff_heatmap.png"), os.path.join(out_dir, "diff_heatmap.png"))
        safe_copy(os.path.join(case_dir, "metrics.json"), os.path.join(out_dir, "metrics.json"))

        rows.append({
            "rank": rank,
            "case_name": item["case_name"],
            "metric": metric,
            "value": item["metrics"][metric],
            "psnr": item["metrics"]["psnr"],
            "ssim": item["metrics"]["ssim"],
            "nrmse": item["metrics"]["nrmse"],
            "mse": item["metrics"]["mse"],
            "hfen": item["metrics"]["hfen"],
            "gms": item["metrics"]["gms"],
            "case_save_dir": case_dir,
        })

    with open(os.path.join(top_dir, "topk_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    with open(os.path.join(top_dir, "topk_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "case_name", "metric", "value", "psnr", "ssim", "nrmse", "mse", "hfen", "gms", "case_save_dir"])
        for r in rows:
            writer.writerow([
                r["rank"],
                r["case_name"],
                r["metric"],
                r["value"],
                r["psnr"],
                r["ssim"],
                r["nrmse"],
                r["mse"],
                r["hfen"],
                r["gms"],
                r["case_save_dir"],
            ])

    return top_dir


# =========================================================
# 11. 参数
# =========================================================
def create_argparser():
    """
    关键修改：
    1) 先读取 guided_diffusion.script_util_x0.model_and_diffusion_defaults() 的完整默认参数；
    2) 再用 defaults.update(...) 覆盖本任务自己的参数。

    这样 args_to_dict(args, model_and_diffusion_defaults().keys()) 不会再因为
    image_size / num_channels / num_res_blocks 等字段缺失而报 AttributeError。
    """
    defaults = model_and_diffusion_defaults()

    defaults.update(dict(
        # 数据路径
        noisy_root="/data3/ddt/BIOSR-SIM/数据集/1014/1",
        gt_root="/data3/ddt/BIOSR-SIM/数据集/1014/3",
        require_gt=False,

        # 模型路径
        unet_model_path="/data3/ddt/BIOSR-SIM/checkpoint-285000",
        text_encoder_path="/data3/ddt/BIOSR-SIM/checkpoint-285000/text_encoder",
        tokenizer_path="/data3/ddt/BIOSR-SIM/checkpoint-285000/tokenizer",

        # prompt
        prompt="mitochondria of BPAE",

        # 输出
        save_dir="./output_10141014",
        save_tiles=False,
        percentile_norm=False,

        # 设备
        device=0,
        use_fp16=True,

        # 采样 / 分块
        num_cases=100,
        denoise_steps=1000,
        output_source="pred_xstart",  # pred_xstart 更适合去噪；sample 更随机
        tile_size=256,
        tile_stride=192,
        tile_noise_seed=1234,

        # 起点设置：gaussian 表示从高斯纯噪声 x_T 开始；Raw 只进入 cond_fn 做引导，不作为采样初值
        init_mode="noisy_plus_noise",

        # 为了和 tile_size 对齐，默认设置为 256；main() 中还会强制同步 image_size=tile_size
        image_size=256,
        clip_denoised=True,

        # 荧光物理噪声模型引导强度
        img_guidance_scale=89.96860923080581,

        # 原 edge/SSIM 直接匹配 noisy 的约束已不再进入 total，保留参数仅为兼容旧命令行
        edge_weight=0.0,
        ssim_weight=0.0,

        # 少量正则
        tv_weight=0.02,
        charbonnier_eps=5e-06,

        # Self2Self-inspired 参数
        use_self2self_mask=True,
        self2self_mask_ratio=0.4,
        self2self_mc_samples=4,
        use_model_dropout=False,

        # 初始化扰动
        init_noise_sigma=0.15,

        # 旧版 noisy 一致性
        use_lowfreq_data_loss=True,

        # 荧光显微退化/噪声模型参数
        fluor_psf_sigma=0.3,
        fluor_psf_kernel_size=3,
        fluor_gain=0.8874204633453998,
        fluor_background=0.0,
        fluor_photon_scale=20.0,
        fluor_read_noise_sigma=0.08,

        # 弱低频一致性
        lowfreq_consistency_weight=1.0,

        # 背景/条纹噪声抑制
        bg_weight=0.08,
        bg_sigma=8.0,
        bg_kernel_size=31,
        stripe_weight=0.0,

        # 输入预处理
        use_preprocess=False,
        preprocess_bg_sigma=9.656471622836541,
        preprocess_bg_strength=0.3175080661653702,
        preprocess_nlm_h=2.0,

        # 低频一致性滤波
        consistency_blur_sigma=0.3,
        consistency_blur_size=9,
        edge_blur_sigma=0.4109510323328211,
        edge_blur_size=3,

        # 退火
        anneal_high=5.0,
        anneal_low=0.2,
        anneal_mid=1.3253091095573837,
        anneal_mid_high=0.5,

        # 日志 / TopK
        cond_log_interval=100,
        topk=20,
        topk_metric="psnr",
        
        sobel_weight=0.8,                 # Sobel 梯度 loss 权重
        log_weight=0.8,                   # LoG (HFEN) loss 权重
        log_sigma=1.5,                    # LoG 高斯模糊的 sigma
        mslap_weight=0.5,                 # 多尺度 Laplacian loss 权重
        mslap_scales=[1,2,4],             # 多尺度倍数列表（以字符串形式传递，需解析      
    ))

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

# =========================================================
# 12. main
# =========================================================
def main():
    args = create_argparser().parse_args()

    if isinstance(args.mslap_scales, str):
        args.mslap_scales = [int(s) for s in args.mslap_scales.split(',')]
    # 强制让 diffusion 的 image_size 与 tile_size 一致。
    # 你的任务是 tile-by-tile 去噪，扩散模型每次处理的尺寸就是 tile_size。
    args.image_size = int(args.tile_size)

    if int(args.tile_stride) <= 0:
        raise ValueError("--tile_stride must be > 0")
    if int(args.tile_stride) > int(args.tile_size):
        raise ValueError("--tile_stride must be <= --tile_size")
    if int(args.denoise_steps) > int(args.diffusion_steps):
        raise ValueError(
            f"--denoise_steps={args.denoise_steps} cannot be larger than "
            f"--diffusion_steps={args.diffusion_steps}"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    logger.configure(dir=args.save_dir)

    if args.gt_root == "":
        args.gt_root = None

    if args.output_source not in ["pred_xstart", "sample"]:
        raise ValueError("--output_source must be pred_xstart or sample")

    if args.init_mode not in ["gaussian", "noisy", "noisy_plus_noise"]:
        raise ValueError("--init_mode must be one of: gaussian, noisy, noisy_plus_noise")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    logger.log("=" * 80)
    logger.log("Fluorescence Poisson-Gaussian degradation guided diffusion 去噪任务")
    logger.log(f"noisy_root: {args.noisy_root}")
    logger.log(f"gt_root   : {args.gt_root}")
    logger.log(f"save_dir  : {args.save_dir}")
    logger.log(f"device    : {device}")
    logger.log(f"init_mode : {args.init_mode}")
    logger.log(f"init_noise_sigma: {getattr(args, 'init_noise_sigma', None)}")
    logger.log(f"Self2Self mask: {args.use_self2self_mask}, mask_ratio={args.self2self_mask_ratio}, mc_samples={args.self2self_mc_samples}")
    logger.log(f"fluor model: psf_sigma={args.fluor_psf_sigma}, photons={args.fluor_photon_scale}, read_noise={args.fluor_read_noise_sigma}")
    logger.log(f"preprocess: {args.use_preprocess}, output_source={args.output_source}")
    logger.log("推荐普通去噪使用 noisy_plus_noise/noisy；纯 gaussian 更适合生成，不适合保真去噪。")
    logger.log("=" * 80)

    logger.log("创建扩散模型...")
    model, diffusion = create_your_unet_and_diffusion(args, device)

    logger.log("加载文本模型...")
    tokenizer, text_encoder = load_text_model(args, device)
    encoder_hidden_states = build_text_condition(
        tokenizer,
        text_encoder,
        args.prompt,
        device,
    )

    cases = build_case_list(args)
    logger.log(f"找到 {len(cases)} 个样本")

    all_summaries = []

    for idx, case in enumerate(cases, start=1):
        logger.log("=" * 80)
        logger.log(f"处理第 {idx}/{len(cases)} 个样本: {case['case_name']}")

        case_save_dir = os.path.join(args.save_dir, case["case_name"])

        summary = denoise_one_image(
            case_name=case["case_name"],
            noisy_path=case["noisy_path"],
            gt_path=case["gt_path"],
            case_save_dir=case_save_dir,
            model=model,
            diffusion=diffusion,
            encoder_hidden_states=encoder_hidden_states,
            device=device,
            args=args,
        )

        all_summaries.append(summary)

    dataset_summary = save_dataset_summary(all_summaries, args.save_dir)
    topk_dir = save_topk(
        all_summaries,
        args.save_dir,
        topk=args.topk,
        metric=args.topk_metric,
    )

    logger.log("=" * 80)
    logger.log("全部完成")
    logger.log(f"summary json: {os.path.join(args.save_dir, 'dataset_summary.json')}")
    logger.log(f"summary csv : {os.path.join(args.save_dir, 'dataset_summary.csv')}")
    logger.log(f"metrics csv : {os.path.join(args.save_dir, 'metrics_all_cases.csv')}")

    if dataset_summary["average_metrics"] is not None:
        m = dataset_summary["average_metrics"]
        logger.log(
            f"Average: PSNR={m['psnr']:.4f}, "
            f"SSIM={m['ssim']:.4f}, "
            f"NRMSE={m['nrmse']:.6f}, "
            f"HFEN={m['hfen']:.4f}, "
            f"GMS={m['gms']:.4f}"
        )
        print(m['psnr']) 
    # if topk_dir is not None:
    #     logger.log(f"Top-K saved: {topk_dir}")


if __name__ == "__main__":
    main()