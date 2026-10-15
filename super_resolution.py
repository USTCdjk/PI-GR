import argparse
import os
import time
import json
import re
import csv
import shutil
import glob
import cv2
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import tifffile as tiff
from PIL import Image

from models.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

from guided_diffusion import logger
from guided_diffusion.script_util_x0 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# ===== 直接引入你的九通道 SIM forward model =====

from degraded import SimpleStableSIM


def add_label(img, text, font_scale=0.6, thickness=2, margin=10):
    """在图像左上角添加绿色文字，并带黑色轮廓以提高可读性"""
    h, w = img.shape[:2]
    # 设定字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 获取文本尺寸
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # 定义背景矩形（可选，增强对比）
    cv2.rectangle(img, (margin-2, margin-2), (margin+text_w+2, margin+text_h+2), (0,0,0), -1)
    # 绘制绿色文字
    cv2.putText(img, text, (margin, margin+text_h), font, font_scale, (0,255,0), thickness)
    return img

# =========================================================
# 1. 工具函数
# =========================================================
def natural_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def list_tif_files(folder):
    exts = (".tif", ".tiff")
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(exts)
    ]
    files.sort(key=lambda x: natural_key(os.path.basename(x)))
    return files


def normalize_to_float01(img):
    img = np.asarray(img).astype(np.float32)
    mn = float(img.min())
    mx = float(img.max())
    return (img - mn) / (mx - mn + 1e-8)


def read_single_image_as_float01(path):
    """
    读取单张 2D tif 图像（WF / GT 用）
    """
    try:
        img = tiff.imread(path)
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.ANYDEPTH)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    img = np.asarray(img)
    img = np.squeeze(img)

    if img.ndim == 2:
        return normalize_to_float01(img)

    if img.ndim == 3:
        if img.shape[0] == 1:
            return normalize_to_float01(img[0])
        if img.shape[-1] == 1:
            return normalize_to_float01(img[..., 0])

    raise ValueError(
        f"Expected a single 2D image for path={path}, but got shape={img.shape}. "
        f"If this is a SIM stack, it should be read by the SIM loader instead."
    )


def save_float01_image(x, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.clip(x, 0.0, 1.0)
    x_u8 = (x * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(x_u8, mode="L").save(path)


# =========================================================
# Tile inference helpers: overlap tile + Hanning blending
# =========================================================
def get_tile_starts(length, tile_size, stride):
    """
    生成 tile 起点，保证最后一个 tile 一定贴住右/下边界。
    例：length=512, tile_size=128, stride=96 -> [0, 96, 192, 288, 384]
    """
    if length < tile_size:
        raise ValueError(
            f"length={length} < tile_size={tile_size}. Please pad the image first."
        )
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if len(starts) == 0 or starts[-1] != last:
        starts.append(last)
    return starts


def make_hanning_weight(h, w, eps=1e-3):
    """
    2D Hanning 权重。中心权重大，边缘权重小。
    eps 防止最外圈严格为 0 导致除零或黑边。
    """
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    weight = np.outer(wy, wx).astype(np.float32)
    return np.maximum(weight, eps)


def pad_2d_reflect_to_multiple(img, multiple, pad_each_side_min=0):
    """
    对 2D 图像 reflect padding 到 multiple 的整数倍。
    对 502, multiple=128, pad_each_side_min=5 时，会得到 512。
    返回: padded_img, (top,bottom,left,right)
    """
    h, w = img.shape
    target_h = int(np.ceil(h / multiple) * multiple)
    target_w = int(np.ceil(w / multiple) * multiple)

    pad_h = max(target_h - h, 2 * pad_each_side_min)
    pad_w = max(target_w - w, 2 * pad_each_side_min)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = np.pad(img, ((top, bottom), (left, right)), mode="reflect")
    return padded.astype(np.float32), (top, bottom, left, right)


def pad_sim_reflect_like_wf(sim_9hw, pad_info):
    """
    sim_9hw: [9,H,W]
    pad_info: (top,bottom,left,right)，与 WF 使用同样 padding。
    """
    top, bottom, left, right = pad_info
    return np.pad(sim_9hw, ((0, 0), (top, bottom), (left, right)), mode="reflect").astype(np.float32)


def crop_by_pad_info(img, pad_info, scale=1):
    """
    从 padded 图像裁回原始尺寸。GT 的 pad 是 WF pad 的 scale 倍。
    """
    top, bottom, left, right = pad_info
    top *= scale
    bottom *= scale
    left *= scale
    right *= scale
    h_end = img.shape[0] - bottom if bottom > 0 else img.shape[0]
    w_end = img.shape[1] - right if right > 0 else img.shape[1]
    return img[top:h_end, left:w_end]

# ===================== 【新增】差异热力图函数 =====================
def save_heatmap(pred, gt, save_path):
    """
    计算 SR 与 GT 的绝对差异热力图。
    如果 pred / gt 尺寸不同，先把 pred resize 到 gt 尺寸，避免 broadcast 报错。
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
    diff = np.abs(pred - gt)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    diff_8u = (diff * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_8u, cv2.COLORMAP_JET)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, heatmap)
# ================================================================

def tensor_to_uint8_np(img_tensor):
    img = img_tensor.squeeze().detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def average_metric_dict(metric_dicts):
    if len(metric_dicts) == 0:
        return {"mse": None, "psnr": None, "ssim": None}
    keys = metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = [float(d[k]) for d in metric_dicts]
        out[k] = float(np.mean(vals))
    return out


def safe_copy(src, dst):
    if src is None:
        return
    if not os.path.isfile(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


# =========================================================
# 1.1 指标计算：PSNR / SSIM
# =========================================================
def compute_psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    psnr = -10.0 * np.log10(max(mse, 1e-12))
    return float(psnr), float(mse)


def compute_ssim_np(pred, target, C1=0.01**2, C2=0.03**2):
    pred_t = torch.from_numpy(pred).float()[None, None]
    target_t = torch.from_numpy(target).float()[None, None]

    mu_x = F.avg_pool2d(pred_t, 3, 1, 1)
    mu_y = F.avg_pool2d(target_t, 3, 1, 1)

    sigma_x = F.avg_pool2d(pred_t * pred_t, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target_t * target_t, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred_t * target_t, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-12
    )
    return float(ssim_map.mean().item())


def compute_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    psnr, mse = compute_psnr(pred, target)
    ssim = compute_ssim_np(pred, target)

    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }


# =========================================================
# 截取中心 200x200
# =========================================================
def crop_center_200x200(img):
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"crop_center_200x200 expects 2D grayscale image, but got shape={img.shape}")

    h, w = img.shape
    crop_h = 200
    crop_w = 200

    if h < crop_h or w < crop_w:
        raise ValueError(
            f"Image size must be at least 200x200 for center crop, but got {h}x{w}"
        )

    h_start = (h - crop_h) // 2
    w_start = (w - crop_w) // 2
    return img[h_start:h_start + crop_h, w_start:w_start + crop_w]


# =========================================================
# 每一步评估图像转 numpy 并对齐到 GT 尺寸 + 生成调试图（SR | GT | DIFF）
# =========================================================
def diffusion_tensor_to_eval_image(img_tensor, gt_shape_hw, gt_np=None, step_dir=None, step_idx=None):
    """
    img_tensor: [1,1,H,W] in [-1,1]
    gt_np: 真实图像 [0,1]
    return:
        img_vis_np: 原始尺寸 [0,1]
        img_eval_np: 与 GT 尺寸一致 [0,1]
    """
    img_vis = ((img_tensor + 1.0) / 2.0).clamp(0, 1)
    img_vis_np = img_vis.squeeze().detach().float().cpu().numpy().astype(np.float32)

    gt_h, gt_w = gt_shape_hw
    if img_vis_np.shape != (gt_h, gt_w):
        img_eval_np = cv2.resize(
            img_vis_np,
            (gt_w, gt_h),
            interpolation=cv2.INTER_CUBIC
        )
    else:
        img_eval_np = img_vis_np

    # ===================== 新增：每步生成 SR | GT | DIFF 调试图 =====================
    if gt_np is not None and step_dir is not None and step_idx is not None:
        try:
            sr = (img_eval_np * 255).astype(np.uint8)
            gt = (gt_np * 255).astype(np.uint8)
            diff = np.abs(sr.astype(np.float32) - gt.astype(np.float32))
            diff = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else np.zeros_like(sr)

            debug_img = np.hstack([sr, gt, diff])
            os.makedirs(step_dir, exist_ok=True)
            save_path = os.path.join(step_dir, f"debug_step_{step_idx:04d}.png")
            cv2.imwrite(save_path, debug_img)
        except:
            pass  # 不影响主流程
    # ============================================================================

    return img_vis_np, img_eval_np


# =========================================================
# 最优记录初始化
# =========================================================
def make_best_record():
    return {
        "value": -1e18,
        "step": None,
        "full_metrics": None,
        "center_metrics": None,
    }


def maybe_update_best(
    record,
    current_value,
    step,
    full_metrics,
    center_metrics,
    full_img_np,
    crop_img_np,
    save_dir,
    prefix,
):
    if current_value > record["value"]:
        record["value"] = float(current_value)
        record["step"] = int(step)
        record["full_metrics"] = {
            "mse": float(full_metrics["mse"]),
            "psnr": float(full_metrics["psnr"]),
            "ssim": float(full_metrics["ssim"]),
        }
        record["center_metrics"] = {
            "mse": float(center_metrics["mse"]),
            "psnr": float(center_metrics["psnr"]),
            "ssim": float(center_metrics["ssim"]),
        }

        save_float01_image(full_img_np, os.path.join(save_dir, f"{prefix}.png"))
        save_float01_image(crop_img_np, os.path.join(save_dir, f"{prefix}_center_200x200.png"))

    return record


# =========================================================
# 2. 宽场退化模型（仅用于 WF consistency）
# =========================================================
def gaussian_psf(kernel_size: int, sigma: float, device="cpu", dtype=torch.float32):
    ax = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    psf = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    psf = psf / (psf.sum() + 1e-12)
    return psf.view(1, 1, kernel_size, kernel_size)


def biosr_degradation(
    hr,
    psf_sigma=3.4638,
    psf_size=15,
    gain=1.3812,
    bias=0.08753,
    poisson_scale=400.0,
    read_noise_std=0.002608,
    downsample_scale=2,
    deterministic=True,
):
    device = hr.device
    dtype = hr.dtype
    C = hr.shape[1]

    ksize = int(psf_size)
    if ksize % 2 == 0:
        ksize += 1

    psf = gaussian_psf(ksize, psf_sigma, device=device, dtype=dtype)
    psf = psf.repeat(C, 1, 1, 1)

    x = F.conv2d(hr, psf, padding=ksize // 2, groups=C)
    x = gain * x + bias
    x = torch.clamp(x, 0.0, 1.0)

    if int(downsample_scale) > 1:
        x = F.avg_pool2d(
            x,
            kernel_size=int(downsample_scale),
            stride=int(downsample_scale),
        )

    if deterministic:
        return x

    photons = x * float(poisson_scale)
    photons = torch.clamp(photons, min=0.0)
    photons_noisy = torch.poisson(photons)
    x = photons_noisy / max(float(poisson_scale), 1e-8)

    if float(read_noise_std) > 0:
        x = x + torch.randn_like(x) * float(read_noise_std)

    x = torch.clamp(x, 0.0, 1.0)
    return x


# =========================================================
# 3. Text model
# =========================================================
def load_text_model(args, device):
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path)
    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder_path).to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    return tokenizer, text_encoder


@th.no_grad()
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
# 4.1 宽场引导函数
# =========================================================
def general_cond_fn_wf(x, t, x_lr=None, args=None):
    assert x_lr is not None and args is not None
    with th.enable_grad():
        x_t = x.detach().requires_grad_(True)
        x_t = torch.clamp(x_t, -1.0, 1.0)

        x_t_norm = (x_t + 1.0) / 2.0
        x_lr_norm = ((x_lr + 1.0) / 2.0).detach()

        x_t_deg = biosr_degradation(
            x_t_norm,
            psf_sigma=args.psf_sigma,
            psf_size=args.psf_size,
            gain=args.gain,
            bias=args.bias,
            poisson_scale=args.poisson_scale,
            read_noise_std=args.read_noise_std,
            downsample_scale=args.downsample_scale,
            deterministic=True,
        )

        if x_t_deg.shape[-2:] != x_lr_norm.shape[-2:]:
            x_t_deg = F.interpolate(
                x_t_deg,
                size=x_lr_norm.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        residual = x_t_deg - x_lr_norm
        charbonnier = torch.sqrt(residual ** 2 + 1e-4)
        data_loss = charbonnier.mean()

        tv_h = torch.abs(x_t_norm[:, :, 1:, :] - x_t_norm[:, :, :-1, :]).mean()
        tv_w = torch.abs(x_t_norm[:, :, :, 1:] - x_t_norm[:, :, :, :-1]).mean()
        tv_loss = tv_h + tv_w

        # ----------------- SSIM 损失 -----------------
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu1 = F.avg_pool2d(x_t_deg, 3, 1, 1)
        mu2 = F.avg_pool2d(x_lr_norm, 3, 1, 1)
        sigma1 = F.avg_pool2d(x_t_deg ** 2, 3, 1, 1) - mu1 ** 2
        sigma2 = F.avg_pool2d(x_lr_norm ** 2, 3, 1, 1) - mu2 ** 2
        sigma12 = F.avg_pool2d(x_t_deg * x_lr_norm, 3, 1, 1) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2) + 1e-12)
        ssim_loss = 1.0 - ssim_map.mean()

        current_t_val = int(t[0].item())
        if current_t_val > 900:
            anneal = 10.0
        elif current_t_val > 800:
            anneal = 8.0
        elif current_t_val > 400:
            anneal = 4.0
        else:
            anneal = 2.0

        # ----------------- 总损失 -----------------
        ssim_weight = 0.5
        loss = -data_loss * args.img_guidance_scale * anneal \
               - tv_loss * args.tv_weight \
               - ssim_loss * ssim_weight

        grad = th.autograd.grad(loss, x_t)[0]

        

        return grad, data_loss.item(), tv_loss.item(), loss.item()


def general_cond_fn_sim(x, t, sim_lr=None, sim_forward_model=None, args=None):
    assert sim_lr is not None and sim_forward_model is not None and args is not None

    with th.enable_grad():
        x_t = x.detach().requires_grad_(True)
        x_t = torch.clamp(x_t, -1.0, 1.0)

        x_t_norm = (x_t + 1.0) / 2.0
        sim_lr_norm = ((sim_lr + 1.0) / 2.0).detach()

        x_t_deg_sim = sim_forward_model(x_t_norm)   # [B,9,H,W]
        current_t_val = int(t[0].item())


        if x_t_deg_sim.shape[-2:] != sim_lr_norm.shape[-2:]:
            x_t_deg_sim = F.interpolate(
                x_t_deg_sim,
                size=sim_lr_norm.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if current_t_val % 100 == 0:
            # 创建保存目录
            #sim_pred_dir = os.path.join("./output8", "sim_pred_steps")
            sim_pred_dir = os.path.join(args.save_dir, "sim_pred_steps")
            os.makedirs(sim_pred_dir, exist_ok=True)

            # 获取预测值和真实值（均为 [B,9,H,W]，值域 [0,1]）
            pred_sim_np = x_t_deg_sim.detach().squeeze(0).cpu().numpy()   # [9, H, W]
            gt_sim_np = sim_lr_norm.detach().squeeze(0).cpu().numpy()     # [9, H, W]

            for ch in range(pred_sim_np.shape[0]):
                pred = pred_sim_np[ch]          # [H, W], 0~1
                gt   = gt_sim_np[ch]            # [H, W], 0~1

                # 归一化差异图并转为热力图 (Jet)
                diff = np.abs(pred - gt)
                diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                diff_heat = (diff_norm * 255).astype(np.uint8)
                diff_heat = cv2.applyColorMap(diff_heat, cv2.COLORMAP_JET)

                # 将 Pred 和 GT 转为 3 通道灰度图
                pred_8u = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
                gt_8u   = (np.clip(gt, 0, 1) * 255).astype(np.uint8)
                pred_color = cv2.cvtColor(pred_8u, cv2.COLOR_GRAY2BGR)
                gt_color   = cv2.cvtColor(gt_8u,   cv2.COLOR_GRAY2BGR)

                # 水平拼接 [Pred] [GT] [Diff]
                h_img = np.hstack([pred_color, gt_color, diff_heat])

                # 在左上角添加文字标注（每个子图单独加）
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(h_img, "sr_deg", (10, 30), font, 0.8, (0, 255, 0), 2)
                cv2.putText(h_img, "lr_obs",   (pred_color.shape[1] + 10, 30), font, 0.8, (0, 255, 0), 2)
                cv2.putText(h_img, "Diff", (pred_color.shape[1] + gt_color.shape[1] + 10, 30), font, 0.8, (0, 255, 0), 2)

                # 保存拼接图
                save_path = os.path.join(sim_pred_dir, f"step_t_{current_t_val:04d}_ch_{ch+1:02d}.png")
                cv2.imwrite(save_path, h_img)
        # ===================== 最强细节损失：纯Torch高通滤波 =====================
        pred = x_t_deg_sim
        target = sim_lr_norm

        # 🔥 1. 高斯模糊（纯torch实现，替代kornia）
        def gaussian_blur(x, kernel_size=3, sigma=1.0):
            C = x.shape[1]
            # 创建高斯核
            k = kernel_size
            x_cord = torch.arange(k, device=x.device)
            x_grid = x_cord.repeat(k).view(k, k)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
            mean = (k - 1) / 2.
            var = sigma ** 2.
            gaussian_kernel = (1. / (2. * th.pi * var)) * th.exp(
                -th.sum((xy_grid - mean) ** 2., dim=-1) / (2 * var)
            )
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            gaussian_kernel = gaussian_kernel.view(1, 1, k, k).repeat(C, 1, 1, 1)
            
            # 卷积
            padding = k // 2
            blurred = F.conv2d(x, gaussian_kernel, padding=padding, groups=C)
            return blurred

        # 🔥 2. 提取高频细节 = 原图 - 模糊图（最关键！）
        pred_blur = gaussian_blur(pred, kernel_size=3, sigma=0.8)
        target_blur = gaussian_blur(target, kernel_size=3, sigma=0.8)
        
        pred_high = pred - pred_blur  # 只保留边缘/纹理/细节
        target_high = target - target_blur

        # 🔥 3. 细节对齐损失（扩散模型能直接感知）
        detail_residual = pred_high - target_high
        detail_loss = torch.sqrt(detail_residual ** 2 + 1e-6).mean()

        # 频域加强（可选，让细节更锐利）
        fft_pred = torch.fft.fft2(pred, norm='ortho')
        fft_target = torch.fft.fft2(target, norm='ortho')
        freq_loss = torch.abs(fft_pred - fft_target).mean()

        # 总损失：只拉细节！！！
        sim_data_loss = 2.0 * detail_loss + 1.3 * freq_loss
        # ======================================================================

        # 你的退火系数不变
        current_t_val = int(t[0].item())
        if current_t_val > 900:
            anneal = 10.0
        elif current_t_val > 800:
            anneal = 8.0
        elif current_t_val > 400:
            anneal = 4.0
        else:
            anneal = 2.0

        # 引导强度翻倍，细节更明显
        loss = -sim_data_loss * args.sim_img_guidance_scale * anneal 
        grad = th.autograd.grad(loss, x_t)[0]
        return grad, sim_data_loss.item(), loss.item()

# =========================================================
# 4.3 总引导函数
# =========================================================
def total_cond_fn(x, t, y=None, **kwargs):
    args = kwargs["args"]
    x_lr = kwargs["x_lr"]
    sim_lr = kwargs["sim_lr"]
    sim_forward_model = kwargs["sim_forward_model"]

    grad_wf, wf_data, wf_tv, wf_loss = general_cond_fn_wf(
        x, t, x_lr=x_lr, args=args
    )

    grad_sim, sim_data, sim_loss = general_cond_fn_sim(
        x, t, sim_lr=sim_lr, sim_forward_model=sim_forward_model, args=args
    )

    grad_total = args.wf_guidance_weight * grad_wf + args.sim_guidance_weight * grad_sim
    t_val = int(t[0].item())

    if args.cond_log_interval > 0:
        if (t_val % args.cond_log_interval == 0) or (t_val == args.denoise_steps - 1) or (t_val == 0):
            print(
                f"[Step t={t_val:4d}] | "
                f"WF(data):{wf_data:.4f} | "
                f"SIM(data):{sim_data:.4f} | "
                f"WF_loss:{wf_loss:.4f} | "
                f"SIM_loss:{sim_loss:.4f}"
            )

    if args.save_per_step_images:
        step_dir = kwargs.get("step_dir", None)
        if step_dir is not None:
            os.makedirs(step_dir, exist_ok=True)
            x_norm = (x + 1) / 2
            x_img = tensor_to_uint8_np(x_norm)
            
            # ========== 拼接：SR | GT | DIFF | GRAD_WF | GRAD_SIM | GRAD_TOTAL ==========
            gt_np = kwargs.get("gt_np", None)
            if gt_np is not None:
                sr = x_img
                gt = (gt_np * 255).astype(np.uint8)
                h, w = sr.shape

                # 1. 差异图
                diff = np.abs(sr.astype(np.float32) - gt.astype(np.float32))
                diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                diff_8u = (diff * 255).astype(np.uint8)

                # ----------------------------------------------------
                # 工具函数：把梯度转成 红蓝配色图 + 左上角绿色标注
                # ----------------------------------------------------
                def grad_to_color(grad, label_text):
                    grad_np = grad.detach().cpu().squeeze().numpy()
                    g_max = max(np.abs(grad_np).max(), 1e-8)
                    grad_norm = grad_np / g_max
                    color = np.zeros((h, w, 3), dtype=np.uint8)
                    mask_neg = grad_norm < 0
                    color[mask_neg, 0] = np.uint8(-grad_norm[mask_neg] * 255)
                    mask_pos = grad_norm > 0
                    color[mask_pos, 2] = np.uint8(grad_norm[mask_pos] * 255)
                    
                    # ✅ 左上角绿色标注
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(color, label_text, (10, 30), font, 1.0, (0,255,0), 2, cv2.LINE_AA)
                    return color

                # 2. 三个梯度图（带标注）
                grad_wf_color = grad_to_color(grad_wf, "SR_Deg vs WF")
                grad_sim_color = grad_to_color(grad_sim, "SR_Deg vs SIM")
                grad_total_color = grad_to_color(grad_total, "Total loss")
                
                # grad_wf_color = add_label(grad_wf_color, "WF grad (data+tv)")
                # grad_sim_color = add_label(grad_sim_color, "SIM grad (detail+freq)")
                # grad_total_color = add_label(grad_total_color, "Total grad")
                # 3. 全部转 3 通道
                sr_color = cv2.cvtColor(sr, cv2.COLOR_GRAY2BGR)
                gt_color = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
                diff_color = cv2.cvtColor(diff_8u, cv2.COLOR_GRAY2BGR)
                
                # 间隔条（3通道）
                gap = np.full((h, 10), 127, dtype=np.uint8)
                gap_color = cv2.cvtColor(gap, cv2.COLOR_GRAY2BGR)

                # ----------------------------------------------------
                # 最终拼接：6 张图
                # SR | GT | DIFF | GRAD_WF | GRAD_SIM | GRAD_TOTAL
                # ----------------------------------------------------
                concat = np.hstack([
                    sr_color, gap_color,
                    gt_color, gap_color,
                    diff_color, gap_color,
                    grad_wf_color, gap_color,
                    grad_sim_color, gap_color,
                    grad_total_color
                ])

                cv2.imwrite(os.path.join(step_dir, f"step_{t_val:04d}.png"), concat)
                #print(os.path.join(step_dir, f"step_{t_val:04d}.png"))
            else:
                Image.fromarray(x_img).save(os.path.join(step_dir, f"step_{t_val:04d}.png"))

    return grad_total


# =========================================================
# 5. 模型加载
# =========================================================
def create_your_unet_and_diffusion(args, device):
    if not os.path.isdir(args.unet_model_path):
        raise FileNotFoundError(f"unet_model_path does not exist: {args.unet_model_path}")

    unet_subdir = os.path.join(args.unet_model_path, "unet_mix")
    if not os.path.isdir(unet_subdir):
        raise FileNotFoundError(f"Missing unet subfolder: {unet_subdir}")

    unet = UNet2DConditionModel.from_pretrained(
        args.unet_model_path,
        subfolder="unet_mix",
    ).to(device)

    logger.log(f"✅ 加载模型成功: {unet_subdir}")
    if args.use_fp16:
        unet = unet.half()

    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    return unet, diffusion


# =========================================================
# 6. 更稳健的 SIM 数据读取
#    同时支持：
#    A. sim_root/001.tif        (一个 tif 里面有 9 帧)
#    B. sim_root/001/*.tif      (一个文件夹里面 9 张 tif)
# =========================================================
def sim_array_to_9hw(arr, expected_frames=9):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        raise ValueError(
            f"SIM file only contains a single 2D image with shape={arr.shape}. "
            f"Expected {expected_frames} frames for SIM."
        )

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            return sim_array_to_9hw(arr[0], expected_frames)
        if arr.shape[-1] == 1:
            return sim_array_to_9hw(arr[..., 0], expected_frames)
        raise ValueError(f"Unsupported 4D SIM shape: {arr.shape}")

    if arr.ndim != 3:
        raise ValueError(f"Unsupported SIM array shape: {arr.shape}")

    # 优先认为第一页维是帧数
    if arr.shape[0] == expected_frames:
        sim = arr
    elif arr.shape[-1] == expected_frames:
        sim = np.transpose(arr, (2, 0, 1))
    elif arr.shape[1] == expected_frames:
        sim = np.transpose(arr, (1, 0, 2))
    else:
        raise ValueError(
            f"Cannot interpret SIM stack shape={arr.shape} as {expected_frames} frames."
        )

    return sim


def load_sim_from_folder(sim_dir, expected_frames=9):
    tif_files = list_tif_files(sim_dir)
    if len(tif_files) != expected_frames:
        raise ValueError(
            f"SIM folder must contain exactly {expected_frames} tif files, "
            f"but got {len(tif_files)} in {sim_dir}"
        )

    frames = []
    for p in tif_files:
        im = read_single_image_as_float01(p)
        frames.append(im)
    sim = np.stack(frames, axis=0)  # [9,H,W]
    return sim


def load_sim_from_file(sim_file, expected_frames=9):
    arr = tiff.imread(sim_file)
    sim = sim_array_to_9hw(arr, expected_frames=expected_frames)  # [9,H,W]
    sim = np.stack([normalize_to_float01(frame) for frame in sim], axis=0)
    return sim


def load_sim_input_as_tensor(sim_path, expected_frames=9):
    if os.path.isdir(sim_path):
        sim = load_sim_from_folder(sim_path, expected_frames=expected_frames)
    elif os.path.isfile(sim_path):
        sim = load_sim_from_file(sim_path, expected_frames=expected_frames)
        print("222",sim.shape)
    else:
        raise FileNotFoundError(f"SIM input does not exist: {sim_path}")

    sim = sim[None, ...]            # [1,9,H,W]
    sim_norm = sim * 2.0 - 1.0
    return torch.from_numpy(sim_norm.astype(np.float32))


# =========================================================
# 7. 路径解析
# =========================================================
def resolve_case_image_path(root, stem):
    """
    解析单张图像路径：
    1) root/stem.tif
    2) root/stem.tiff
    3) root/*/stem.tif
    4) root/*/stem.tiff
    """
    direct_candidates = [
        os.path.join(root, stem + ".tif"),
        os.path.join(root, stem + ".tiff"),
    ]
    for p in direct_candidates:
        if os.path.isfile(p):
            return p

    recursive_candidates = []
    recursive_candidates.extend(glob.glob(os.path.join(root, "*", stem + ".tif")))
    recursive_candidates.extend(glob.glob(os.path.join(root, "*", stem + ".tiff")))
    recursive_candidates = sorted(recursive_candidates, key=natural_key)

    if len(recursive_candidates) > 0:
        return recursive_candidates[0]

    return None


def resolve_case_sim_input(root, stem):
    """
    解析 SIM 输入路径：
    1) root/stem.tif
    2) root/stem.tiff
    3) root/stem/           (目录里 9 张 tif)
    4) root/*/stem.tif
    5) root/*/stem.tiff
    6) root/*/stem/         (上级还有 level_xx 等)
    """
    direct_candidates = [
        os.path.join(root, stem + ".tif"),
        os.path.join(root, stem + ".tiff"),
        os.path.join(root, stem),
    ]
    for p in direct_candidates:
        if os.path.isfile(p) or os.path.isdir(p):
            return p

    recursive_candidates = []
    recursive_candidates.extend(glob.glob(os.path.join(root, "*", stem + ".tif")))
    recursive_candidates.extend(glob.glob(os.path.join(root, "*", stem + ".tiff")))
    recursive_candidates.extend(glob.glob(os.path.join(root, "*", stem)))
    recursive_candidates = [
        p for p in recursive_candidates if os.path.isfile(p) or os.path.isdir(p)
    ]
    recursive_candidates = sorted(recursive_candidates, key=natural_key)

    if len(recursive_candidates) > 0:
        return recursive_candidates[0]

    return None


# =========================================================
# 8. 单图重建
# =========================================================
def reconstruct_one_case(
    case_name,
    wf_path,
    sim_path,
    gt_path,
    case_save_dir,
    model,
    diffusion,
    encoder_hidden_states,
    sim_forward_model,
    device,
    args,
):
    """
    整幅图输入版本：
    WF:  502x502       -> pad 到 512x512 -> 128x128 overlap tiles
    SIM: 9x502x502     -> 同步 pad 到 9x512x512 -> 9x128x128 overlap tiles
    GT:  1004x1004     -> pad 到 1024x1024 -> 用于 tile debug / 最终指标

    每个 tile:
        WF/SIM tile: 128x128
        diffusion output: 256x256
    最后：
        25 个 HR tile 用 Hanning blending 融合成 1024x1024，再 crop 回 1004x1004。
    """
    os.makedirs(case_save_dir, exist_ok=True)

    # =========================================================
    # 1. 读取整幅 WF / SIM / GT
    # =========================================================
    wf = read_single_image_as_float01(wf_path).astype(np.float32)
    sim_np = load_sim_input_as_tensor(
        sim_path,
        expected_frames=args.sim_n_angles * args.sim_n_phases
    ).squeeze(0).cpu().numpy()  # [-1,1], [9,H,W]
    sim_np = ((sim_np + 1.0) / 2.0).astype(np.float32)  # [0,1], [9,H,W]
    gt = read_single_image_as_float01(gt_path).astype(np.float32)

    print("111", wf.shape)
    print("222", sim_np.shape)
    print("333", gt.shape)

    save_float01_image(wf, os.path.join(case_save_dir, "wf_obs_raw.png"))
    save_float01_image(gt, os.path.join(case_save_dir, "gt.png"))
    gt_crop = crop_center_200x200(gt)
    save_float01_image(gt_crop, os.path.join(case_save_dir, "gt_center_200x200.png"))

    if sim_np.ndim != 3:
        raise ValueError(f"SIM should be [9,H,W], but got {sim_np.shape}")
    if sim_np.shape[1:] != wf.shape:
        raise ValueError(f"SIM/WF size mismatch: SIM={sim_np.shape}, WF={wf.shape}")
    if gt.shape != (wf.shape[0] * 2, wf.shape[1] * 2):
        raise ValueError(f"GT should be 2x WF. WF={wf.shape}, GT={gt.shape}")

    if args.save_observation_mean:
        sim_mean_np = sim_np.mean(axis=0)
        save_float01_image(sim_mean_np, os.path.join(case_save_dir, "sim_mean.png"))

    logger.log(f"🎯 开始重建: {case_name}")
    logger.log(f"    WF : {wf_path}")
    logger.log(f"    SIM: {sim_path}")
    logger.log(f"    GT : {gt_path}")

    # =========================================================
    # 2. padding + tile 参数
    # =========================================================
    tile_size_lr = int(getattr(args, "tile_size_lr", 128))
    stride_lr = int(getattr(args, "tile_stride_lr", 96))
    scale = 2
    tile_size_hr = tile_size_lr * scale
    stride_hr = stride_lr * scale

    if args.hr_size != tile_size_hr:
        logger.log(
            f"⚠️ args.hr_size={args.hr_size}, but tile_size_hr={tile_size_hr}. "
            f"建议保持 args.hr_size={tile_size_hr}."
        )

    # 对 502x502 -> 512x512；对其他尺寸则 pad 到 tile_size_lr 的整数倍
    wf_padded, wf_pad_info = pad_2d_reflect_to_multiple(
        wf, multiple=tile_size_lr, pad_each_side_min=5
    )
    sim_padded = pad_sim_reflect_like_wf(sim_np, wf_pad_info)

    gt_pad_info = tuple(v * scale for v in wf_pad_info)
    gt_padded = np.pad(
        gt,
        ((gt_pad_info[0], gt_pad_info[1]), (gt_pad_info[2], gt_pad_info[3])),
        mode="reflect"
    ).astype(np.float32)

    save_float01_image(wf_padded, os.path.join(case_save_dir, "wf_padded.png"))
    save_float01_image(gt_padded, os.path.join(case_save_dir, "gt_padded.png"))

    lr_h, lr_w = wf_padded.shape
    hr_h, hr_w = gt_padded.shape
    if (hr_h, hr_w) != (lr_h * scale, lr_w * scale):
        raise ValueError(
            f"Padded WF/GT size mismatch: WF_padded={wf_padded.shape}, "
            f"GT_padded={gt_padded.shape}, expected {(lr_h * scale, lr_w * scale)}"
        )

    ys = get_tile_starts(lr_h, tile_size_lr, stride_lr)
    xs = get_tile_starts(lr_w, tile_size_lr, stride_lr)

    logger.log(
        f"    padded WF={wf_padded.shape}, SIM={sim_padded.shape}, GT={gt_padded.shape} | "
        f"tile={tile_size_lr}, stride={stride_lr}, "
        f"overlap_lr={tile_size_lr - stride_lr}, overlap_hr={tile_size_hr - stride_hr}, "
        f"num_tiles={len(ys)}x{len(xs)}={len(ys) * len(xs)}"
    )

    # =========================================================
    # 3. 初始化融合容器
    # =========================================================
    merged_eval = np.zeros_like(gt_padded, dtype=np.float32)
    merged_sample = np.zeros_like(gt_padded, dtype=np.float32)
    weight_map = np.zeros_like(gt_padded, dtype=np.float32)
    hanning_weight = make_hanning_weight(tile_size_hr, tile_size_hr, eps=1e-3)

    save_float01_image(
        hanning_weight / (hanning_weight.max() + 1e-8),
        os.path.join(case_save_dir, "hanning_weight_256.png")
    )

    c_time = time.time()
    model_kwargs = {"encoder_hidden_states": encoder_hidden_states}
    tile_summaries = []

    # =========================================================
    # 4. overlap tile 推理
    # =========================================================
    tile_idx = 0
    for y in ys:
        for x in xs:
            tile_dir = os.path.join(case_save_dir, f"tile_{tile_idx:02d}_y{y}_x{x}")
            os.makedirs(tile_dir, exist_ok=True)

            wf_tile = wf_padded[y:y + tile_size_lr, x:x + tile_size_lr]
            sim_tile = sim_padded[:, y:y + tile_size_lr, x:x + tile_size_lr]

            hy = y * scale
            hx = x * scale
            gt_tile = gt_padded[hy:hy + tile_size_hr, hx:hx + tile_size_hr]

            if wf_tile.shape != (tile_size_lr, tile_size_lr):
                raise ValueError(f"Invalid WF tile shape: {wf_tile.shape} at y={y}, x={x}")
            if sim_tile.shape != (args.sim_n_angles * args.sim_n_phases, tile_size_lr, tile_size_lr):
                raise ValueError(f"Invalid SIM tile shape: {sim_tile.shape} at y={y}, x={x}")
            if gt_tile.shape != (tile_size_hr, tile_size_hr):
                raise ValueError(f"Invalid GT tile shape: {gt_tile.shape} at hy={hy}, hx={hx}")

            x_lr_tile = torch.from_numpy((wf_tile * 2.0 - 1.0)[None, None].astype(np.float32)).to(device)
            sim_lr_tile = torch.from_numpy((sim_tile * 2.0 - 1.0)[None].astype(np.float32)).to(device)

            # 保存 tile 输入，便于核对是否切块正确
            save_float01_image(wf_tile, os.path.join(tile_dir, "wf_tile.png"))
            save_float01_image(gt_tile, os.path.join(tile_dir, "gt_tile.png"))
            if args.save_observation_mean:
                save_float01_image(sim_tile.mean(axis=0), os.path.join(tile_dir, "sim_tile_mean.png"))

            shape = (1, 1, args.hr_size, args.hr_size)

            # 确定性 noise：同一张图每个 tile 可复现，但不同 tile 仍不同
            tile_seed = int(getattr(args, "tile_noise_seed", 1234)) + tile_idx
            gen = torch.Generator(device=device)
            gen.manual_seed(tile_seed)
            noise = torch.randn(
                shape,
                generator=gen,
                device=device,
                dtype=torch.float16 if args.use_fp16 else torch.float32,
            )

            step_dir = os.path.join(tile_dir, "steps") if args.save_per_step_images else None

            progressive_samples = diffusion.p_sample_loop_progressive(
                lambda x_in, t, **kwargs: model(
                    x_in, t, encoder_hidden_states=encoder_hidden_states
                ).sample,
                shape,
                noise=noise,
                model_kwargs=model_kwargs,
                cond_fn=lambda x_in, t, y=None, **kwargs: total_cond_fn(
                    x_in,
                    t,
                    x_lr=x_lr_tile,
                    sim_lr=sim_lr_tile,
                    sim_forward_model=sim_forward_model,
                    args=args,
                    step_dir=step_dir,
                    gt_np=gt_tile,   # ✅ 关键：传入 256x256 的 GT tile，而不是 1004x1004 整图
                ),
                device=device,
                denoise_steps=args.denoise_steps,
                clip_denoised=args.clip_denoised,
            )

            final_sample_tile = None
            final_eval_tile = None
            for out in progressive_samples:
                if isinstance(out, dict):
                    final_sample_tile = out["sample"] if "sample" in out else out
                    final_eval_tile = out["pred_xstart"] if "pred_xstart" in out else final_sample_tile
                else:
                    final_sample_tile = out
                    final_eval_tile = out

            if final_sample_tile is None or final_eval_tile is None:
                raise RuntimeError(f"Sampling failed at tile {tile_idx}, y={y}, x={x}")

            sample_tile_np, _ = diffusion_tensor_to_eval_image(final_sample_tile, gt_tile.shape)
            eval_tile_np, _ = diffusion_tensor_to_eval_image(final_eval_tile, gt_tile.shape)

            sample_tile_np = np.asarray(sample_tile_np, dtype=np.float32)
            eval_tile_np = np.asarray(eval_tile_np, dtype=np.float32)
            if sample_tile_np.shape != (tile_size_hr, tile_size_hr):
                sample_tile_np = cv2.resize(sample_tile_np, (tile_size_hr, tile_size_hr), interpolation=cv2.INTER_CUBIC)
            if eval_tile_np.shape != (tile_size_hr, tile_size_hr):
                eval_tile_np = cv2.resize(eval_tile_np, (tile_size_hr, tile_size_hr), interpolation=cv2.INTER_CUBIC)

            save_float01_image(sample_tile_np, os.path.join(tile_dir, "tile_sample.png"))
            save_float01_image(eval_tile_np, os.path.join(tile_dir, "tile_eval.png"))

            # tile 局部指标，仅用于检查；最终指标仍然用融合后的整图计算
            tile_metrics = compute_metrics(eval_tile_np, gt_tile)
            tile_summaries.append({
                "tile_idx": int(tile_idx),
                "y": int(y),
                "x": int(x),
                "hy": int(hy),
                "hx": int(hx),
                "psnr": float(tile_metrics["psnr"]),
                "ssim": float(tile_metrics["ssim"]),
                "mse": float(tile_metrics["mse"]),
            })

            y1, x1 = hy, hx
            y2, x2 = hy + tile_size_hr, hx + tile_size_hr
            merged_sample[y1:y2, x1:x2] += sample_tile_np * hanning_weight
            merged_eval[y1:y2, x1:x2] += eval_tile_np * hanning_weight
            weight_map[y1:y2, x1:x2] += hanning_weight

            logger.log(
                f"[{case_name}] tile {tile_idx + 1:02d}/{len(ys) * len(xs):02d} "
                f"LR(y={y},x={x}) HR(y={hy},x={hx}) "
                f"tile_PSNR={tile_metrics['psnr']:.4f}, tile_SSIM={tile_metrics['ssim']:.4f}"
            )
            tile_idx += 1

    # =========================================================
    # 5. 融合 + crop 回原始尺寸
    # =========================================================
    if np.any(weight_map <= 0):
        zero_count = int(np.sum(weight_map <= 0))
        raise RuntimeError(f"weight_map has {zero_count} zero pixels. Tiling did not cover full image.")

    merged_sample = np.clip(merged_sample / (weight_map + 1e-8), 0.0, 1.0).astype(np.float32)
    merged_eval = np.clip(merged_eval / (weight_map + 1e-8), 0.0, 1.0).astype(np.float32)

    save_float01_image(merged_sample, os.path.join(case_save_dir, "result_sample_padded_blend.png"))
    save_float01_image(merged_eval, os.path.join(case_save_dir, "result_eval_padded_blend.png"))
    save_float01_image(weight_map / (weight_map.max() + 1e-8), os.path.join(case_save_dir, "weight_map.png"))

    final_sample_np = crop_by_pad_info(merged_sample, wf_pad_info, scale=scale)
    final_eval_np = crop_by_pad_info(merged_eval, wf_pad_info, scale=scale)

    if final_eval_np.shape != gt.shape:
        raise ValueError(f"Final reconstruction shape mismatch: final={final_eval_np.shape}, GT={gt.shape}")

    # =========================================================
    # 6. 保存最终结果 + 指标
    # =========================================================
    total_time = time.time() - c_time
    logger.log(f"✅ {case_name} tile sampling 完成，总耗时: {total_time:.2f}s")

    save_float01_image(final_sample_np, os.path.join(case_save_dir, "result_sample.png"))
    save_float01_image(final_eval_np, os.path.join(case_save_dir, "result_eval_final.png"))

    final_eval_crop_np = crop_center_200x200(final_eval_np)
    save_float01_image(final_eval_crop_np, os.path.join(case_save_dir, "result_eval_final_center_200x200.png"))

    save_heatmap(
        pred=final_sample_np,
        gt=gt,
        save_path=os.path.join(case_save_dir, "heatmap_sample_vs_gt.png")
    )
    save_heatmap(
        pred=final_eval_np,
        gt=gt,
        save_path=os.path.join(case_save_dir, "heatmap_eval_vs_gt.png")
    )

    if args.save_numpy_array:
        np.savez(
            os.path.join(case_save_dir, "result.npz"),
            sample=final_sample_np,
            eval=final_eval_np,
            weight_map=weight_map,
        )

    # 最终 SIM 预测：SimpleStableSIM 如果固定 image_size=256，整图 forward 可能不兼容；这里安全处理。
    try:
        with torch.no_grad():
            final_eval_tensor = torch.from_numpy(final_eval_np[None, None].astype(np.float32)).to(device)
            pred_sim = sim_forward_model(final_eval_tensor)
            pred_sim_np = pred_sim.squeeze(0).detach().cpu().numpy()

            pred_sim_dir = os.path.join(case_save_dir, "pred_sim_9ch")
            os.makedirs(pred_sim_dir, exist_ok=True)
            for i in range(pred_sim_np.shape[0]):
                save_float01_image(pred_sim_np[i], os.path.join(pred_sim_dir, f"{i+1}.png"))
    except Exception as e:
        logger.log(f"⚠️ 跳过整图 pred_sim_9ch 保存：{e}")

    final_metrics_full = compute_metrics(final_eval_np, gt)
    final_metrics_crop = compute_metrics(final_eval_crop_np, gt_crop)

    per_step_metrics = [{
        "step": 1,
        "full_image": {
            "mse": float(final_metrics_full["mse"]),
            "psnr": float(final_metrics_full["psnr"]),
            "ssim": float(final_metrics_full["ssim"]),
        },
        "center_200x200": {
            "mse": float(final_metrics_crop["mse"]),
            "psnr": float(final_metrics_crop["psnr"]),
            "ssim": float(final_metrics_crop["ssim"]),
        },
    }]

    best_records = {
        "best_full_psnr": make_best_record(),
        "best_full_ssim": make_best_record(),
        "best_center_psnr": make_best_record(),
        "best_center_ssim": make_best_record(),
    }
    best_dir = os.path.join(case_save_dir, "best_step_results")
    os.makedirs(best_dir, exist_ok=True)
    current_crop_np = final_eval_crop_np
    for key, value in [
        ("best_full_psnr", final_metrics_full["psnr"]),
        ("best_full_ssim", final_metrics_full["ssim"]),
        ("best_center_psnr", final_metrics_crop["psnr"]),
        ("best_center_ssim", final_metrics_crop["ssim"]),
    ]:
        best_records[key] = maybe_update_best(
            best_records[key],
            value,
            1,
            final_metrics_full,
            final_metrics_crop,
            final_eval_np,
            current_crop_np,
            best_dir,
            key,
        )

    with open(os.path.join(case_save_dir, "tile_summary.json"), "w") as f:
        json.dump(tile_summaries, f, indent=2)

    case_summary = {
        "case_name": case_name,
        "wf_path": wf_path,
        "sim_path": sim_path,
        "gt_path": gt_path,
        "case_save_dir": case_save_dir,
        "runtime_sec": float(total_time),
        "tile_inference": {
            "tile_size_lr": tile_size_lr,
            "stride_lr": stride_lr,
            "overlap_lr": tile_size_lr - stride_lr,
            "tile_size_hr": tile_size_hr,
            "stride_hr": stride_hr,
            "overlap_hr": tile_size_hr - stride_hr,
            "num_tiles_y": len(ys),
            "num_tiles_x": len(xs),
            "num_tiles_total": len(ys) * len(xs),
            "wf_pad_info": list(map(int, wf_pad_info)),
            "gt_pad_info": list(map(int, gt_pad_info)),
            "blend": "hanning_weighted_overlap_add",
        },
        "final_step": {
            "full_image": final_metrics_full,
            "center_200x200": final_metrics_crop,
        },
        "best_over_all_steps": best_records,
        "eval_image_source": "overlap tile inference with Hanning blending",
        "num_evaluated_steps": len(per_step_metrics),
    }

    with open(os.path.join(case_save_dir, "final_metrics.json"), "w") as f:
        json.dump(case_summary, f, indent=2)
    with open(os.path.join(case_save_dir, "per_step_metrics.json"), "w") as f:
        json.dump(per_step_metrics, f, indent=2)

    logger.log(
        f"[{case_name}] Final Full: PSNR={final_metrics_full['psnr']:.4f}, "
        f"SSIM={final_metrics_full['ssim']:.4f}"
    )
    logger.log(
        f"[{case_name}] Final Center200: PSNR={final_metrics_crop['psnr']:.4f}, "
        f"SSIM={final_metrics_crop['ssim']:.4f}"
    )

    return case_summary


# =========================================================
# 9. 构建样本列表
# =========================================================
def build_case_list(args):
    wf_files = list_tif_files(args.wf_root)
    
    print(wf_files)
    if len(wf_files) == 0:
        raise FileNotFoundError(f"No tif files found in wf_root: {args.wf_root}")

    cases = []
    for wf_path in wf_files:
        stem = os.path.splitext(os.path.basename(wf_path))[0]

        sim_path = resolve_case_sim_input(args.sim_root, stem)

        gt_path = resolve_case_image_path(args.gt_root, stem)

        if sim_path is None:
            logger.log(f"⚠️ 跳过 {stem}: 未找到对应的 SIM 输入")
            continue
        if gt_path is None:
            logger.log(f"⚠️ 跳过 {stem}: 未找到对应的 GT 图像")
            continue

        cases.append({
            "case_name": stem,
            "wf_path": wf_path,
            "sim_path": sim_path,
            "gt_path": gt_path,
        })

        if len(cases) >= args.num_cases:
            break

    if len(cases) == 0:
        raise RuntimeError("No valid matched cases found.")
    if len(cases) < args.num_cases:
        logger.log(f"⚠️ 仅找到 {len(cases)} 个有效样本，少于请求的 {args.num_cases} 个。")

    return cases


# =========================================================
# 10. 汇总保存
# =========================================================
def save_dataset_summary(all_case_summaries, save_dir):
    final_full_list = [x["final_step"]["full_image"] for x in all_case_summaries]
    final_center_list = [x["final_step"]["center_200x200"] for x in all_case_summaries]

    best_full_psnr_list = [x["best_over_all_steps"]["best_full_psnr"]["full_metrics"] for x in all_case_summaries]
    best_full_ssim_list = [x["best_over_all_steps"]["best_full_ssim"]["full_metrics"] for x in all_case_summaries]
    best_center_psnr_list = [x["best_over_all_steps"]["best_center_psnr"]["center_metrics"] for x in all_case_summaries]
    best_center_ssim_list = [x["best_over_all_steps"]["best_center_ssim"]["center_metrics"] for x in all_case_summaries]

    runtimes = [float(x["runtime_sec"]) for x in all_case_summaries]

    summary = {
        "num_cases": len(all_case_summaries),
        "average_runtime_sec": float(np.mean(runtimes)) if len(runtimes) > 0 else None,
        "average_final_step": {
            "full_image": average_metric_dict(final_full_list),
            "center_200x200": average_metric_dict(final_center_list),
        },
        "average_best_over_all_steps": {
            "best_full_psnr": average_metric_dict(best_full_psnr_list),
            "best_full_ssim": average_metric_dict(best_full_ssim_list),
            "best_center_psnr": average_metric_dict(best_center_psnr_list),
            "best_center_ssim": average_metric_dict(best_center_ssim_list),
        },
        "per_case": all_case_summaries,
    }

    with open(os.path.join(save_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(save_dir, "dataset_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_name",
            "runtime_sec",

            "final_full_psnr", "final_full_ssim", "final_full_mse",
            "final_center_psnr", "final_center_ssim", "final_center_mse",

            "best_full_psnr_step", "best_full_psnr", "best_full_psnr_ssim", "best_full_psnr_mse",
            "best_full_ssim_step", "best_full_ssim_psnr", "best_full_ssim", "best_full_ssim_mse",

            "best_center_psnr_step", "best_center_psnr", "best_center_psnr_ssim", "best_center_psnr_mse",
            "best_center_ssim_step", "best_center_ssim_psnr", "best_center_ssim", "best_center_ssim_mse",
        ])
        for x in all_case_summaries:
            writer.writerow([
                x["case_name"],
                x["runtime_sec"],

                x["final_step"]["full_image"]["psnr"],
                x["final_step"]["full_image"]["ssim"],
                x["final_step"]["full_image"]["mse"],

                x["final_step"]["center_200x200"]["psnr"],
                x["final_step"]["center_200x200"]["ssim"],
                x["final_step"]["center_200x200"]["mse"],

                x["best_over_all_steps"]["best_full_psnr"]["step"],
                x["best_over_all_steps"]["best_full_psnr"]["full_metrics"]["psnr"],
                x["best_over_all_steps"]["best_full_psnr"]["full_metrics"]["ssim"],
                x["best_over_all_steps"]["best_full_psnr"]["full_metrics"]["mse"],

                x["best_over_all_steps"]["best_full_ssim"]["step"],
                x["best_over_all_steps"]["best_full_ssim"]["full_metrics"]["psnr"],
                x["best_over_all_steps"]["best_full_ssim"]["full_metrics"]["ssim"],
                x["best_over_all_steps"]["best_full_ssim"]["full_metrics"]["mse"],

                x["best_over_all_steps"]["best_center_psnr"]["step"],
                x["best_over_all_steps"]["best_center_psnr"]["center_metrics"]["psnr"],
                x["best_over_all_steps"]["best_center_psnr"]["center_metrics"]["ssim"],
                x["best_over_all_steps"]["best_center_psnr"]["center_metrics"]["mse"],

                x["best_over_all_steps"]["best_center_ssim"]["step"],
                x["best_over_all_steps"]["best_center_ssim"]["center_metrics"]["psnr"],
                x["best_over_all_steps"]["best_center_ssim"]["center_metrics"]["ssim"],
                x["best_over_all_steps"]["best_center_ssim"]["center_metrics"]["mse"],
            ])

    return summary


# =========================================================
# 11. Top-K 排名与对应图像导出
# =========================================================
def extract_rank_info(case_summary, rank_key):
    case_dir = case_summary["case_save_dir"]

    if rank_key == "best_full_psnr":
        rec = case_summary["best_over_all_steps"]["best_full_psnr"]
        metrics = rec["full_metrics"]
        step = rec["step"]
        main_image = os.path.join(case_dir, "best_step_results", "best_full_psnr.png")
        crop_image = os.path.join(case_dir, "best_step_results", "best_full_psnr_center_200x200.png")

    elif rank_key == "best_full_ssim":
        rec = case_summary["best_over_all_steps"]["best_full_ssim"]
        metrics = rec["full_metrics"]
        step = rec["step"]
        main_image = os.path.join(case_dir, "best_step_results", "best_full_ssim.png")
        crop_image = os.path.join(case_dir, "best_step_results", "best_full_ssim_center_200x200.png")

    elif rank_key == "best_center_psnr":
        rec = case_summary["best_over_all_steps"]["best_center_psnr"]
        metrics = rec["center_metrics"]
        step = rec["step"]
        main_image = os.path.join(case_dir, "best_step_results", "best_center_psnr.png")
        crop_image = os.path.join(case_dir, "best_step_results", "best_center_psnr_center_200x200.png")

    elif rank_key == "best_center_ssim":
        rec = case_summary["best_over_all_steps"]["best_center_ssim"]
        metrics = rec["center_metrics"]
        step = rec["step"]
        main_image = os.path.join(case_dir, "best_step_results", "best_center_ssim.png")
        crop_image = os.path.join(case_dir, "best_step_results", "best_center_ssim_center_200x200.png")

    elif rank_key == "final_full_psnr":
        metrics = case_summary["final_step"]["full_image"]
        step = case_summary["num_evaluated_steps"]
        main_image = os.path.join(case_dir, "result_eval_final.png")
        crop_image = os.path.join(case_dir, "result_eval_final_center_200x200.png")

    elif rank_key == "final_full_ssim":
        metrics = case_summary["final_step"]["full_image"]
        step = case_summary["num_evaluated_steps"]
        main_image = os.path.join(case_dir, "result_eval_final.png")
        crop_image = os.path.join(case_dir, "result_eval_final_center_200x200.png")

    elif rank_key == "final_center_psnr":
        metrics = case_summary["final_step"]["center_200x200"]
        step = case_summary["num_evaluated_steps"]
        main_image = os.path.join(case_dir, "result_eval_final.png")
        crop_image = os.path.join(case_dir, "result_eval_final_center_200x200.png")

    elif rank_key == "final_center_ssim":
        metrics = case_summary["final_step"]["center_200x200"]
        step = case_summary["num_evaluated_steps"]
        main_image = os.path.join(case_dir, "result_eval_final.png")
        crop_image = os.path.join(case_dir, "result_eval_final_center_200x200.png")

    else:
        raise ValueError(
            f"Unsupported topk_metric={rank_key}. "
            f"Supported: best_full_psnr, best_full_ssim, best_center_psnr, best_center_ssim, "
            f"final_full_psnr, final_full_ssim, final_center_psnr, final_center_ssim"
        )

    if rank_key.endswith("psnr"):
        rank_value = float(metrics["psnr"])
    elif rank_key.endswith("ssim"):
        rank_value = float(metrics["ssim"])
    else:
        raise ValueError(f"Cannot parse rank metric from rank_key={rank_key}")

    info = {
        "case_name": case_summary["case_name"],
        "case_save_dir": case_dir,
        "rank_key": rank_key,
        "rank_value": rank_value,
        "step": int(step),
        "psnr": float(metrics["psnr"]),
        "ssim": float(metrics["ssim"]),
        "mse": float(metrics["mse"]),
        "main_image": main_image,
        "crop_image": crop_image,
        "final_image": os.path.join(case_dir, "result_eval_final.png"),
        "final_crop_image": os.path.join(case_dir, "result_eval_final_center_200x200.png"),
        "sample_image": os.path.join(case_dir, "result_sample.png"),
        "gt_image": os.path.join(case_dir, "gt.png"),
        "gt_crop_image": os.path.join(case_dir, "gt_center_200x200.png"),
        "wf_image": os.path.join(case_dir, "wf_obs_raw.png"),
        "sim_mean_image": os.path.join(case_dir, "sim_mean.png"),
        "metrics_json": os.path.join(case_dir, "final_metrics.json"),
    }
    return info


def save_topk_results(all_case_summaries, save_dir, topk=20, rank_key="best_full_psnr"):
    ranked = []
    for case_summary in all_case_summaries:
        info = extract_rank_info(case_summary, rank_key)
        ranked.append(info)

    ranked.sort(key=lambda x: x["rank_value"], reverse=True)
    ranked = ranked[:min(topk, len(ranked))]

    topk_dir = os.path.join(save_dir, f"top{len(ranked)}_{rank_key}")
    os.makedirs(topk_dir, exist_ok=True)

    for i, item in enumerate(ranked, start=1):
        item["rank"] = i

    with open(os.path.join(topk_dir, "topk_summary.json"), "w") as f:
        json.dump(ranked, f, indent=2)

    with open(os.path.join(topk_dir, "topk_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "case_name",
            "rank_key",
            "rank_value",
            "step",
            "psnr",
            "ssim",
            "mse",
            "case_save_dir",
            "main_image",
            "crop_image",
            "final_image",
            "gt_image",
            "wf_image",
        ])
        for item in ranked:
            writer.writerow([
                item["rank"],
                item["case_name"],
                item["rank_key"],
                item["rank_value"],
                item["step"],
                item["psnr"],
                item["ssim"],
                item["mse"],
                item["case_save_dir"],
                item["main_image"],
                item["crop_image"],
                item["final_image"],
                item["gt_image"],
                item["wf_image"],
            ])

    txt_path = os.path.join(topk_dir, "topk_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Top-{len(ranked)} cases ranked by: {rank_key}\n")
        f.write("=" * 100 + "\n")
        for item in ranked:
            f.write(
                f"[Rank {item['rank']:02d}] "
                f"{item['case_name']} | "
                f"{rank_key}={item['rank_value']:.6f} | "
                f"step={item['step']} | "
                f"PSNR={item['psnr']:.6f} | "
                f"SSIM={item['ssim']:.6f} | "
                f"MSE={item['mse']:.8f}\n"
            )

    for item in ranked:
        rank_dir = os.path.join(topk_dir, f"rank_{item['rank']:02d}_{item['case_name']}")
        os.makedirs(rank_dir, exist_ok=True)

        safe_copy(item["main_image"], os.path.join(rank_dir, os.path.basename(item["main_image"])))
        safe_copy(item["crop_image"], os.path.join(rank_dir, os.path.basename(item["crop_image"])))
        safe_copy(item["final_image"], os.path.join(rank_dir, "result_eval_final.png"))
        safe_copy(item["final_crop_image"], os.path.join(rank_dir, "result_eval_final_center_200x200.png"))
        safe_copy(item["sample_image"], os.path.join(rank_dir, "result_sample.png"))
        safe_copy(item["gt_image"], os.path.join(rank_dir, "gt.png"))
        safe_copy(item["gt_crop_image"], os.path.join(rank_dir, "gt_center_200x200.png"))
        safe_copy(item["wf_image"], os.path.join(rank_dir, "wf_obs_raw.png"))
        safe_copy(item["sim_mean_image"], os.path.join(rank_dir, "sim_mean.png"))
        safe_copy(item["metrics_json"], os.path.join(rank_dir, "final_metrics.json"))

        with open(os.path.join(rank_dir, "rank_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"rank={item['rank']}\n")
            f.write(f"case_name={item['case_name']}\n")
            f.write(f"rank_key={item['rank_key']}\n")
            f.write(f"rank_value={item['rank_value']:.8f}\n")
            f.write(f"step={item['step']}\n")
            f.write(f"psnr={item['psnr']:.8f}\n")
            f.write(f"ssim={item['ssim']:.8f}\n")
            f.write(f"mse={item['mse']:.10f}\n")

    return ranked, topk_dir


# =========================================================
# 12. 主函数
# =========================================================
def main():
    args = create_argparser().parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    logger.configure(dir=args.save_dir)

    logger.log("🚀 创建扩散模型...")
    args.diffusion_steps = 3000
    model, diffusion = create_your_unet_and_diffusion(args, device)
    model.eval()

    logger.log("🚀 加载文本模型...")
    tokenizer, text_encoder = load_text_model(args, device)
    encoder_hidden_states = build_text_condition(tokenizer, text_encoder, args.prompt, device)
    if args.use_fp16:
        encoder_hidden_states = encoder_hidden_states.half()
    del text_encoder
    torch.cuda.empty_cache()

    logger.log("🚀 构建真实 SIMForwardModel...")
    # sim_forward_model = SIMForwardModel(
    #     image_size=args.hr_size,
    #     NA=args.sim_NA,
    #     wavelength=args.sim_wavelength,
    #     pixel_size=args.sim_pixel_size,
    #     modulation=args.sim_modulation,
    #     n_angles=args.sim_n_angles,
    #     n_phases=args.sim_n_phases,
    #     otf_cutoff_scale=args.sim_otf_cutoff_scale,
    #     read_noise_std=args.sim_read_noise_std,
    #     poisson_scale=args.sim_poisson_scale,
    #     background=args.sim_background,
    #     deterministic=True,
    #     device=device,
    # ).to(device)
    # sim_forward_model.eval()
    # ========== 你搜索出来的【固定最优参数】 ==========
    # mod=0.6, fr=0.9, bg=0.02, otf=0.9, blur=0.7
    Newmodel = SimpleStableSIM(
        image_size=256,
        modulation=0.6,
        illumination_freq_ratio=0.9,
        background=0.02,
        otf_cutoff_scale=0.9,
        post_blur_sigma=0.7,
        deterministic=True
    ).to(device)
    Newmodel.eval()
    cases = build_case_list(args)
    logger.log(f"✅ 共准备处理 {len(cases)} 个样本")

    all_case_summaries = []

    for idx, case in enumerate(cases, start=1):
        logger.log("=" * 80)
        logger.log(f"开始处理第 {idx}/{len(cases)} 个样本: {case['case_name']}")
        case_save_dir = os.path.join(args.save_dir, case["case_name"])

        case_summary = reconstruct_one_case(
            case_name=case["case_name"],
            wf_path=case["wf_path"],
            sim_path=case["sim_path"],
            gt_path=case["gt_path"],
            case_save_dir=case_save_dir,
            model=model,
            diffusion=diffusion,
            encoder_hidden_states=encoder_hidden_states,
            #sim_forward_model=sim_forward_model,
            sim_forward_model=Newmodel,
            device=device,
            args=args,
        )
        all_case_summaries.append(case_summary)

    dataset_summary = save_dataset_summary(all_case_summaries, args.save_dir)

    logger.log("\n" + "=" * 80)
    logger.log("===== 数据集平均指标（final step） =====")
    logger.log(
        f"[Avg Final Full] "
        f"MSE={dataset_summary['average_final_step']['full_image']['mse']:.6f}, "
        f"PSNR={dataset_summary['average_final_step']['full_image']['psnr']:.4f}, "
        f"SSIM={dataset_summary['average_final_step']['full_image']['ssim']:.4f}"
    )
    logger.log(
        f"[Avg Final Center200] "
        f"MSE={dataset_summary['average_final_step']['center_200x200']['mse']:.6f}, "
        f"PSNR={dataset_summary['average_final_step']['center_200x200']['psnr']:.4f}, "
        f"SSIM={dataset_summary['average_final_step']['center_200x200']['ssim']:.4f}"
    )

    logger.log("\n===== 数据集平均最佳指标（best over all steps） =====")
    logger.log(
        f"[Avg Best Full PSNR] "
        f"MSE={dataset_summary['average_best_over_all_steps']['best_full_psnr']['mse']:.6f}, "
        f"PSNR={dataset_summary['average_best_over_all_steps']['best_full_psnr']['psnr']:.4f}, "
        f"SSIM={dataset_summary['average_best_over_all_steps']['best_full_psnr']['ssim']:.4f}"
    )
    logger.log(
        f"[Avg Best Full SSIM] "
        f"MSE={dataset_summary['average_best_over_all_steps']['best_full_ssim']['mse']:.6f}, "
        f"PSNR={dataset_summary['average_best_over_all_steps']['best_full_ssim']['psnr']:.4f}, "
        f"SSIM={dataset_summary['average_best_over_all_steps']['best_full_ssim']['ssim']:.4f}"
    )
    logger.log(
        f"[Avg Best Center PSNR] "
        f"MSE={dataset_summary['average_best_over_all_steps']['best_center_psnr']['mse']:.6f}, "
        f"PSNR={dataset_summary['average_best_over_all_steps']['best_center_psnr']['psnr']:.4f}, "
        f"SSIM={dataset_summary['average_best_over_all_steps']['best_center_psnr']['ssim']:.4f}"
    )
    logger.log(
        f"[Avg Best Center SSIM] "
        f"MSE={dataset_summary['average_best_over_all_steps']['best_center_ssim']['mse']:.6f}, "
        f"PSNR={dataset_summary['average_best_over_all_steps']['best_center_ssim']['psnr']:.4f}, "
        f"SSIM={dataset_summary['average_best_over_all_steps']['best_center_ssim']['ssim']:.4f}"
    )

    topk_ranked, topk_dir = save_topk_results(
        all_case_summaries=all_case_summaries,
        save_dir=args.save_dir,
        topk=args.topk,
        rank_key=args.topk_metric,
    )

    logger.log("\n" + "=" * 80)
    logger.log(f"===== 全部样本中排名前 {len(topk_ranked)} 的结果（按 {args.topk_metric} 排序） =====")
    for item in topk_ranked:
        logger.log(
            f"[Rank {item['rank']:02d}] "
            f"{item['case_name']} | "
            f"{args.topk_metric}={item['rank_value']:.6f} | "
            f"step={item['step']} | "
            f"PSNR={item['psnr']:.6f} | "
            f"SSIM={item['ssim']:.6f} | "
            f"MSE={item['mse']:.8f}"
        )

    logger.log(f"\n💾 数据集汇总已保存: {os.path.join(args.save_dir, 'dataset_summary.json')}")
    logger.log(f"💾 数据集表格已保存: {os.path.join(args.save_dir, 'dataset_summary.csv')}")
    logger.log(f"💾 Top-K 排名结果已保存: {topk_dir}")
    logger.log(f"🎉 全部完成：{args.save_dir}")


# =========================================================
# 参数
# =========================================================
def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        diffusion_steps=3000,
        learn_sigma=False,
        sigma_small=True,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=True,
        rescale_learned_sigmas=False,
        # ===== 数据路径 =====
        wf_root="/data3/ddt/BIOSR-SIM/数据集/1014/1",
        sim_root="/data3/ddt/BIOSR-SIM/数据集/1014/2",
        gt_root="/data3/ddt/BIOSR-SIM/数据集/1014/3",
         # ===== 模型路径 =====
        unet_model_path="/data3/ddt/BIOSR-SIM/checkpoint-285000",
        text_encoder_path="/lab310/FoundationModelCkpts/models--sd-legacy--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/text_encoder",
        tokenizer_path="/lab310/FoundationModelCkpts/models--sd-legacy--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/tokenizer",

        prompt="Microtubules of COS-7",
        save_dir="./output_1014",
        save_numpy_array=False,
        save_per_step_images=False,
        save_observation_mean=True,
        device=0,
        use_fp16=True,
        denoise_steps=1000,
        hr_size=256,
        num_cases=100,
        tile_size_lr=128,
        tile_stride_lr=96,
        tile_noise_seed=1234,

        # ===== Top-K 排名输出 =====
        topk=20,
        topk_metric="best_full_psnr",

        # 日志
        eval_log_interval=5,
        cond_log_interval=100,

        # ===== 双引导权重 =====
        wf_guidance_weight=1.0,
        sim_guidance_weight=1.0,
        img_guidance_scale=120.0,
        sim_img_guidance_scale=120.0,
        tv_weight=0.1,

        # ===== WF退化参数 =====
        psf_sigma=3.3108672159510473,
        psf_size=15,  # 原14修正为奇数，符合卷积核要求
        gain=2.256348702536589,
        bias=0.05160694419403825,
        poisson_scale=117.87468359887765,
        read_noise_std=1.6913330857877473,
        downsample_scale=2,

        # ===== 真实SIMForwardModel参数 =====
        sim_NA=2.133474184418418,
        sim_wavelength=5.962737329617754e-07,
        sim_pixel_size=6.021088577943124e-08,
        sim_modulation=0.6173538728484552,
        sim_n_angles=3,
        sim_n_phases=3,
        sim_otf_cutoff_scale=3.939675104207492,
        sim_read_noise_std=0.0537164477034134,
        sim_poisson_scale=20.09950198602637,
        sim_background=2.6322560142879534,
            )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()