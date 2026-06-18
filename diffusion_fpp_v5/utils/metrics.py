import math

import torch
import torch.nn.functional as F


def gradient_xy(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


def _as_bool_mask(mask, like):
    if mask is None:
        return None
    mask = mask.to(device=like.device)
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    return mask


def edge_mask(target, quantile=0.80, mask=None):
    dx, dy = gradient_xy(target)
    mag = torch.sqrt(dx * dx + dy * dy)
    valid = _as_bool_mask(mask, target)
    if valid is None:
        flat = mag.flatten(1)
        thresh = torch.quantile(flat, quantile, dim=1).view(-1, 1, 1, 1)
        return mag >= thresh
    out = torch.zeros_like(valid)
    for i in range(target.shape[0]):
        vals = mag[i:i + 1][valid[i:i + 1]]
        if vals.numel() == 0:
            continue
        thresh = torch.quantile(vals, quantile)
        out[i:i + 1] = (mag[i:i + 1] >= thresh) & valid[i:i + 1]
    return out


def normal_error_deg(pred, target, mask=None):
    pdx, pdy = gradient_xy(pred)
    tdx, tdy = gradient_xy(target)
    pn = torch.cat([-pdx, -pdy, torch.ones_like(pred)], dim=1)
    tn = torch.cat([-tdx, -tdy, torch.ones_like(target)], dim=1)
    pn = F.normalize(pn, dim=1)
    tn = F.normalize(tn, dim=1)
    cos = (pn * tn).sum(dim=1).clamp(-1.0, 1.0)
    err = torch.rad2deg(torch.acos(cos)).unsqueeze(1)
    valid = _as_bool_mask(mask, pred)
    if valid is not None and valid.any():
        return err[valid].mean()
    return err.mean()


def ssim_simple(pred, target, data_range=1.0):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = F.avg_pool2d(pred, 7, stride=1, padding=3)
    mu_y = F.avg_pool2d(target, 7, stride=1, padding=3)
    sigma_x = F.avg_pool2d(pred * pred, 7, stride=1, padding=3) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 7, stride=1, padding=3) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 7, stride=1, padding=3) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8)
    return ssim.mean()


@torch.no_grad()
def compute_metrics(pred_mm, target_mm, mask=None):
    diff = pred_mm - target_mm
    valid = _as_bool_mask(mask, pred_mm)
    if valid is not None and valid.any():
        diff_valid = diff[valid]
        target_valid = target_mm[valid]
        rmse = torch.sqrt(torch.mean(diff_valid * diff_valid))
        mae = torch.mean(torch.abs(diff_valid))
        e_mask = edge_mask(target_mm, mask=valid)
        edge_rmse = torch.sqrt(torch.mean((diff[e_mask]) ** 2)) if e_mask.any() else rmse
        normal = normal_error_deg(pred_mm, target_mm, mask=valid)
        data_range = float((target_valid.max() - target_valid.min()).clamp(min=1.0).item())
        pred_ssim = pred_mm.masked_fill(~valid, 0.0)
        target_ssim = target_mm.masked_fill(~valid, 0.0)
        ssim = ssim_simple(pred_ssim, target_ssim, data_range=data_range)
    else:
        rmse = torch.sqrt(torch.mean(diff * diff))
        mae = torch.mean(torch.abs(diff))
        e_mask = edge_mask(target_mm)
        edge_rmse = torch.sqrt(torch.mean((diff[e_mask]) ** 2)) if e_mask.any() else rmse
        normal = normal_error_deg(pred_mm, target_mm)
        data_range = float((target_mm.max() - target_mm.min()).clamp(min=1.0).item())
        ssim = ssim_simple(pred_mm, target_mm, data_range=data_range)
    return {
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
        "edge_rmse": float(edge_rmse.item()),
        "normal_deg": float(normal.item()),
        "ssim": float(ssim.item()),
    }
