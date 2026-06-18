import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def save_comparison(fringe, target_mm, pred_mm, save_path, title="", mask=None):
    fringe = fringe.detach().cpu()[0, 0]
    target = target_mm.detach().cpu()[0, 0]
    pred = pred_mm.detach().cpu()[0, 0]
    err = (pred - target).abs()
    if mask is not None:
        mask = mask.detach().cpu()[0, 0] > 0.5
        valid = mask.numpy()
        target_show = np.ma.masked_where(~valid, target.numpy())
        pred_show = np.ma.masked_where(~valid, pred.numpy())
        err_show = np.ma.masked_where(~valid, err.numpy())
        valid_target = target[mask]
        vmin = float(valid_target.min()) if valid_target.numel() else float(target.min())
        vmax = float(valid_target.max()) if valid_target.numel() else float(target.max())
        err_mean = float(err[mask].mean()) if mask.any() else float(err.mean())
    else:
        target_show = target
        pred_show = pred
        err_show = err
        vmin, vmax = float(target.min()), float(target.max())
        err_mean = float(err.mean())

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].imshow(fringe, cmap="gray")
    axes[0].set_title("Fringe")
    axes[1].imshow(target_show, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("GT height")
    axes[2].imshow(pred_show, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].set_title(title or "Prediction")
    im = axes[3].imshow(err_show, cmap="hot")
    axes[3].set_title(f"Abs error mean={err_mean:.2f}mm")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
