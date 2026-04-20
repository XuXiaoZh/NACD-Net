# v3/evaluate_ablation_v3.py
"""
消融实验验证脚本（4个变体 + P波拾取 + Quality相关性验证）
  - baseline_recon_only
  - wo_freq_grad_loss
  - wo_noise_condition
  - full

新增：
  - STA/LTA P波到时拾取
  - Quality 分数 vs P波拾取误差 相关性验证
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

sys.path.append(r"D:/X/denoise/part1")
sys.path.append(r"D:/X/denoise/part1/v3")

from v3.dataset_v3 import STEADDatasetV3
from v3.model_v3   import NoiseAwareDenoiserV3

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
CONFIG = {
    "event_h5":      "D:/X/p_wave/data/chunk2.hdf5",
    "val_csv":       "D:/X/p_wave/data/chunk2_val.csv",
    "noise_h5":      "D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv":     "D:/X/p_wave/data/chunk1.csv",
    "ablation_root": "D:/X/denoise/part1/v3/v3/checkpoints_ablation",
    "output_dir":    "D:/X/denoise/part1/v3/v3/eval_ablation",
    "signal_len":    6000,
    "cond_len":      400,
    "z_dim":         128,
    "num_heads":     8,
    "fs":            100,       # 采样率 Hz
    "batch_size":    1,
    "num_workers":   0,

    # STA/LTA 参数
    "sta_len":   0.5,    # STA 窗口（秒）
    "lta_len":   10.0,   # LTA 窗口（秒）
    "stalta_thr": 3.0,   # 触发阈值
    "pick_tol":   50,    # 拾取容忍误差（采样点），100Hz下=0.5s
}

EPS = 1e-10

# ─────────────────────────────────────────────────────
# 消融变体定义
# ─────────────────────────────────────────────────────
ABLATION_VARIANTS = [
    {
        "exp_name":    "baseline_recon_only",
        "ckpt_name":   "best_baseline_recon_only.pth",
        "z_cond_zero": False,
        "display":     "Baseline (Recon Only)",
    },
    {
        "exp_name":    "wo_freq_grad_loss",
        "ckpt_name":   "best_wo_freq_grad_loss.pth",
        "z_cond_zero": False,
        "display":     "w/o Freq+Grad Loss",
    },
    {
        "exp_name":    "wo_noise_condition",
        "ckpt_name":   "best_wo_noise_condition.pth",
        "z_cond_zero": True,
        "display":     "w/o Noise Condition",
    },
    {
        "exp_name":    "full",
        "ckpt_name":   "best_full.pth",
        "z_cond_zero": False,
        "display":     "Full Model",
    },
]

# ─────────────────────────────────────────────────────
# STA/LTA P波拾取
# ─────────────────────────────────────────────────────
def stalta_pick(wave: np.ndarray, fs: int,
                sta_len: float, lta_len: float,
                threshold: float) -> int:
    """
    STA/LTA P波到时拾取（取 Z 分量）

    Parameters
    ----------
    wave      : [3, T] 或 [T,] 波形
    fs        : 采样率
    sta_len   : STA 窗口长度（秒）
    lta_len   : LTA 窗口长度（秒）
    threshold : 触发阈值

    Returns
    -------
    pick : 拾取到的采样点索引，未拾取返回 -1
    """
    # 取 Z 分量
    if wave.ndim == 2:
        x = wave[2].astype(np.float64)
    else:
        x = wave.astype(np.float64)

    nsta = int(sta_len * fs)
    nlta = int(lta_len * fs)
    T    = len(x)

    if T < nlta + nsta:
        return -1

    # 计算特征函数（振幅平方）
    cf = x ** 2

    # 累积和加速计算
    cs    = np.cumsum(cf)
    cs    = np.concatenate([[0.0], cs])

    ratio = np.zeros(T)
    for i in range(nlta, T - nsta):
        lta = (cs[i]        - cs[i - nlta]) / nlta
        sta = (cs[i + nsta] - cs[i])        / nsta
        if lta > EPS:
            ratio[i] = sta / lta

    # 找第一个超过阈值的点
    triggered = np.where(ratio > threshold)[0]
    if len(triggered) == 0:
        return -1

    return int(triggered[0])

def stalta_pick_fast(wave: np.ndarray, fs: int,
                     sta_len: float, lta_len: float,
                     threshold: float) -> int:
    """
    向量化快速版 STA/LTA（避免 Python 循环）
    """
    if wave.ndim == 2:
        x = wave[2].astype(np.float64)
    else:
        x = wave.astype(np.float64)

    nsta = int(sta_len * fs)
    nlta = int(lta_len * fs)
    T    = len(x)

    if T < nlta + nsta:
        return -1

    cf = x ** 2
    cs = np.cumsum(np.concatenate([[0.0], cf]))

    # 有效范围
    i_start = nlta
    i_end   = T - nsta

    if i_start >= i_end:
        return -1

    idx  = np.arange(i_start, i_end)
    lta  = (cs[idx]        - cs[idx - nlta]) / nlta
    sta  = (cs[idx + nsta] - cs[idx])        / nsta

    # 避免除零
    valid = lta > EPS
    ratio = np.zeros(len(idx))
    ratio[valid] = sta[valid] / lta[valid]

    triggered = np.where(ratio > threshold)[0]
    if len(triggered) == 0:
        return -1

    return int(triggered[0] + i_start)

# ─────────────────────────────────────────────────────
# 指标函数
# ─────────────────────────────────────────────────────
def snr_db(clean, test):
    clean = np.asarray(clean, dtype=np.float64)
    test  = np.asarray(test,  dtype=np.float64)
    sig   = np.sum(clean ** 2)
    noi   = np.sum((test - clean) ** 2)
    return float(10.0 * np.log10((sig + EPS) / (noi + EPS)))

def cc_fn(clean, test):
    c = clean - clean.mean()
    t = test  - test.mean()
    d = np.sqrt(np.sum(c**2) * np.sum(t**2))
    return float(np.sum(c * t) / d) if d > EPS else 0.0

def rmse_fn(clean, test):
    return float(np.sqrt(np.mean((clean - test) ** 2)))

def prd_fn(clean, test):
    return float(np.sqrt(
        np.sum((clean - test) ** 2) / (np.sum(clean ** 2) + EPS)
    ))

def st_mae_mean(clean, test, fs=100, win_ms=100, overlap=0.5):
    n   = min(len(clean), len(test))
    win = min(int(fs * win_ms / 1000), n)
    if win <= 0:
        return float('nan')
    hop  = max(1, int(win * (1.0 - overlap)))
    vals = [np.abs(clean[s:s+win] - test[s:s+win]).mean()
            for s in range(0, n - win + 1, hop)]
    return float(np.mean(vals)) if vals else float('nan')

# ─────────────────────────────────────────────────────
# SNR 分组
# ─────────────────────────────────────────────────────
SNR_BINS   = [-np.inf, -5, 0, 5, 10, 15, np.inf]
SNR_LABELS = ['<-5', '-5~0', '0~5', '5~10', '10~15', '>15']

def assign_snr_group(snr_val):
    for i in range(len(SNR_BINS) - 1):
        if SNR_BINS[i] <= snr_val < SNR_BINS[i + 1]:
            return SNR_LABELS[i]
    return SNR_LABELS[-1]

# ─────────────────────────────────────────────────────
# Quality vs 拾取误差 相关性图
# ─────────────────────────────────────────────────────
def plot_quality_vs_pick_error(quality_list, pick_error_list,
                               pick_success_list,
                               save_path, title="Quality vs P-pick Error"):
    """
    quality_list      : list[float]  模型输出质量分
    pick_error_list   : list[float]  拾取误差（采样点），未拾取为 NaN
    pick_success_list : list[bool]   是否成功拾取
    """
    q   = np.array(quality_list,      dtype=np.float64)
    err = np.array(pick_error_list,   dtype=np.float64)
    suc = np.array(pick_success_list, dtype=bool)

    # 只取成功拾取的样本做相关性
    valid = suc & np.isfinite(err)
    q_v   = q[valid]
    e_v   = err[valid]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # ── 图1：散点图 Quality vs 拾取误差 ──────────────
    ax = axes[0]
    sc = ax.scatter(q_v, e_v, c=e_v, cmap='RdYlGn_r',
                    alpha=0.6, s=20, vmin=0, vmax=CONFIG["pick_tol"] * 2)
    plt.colorbar(sc, ax=ax, label='Pick Error (samples)')
    if len(q_v) > 2:
        pr, pp = pearsonr(q_v, e_v)
        sr, sp = spearmanr(q_v, e_v)
        ax.set_title(
            f"Quality vs Pick Error\n"
            f"Pearson r={pr:.3f}(p={pp:.3e})  "
            f"Spearman ρ={sr:.3f}(p={sp:.3e})",
            fontsize=9
        )
    ax.set_xlabel("Quality Score", fontsize=10)
    ax.set_ylabel("Pick Error (samples)", fontsize=10)
    ax.axvline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.grid(alpha=0.3)

    # ── 图2：Quality 分箱 → 平均拾取误差 ─────────────
    ax = axes[1]
    bins   = np.linspace(0, 1, 11)          # 0.0, 0.1, ..., 1.0
    labels = [f"{bins[i]:.1f}~{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    bin_idx = np.digitize(q_v, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(labels) - 1)

    means, stds, counts = [], [], []
    for b in range(len(labels)):
        mask = bin_idx == b
        if mask.sum() > 0:
            means.append(e_v[mask].mean())
            stds.append(e_v[mask].std())
            counts.append(mask.sum())
        else:
            means.append(np.nan)
            stds.append(0)
            counts.append(0)

    x_pos = np.arange(len(labels))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(labels)))
    bars = ax.bar(x_pos, means, yerr=stds, color=colors,
                  alpha=0.85, capsize=4, width=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.set_xlabel("Quality Score Bin", fontsize=10)
    ax.set_ylabel("Mean Pick Error (samples)", fontsize=10)
    ax.set_title("Quality Bin → Mean Pick Error\n(↑Quality → ↓Error?)", fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    # 标注样本数
    for bar, cnt, m in zip(bars, counts, means):
        if cnt > 0 and not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width()/2,
                    m + 0.5, f"n={cnt}",
                    ha='center', fontsize=6, color='black')

    # ── 图3：Quality 阈值 → 拾取成功率 ───────────────
    ax = axes[2]
    thresholds = np.linspace(0.0, 0.95, 20)
    success_rates = []
    sample_ratios = []

    tol = CONFIG["pick_tol"]
    for thr in thresholds:
        mask_thr = q >= thr
        if mask_thr.sum() == 0:
            success_rates.append(np.nan)
            sample_ratios.append(0)
            continue
        # 在 quality >= thr 的样本中，拾取误差 <= tol 的比例
        sub_suc = suc[mask_thr]
        sub_err = err[mask_thr]
        good    = sub_suc & (sub_err <= tol)
        success_rates.append(good.sum() / mask_thr.sum())
        sample_ratios.append(mask_thr.sum() / len(q))

    ax2 = ax.twinx()
    ax.plot(thresholds, success_rates, 'b-o', ms=4,
            label=f'Pick Success Rate (err≤{tol})')
    ax2.plot(thresholds, sample_ratios, 'r--s', ms=4,
             label='Sample Ratio', alpha=0.7)
    ax.set_xlabel("Quality Score Threshold", fontsize=10)
    ax.set_ylabel("Pick Success Rate", fontsize=10, color='blue')
    ax2.set_ylabel("Remaining Sample Ratio", fontsize=10, color='red')
    ax.set_title("Quality Threshold → Pick Success Rate", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] Quality相关性图: {os.path.basename(save_path)}")

# ─────────────────────────────────────────────────────
# SNR summary 图
# ─────────────────────────────────────────────────────
def plot_snr_summary(all_snr_in, all_snr_out, all_groups,
                     save_path, title="SNR Analysis"):
    color_map = {
        '<-5':   '#d62728', '-5~0':  '#ff7f0e',
        '0~5':   '#2ca02c', '5~10':  '#1f77b4',
        '10~15': '#9467bd', '>15':   '#8c564b',
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax = axes[0]
    for label in SNR_LABELS:
        si = [s for s, g in zip(all_snr_in,  all_groups) if g == label]
        so = [s for s, g in zip(all_snr_out, all_groups) if g == label]
        if si:
            ax.scatter(si, so, label=label,
                       color=color_map[label], alpha=0.75, s=45, zorder=3)
    lo = min(all_snr_in + all_snr_out) - 2
    hi = max(all_snr_in + all_snr_out) + 2
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, alpha=0.5, label='y=x')
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("Output SNR (dB)")
    ax.set_title("Input vs Output SNR")
    ax.legend(title="Group", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    group_gains = {}
    for si, so, g in zip(all_snr_in, all_snr_out, all_groups):
        group_gains.setdefault(g, []).append(so - si)
    labels_present = [l for l in SNR_LABELS if l in group_gains]
    means  = [np.mean(group_gains[l]) for l in labels_present]
    stds   = [np.std(group_gains[l])  for l in labels_present]
    colors = [color_map[l]            for l in labels_present]
    bars   = ax.bar(labels_present, means, yerr=stds,
                    color=colors, alpha=0.8, capsize=5, width=0.6)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel("SNR Group (Input)")
    ax.set_ylabel("Mean SNR Gain (dB)")
    ax.set_title("Per-Group SNR Gain")
    ax.grid(axis='y', alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                m + (0.1 if m >= 0 else -0.4),
                f"{m:+.2f}", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ─────────────────────────────────────────────────────
# 多模型对比柱状图
# ─────────────────────────────────────────────────────
def plot_multi_model_compare(summary_rows, output_dir):
    metrics = [
        ("ΔSNR (dB)",         "delta_snr",       True),
        ("CC",                "cc",               True),
        ("RMSE",              "rmse",             False),
        ("ST-MAE (denoised)", "st_mae_denoised",  False),
        ("Pick Success Rate", "pick_success_rate",True),   # 新增
        ("Quality Score",     "quality",          True),   # 新增
    ]
    names  = [r["display"] for r in summary_rows]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(names)]

    fig, axes = plt.subplots(1, len(metrics), figsize=(24, 5))
    fig.suptitle("消融实验指标对比", fontsize=13, fontweight="bold")

    for ax, (ylabel, key, higher_better) in zip(axes, metrics):
        vals = [r.get(key, float('nan')) for r in summary_rows]
        valid_vals = [v for v in vals if not np.isnan(v)]
        if not valid_vals:
            continue
        bars = ax.bar(names, vals, color=colors, alpha=0.85, width=0.5)
        ax.set_title(ylabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='x', labelrotation=15, labelsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(valid_vals[0], color='gray', lw=0.8, ls='--', alpha=0.6)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        v + abs(v) * 0.01,
                        f"{v:.3f}", ha='center', fontsize=8)
        note = "↑ better" if higher_better else "↓ better"
        ax.text(0.98, 0.98, note, transform=ax.transAxes,
                fontsize=8, ha='right', va='top', color='gray')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "ablation_compare_bar.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 多模型对比图: {os.path.basename(save_path)}")

# ─────────────────────────────────────────────────────
# 单模型评估核心函数
# ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_one_variant(model, loader, device, variant):
    model.eval()

    results_by_group = {label: [] for label in SNR_LABELS}
    all_snr_in, all_snr_out, all_groups = [], [], []

    # P波拾取 & Quality 相关性收集
    quality_list      = []   # 模型质量分
    pick_error_noisy  = []   # 含噪输入的拾取误差
    pick_error_denoise= []   # 去噪后的拾取误差
    pick_success_noisy  = []
    pick_success_denoise= []

    skip = 0

    for batch_idx, batch in enumerate(loader):
        x          = batch["x"].to(device)
        y_clean    = batch["y_clean"].to(device)
        z_cond     = batch["z_cond"].to(device)
        has_target = batch["has_target"].to(device)
        p_onset    = batch["p_onset"]              # [B] 真实P波到时（CPU）

        sup = has_target.bool()
        if not sup.any():
            continue

        x_s     = x[sup]
        y_s     = y_clean[sup]
        zc_s    = z_cond[sup]
        p_s     = p_onset[sup.cpu()]               # 对应真实到时

        if variant["z_cond_zero"]:
            zc_s = torch.zeros_like(zc_s)

        if not all(torch.isfinite(t).all() for t in [x_s, y_s, zc_s]):
            skip += 1
            continue

        try:
            pred, quality, _ = model(x_s, zc_s)
        except Exception as e:
            skip += 1
            if skip <= 3:
                print(f"  ⚠ forward error: {e}")
            continue

        if not torch.isfinite(pred).all():
            skip += 1
            continue

        # ── 逐样本处理 ─────────────────────────────────
        for i in range(x_s.shape[0]):
            x_np    = x_s[i].cpu().numpy()       # [3, T]
            y_np    = y_s[i].cpu().numpy()
            pred_np = pred[i].cpu().numpy()
            qual    = float(quality[i].item())
            p_true  = int(p_s[i].item())

            # Z 分量
            x_z    = x_np[2].astype(np.float64)
            y_z    = y_np[2].astype(np.float64)
            pred_z = pred_np[2].astype(np.float64)

            # ── 去噪指标 ──────────────────────────────
            snr_i = snr_db(y_z, x_z)
            snr_o = snr_db(y_z, pred_z)
            group = assign_snr_group(snr_i)

            record = {
                "snr_in":          snr_i,
                "snr_out":         snr_o,
                "delta_snr":       snr_o - snr_i,
                "cc":              cc_fn(y_z, pred_z),
                "rmse":            rmse_fn(y_z, pred_z),
                "prd":             prd_fn(y_z, pred_z),
                "st_mae_noisy":    st_mae_mean(y_z, x_z,    CONFIG["fs"]),
                "st_mae_denoised": st_mae_mean(y_z, pred_z, CONFIG["fs"]),
                "quality":         qual,
            }
            results_by_group[group].append(record)
            all_snr_in.append(snr_i)
            all_snr_out.append(snr_o)
            all_groups.append(group)

            # ── STA/LTA P波拾取 ───────────────────────
            # 含噪输入拾取
            pick_n = stalta_pick_fast(
                x_np,
                fs        = CONFIG["fs"],
                sta_len   = CONFIG["sta_len"],
                lta_len   = CONFIG["lta_len"],
                threshold = CONFIG["stalta_thr"],
            )
            # 去噪后拾取
            pick_d = stalta_pick_fast(
                pred_np,
                fs        = CONFIG["fs"],
                sta_len   = CONFIG["sta_len"],
                lta_len   = CONFIG["lta_len"],
                threshold = CONFIG["stalta_thr"],
            )

            tol = CONFIG["pick_tol"]

            # 含噪拾取误差
            if pick_n >= 0:
                err_n = abs(pick_n - p_true)
                pick_error_noisy.append(float(err_n))
                pick_success_noisy.append(err_n <= tol)
            else:
                pick_error_noisy.append(float('nan'))
                pick_success_noisy.append(False)

            # 去噪后拾取误差
            if pick_d >= 0:
                err_d = abs(pick_d - p_true)
                pick_error_denoise.append(float(err_d))
                pick_success_denoise.append(err_d <= tol)
            else:
                pick_error_denoise.append(float('nan'))
                pick_success_denoise.append(False)

            quality_list.append(qual)

    if skip > 0:
        print(f"  ⚠ 跳过 {skip} 个异常样本")

    pick_results = {
        "quality":               np.array(quality_list),
        "pick_error_noisy":      np.array(pick_error_noisy),
        "pick_error_denoise":    np.array(pick_error_denoise),
        "pick_success_noisy":    np.array(pick_success_noisy),
        "pick_success_denoise":  np.array(pick_success_denoise),
    }

    return results_by_group, all_snr_in, all_snr_out, all_groups, pick_results

# ─────────────────────────────────────────────────────
# 打印 P波拾取统计
# ─────────────────────────────────────────────────────
def print_pick_stats(pick_results, display_name):
    q   = pick_results["quality"]
    en  = pick_results["pick_error_noisy"]
    ed  = pick_results["pick_error_denoise"]
    sn  = pick_results["pick_success_noisy"]
    sd  = pick_results["pick_success_denoise"]
    tol = CONFIG["pick_tol"]

    n_total = len(q)

    # 成功拾取率
    rate_n = sn.sum() / n_total if n_total > 0 else 0
    rate_d = sd.sum() / n_total if n_total > 0 else 0

    # 有效误差（成功拾取的样本）
    valid_n = np.isfinite(en) & sn
    valid_d = np.isfinite(ed) & sd

    mae_n = float(en[valid_n].mean()) if valid_n.sum() > 0 else float('nan')
    mae_d = float(ed[valid_d].mean()) if valid_d.sum() > 0 else float('nan')

    # Quality vs 去噪后拾取误差 相关性
    valid_corr = np.isfinite(ed) & sd
    if valid_corr.sum() > 5:
        pr, pp = pearsonr(q[valid_corr], ed[valid_corr])
        sr, sp = spearmanr(q[valid_corr], ed[valid_corr])
    else:
        pr = pp = sr = sp = float('nan')

    print(f"\n  {'─'*55}")
    print(f"  P波拾取统计 [{display_name}]")
    print(f"  {'─'*55}")
    print(f"  总样本数              : {n_total}")
    print(f"  含噪输入拾取成功率     : {rate_n:.3f}  ({sn.sum()}/{n_total})")
    print(f"  去噪后拾取成功率       : {rate_d:.3f}  ({sd.sum()}/{n_total})")
    print(f"  含噪输入平均拾取误差   : {mae_n:.1f} samples "
          f"({mae_n/CONFIG['fs']*1000:.0f} ms)" if not np.isnan(mae_n) else
          f"  含噪输入平均拾取误差   : N/A")
    print(f"  去噪后平均拾取误差     : {mae_d:.1f} samples "
          f"({mae_d/CONFIG['fs']*1000:.0f} ms)" if not np.isnan(mae_d) else
          f"  去噪后平均拾取误差     : N/A")
    print(f"  Quality vs 拾取误差相关性:")
    print(f"    Pearson  r = {pr:.4f}  (p={pp:.3e})")
    print(f"    Spearman ρ = {sr:.4f}  (p={sp:.3e})")
    if pr < -0.1:
        print(f"    ✅ 负相关：Quality↑ → 拾取误差↓（符合预期）")
    elif pr > 0.1:
        print(f"    ⚠️  正相关：Quality↑ → 拾取误差↑（不符合预期）")
    else:
        print(f"    ➖ 弱相关：Quality 与拾取误差关系不显著")

    return {
        "pick_success_rate_noisy":   round(rate_n, 4),
        "pick_success_rate_denoise": round(rate_d, 4),
        "pick_mae_noisy":            round(mae_n, 2) if not np.isnan(mae_n) else None,
        "pick_mae_denoise":          round(mae_d, 2) if not np.isnan(mae_d) else None,
        "pearson_r":                 round(pr, 4)    if not np.isnan(pr)    else None,
        "spearman_r":                round(sr, 4)    if not np.isnan(sr)    else None,
    }

# ─────────────────────────────────────────────────────
# 聚合分组均值
# ─────────────────────────────────────────────────────
def aggregate_group_rows(results_by_group, display_name):
    def mean_of(records, key):
        vs = [r[key] for r in records if not np.isnan(float(r[key]))]
        return float(np.mean(vs)) if vs else float('nan')

    group_rows  = []
    all_records = []

    for label in SNR_LABELS:
        records = results_by_group[label]
        if not records:
            continue
        all_records.extend(records)
        group_rows.append({
            "Model":            display_name,
            "SNR_Group":        label,
            "Count":            len(records),
            "SNR_in":           round(mean_of(records, "snr_in"),          3),
            "SNR_out":          round(mean_of(records, "snr_out"),         3),
            "ΔSNR":             round(mean_of(records, "delta_snr"),       3),
            "CC":               round(mean_of(records, "cc"),              4),
            "RMSE":             round(mean_of(records, "rmse"),            4),
            "PRD":              round(mean_of(records, "prd"),             4),
            "ST-MAE(noisy)":    round(mean_of(records, "st_mae_noisy"),    4),
            "ST-MAE(denoised)": round(mean_of(records, "st_mae_denoised"), 4),
            "Quality":          round(mean_of(records, "quality"),         4),
        })

    global_summary = {
        "display":         display_name,
        "total":           len(all_records),
        "delta_snr":       mean_of(all_records, "delta_snr"),
        "cc":              mean_of(all_records, "cc"),
        "rmse":            mean_of(all_records, "rmse"),
        "prd":             mean_of(all_records, "prd"),
        "st_mae_noisy":    mean_of(all_records, "st_mae_noisy"),
        "st_mae_denoised": mean_of(all_records, "st_mae_denoised"),
        "quality":         mean_of(all_records, "quality"),
        "snr_in":          mean_of(all_records, "snr_in"),
        "snr_out":         mean_of(all_records, "snr_out"),
    }

    return group_rows, global_summary

def print_group_table(group_rows, display_name, global_summary):
    print(f"\n{'='*70}")
    print(f"  模型: {display_name}")
    print(f"{'='*70}")
    print(f"  {'Group':>7} | {'N':>4} | "
          f"{'SNR_in':>7} → {'SNR_out':>7} ({'ΔSNR':>6}) | "
          f"{'CC':>6} | {'RMSE':>7} | {'PRD':>7} | "
          f"{'ST-MAE(n)':>10} | {'ST-MAE(d)':>10}")
    print(f"  {'-'*100}")
    for r in group_rows:
        print(f"  {r['SNR_Group']:>7} | {r['Count']:>4} | "
              f"{r['SNR_in']:>7.2f} → {r['SNR_out']:>7.2f} "
              f"({r['ΔSNR']:>+6.2f}) | "
              f"{r['CC']:>6.4f} | {r['RMSE']:>7.4f} | {r['PRD']:>7.4f} | "
              f"{r['ST-MAE(noisy)']:>10.4f} | {r['ST-MAE(denoised)']:>10.4f}")

    # ── 整体均值行 ────────────────────────────────────
    g = global_summary
    print(f"  {'-'*100}")
    print(f"  {'Overall':>7} | {g['total']:>4} | "
          f"{g['snr_in']:>7.2f} → {g['snr_out']:>7.2f} "
          f"({g['delta_snr']:>+6.2f}) | "
          f"{g['cc']:>6.4f} | {g['rmse']:>7.4f} | {g['prd']:>7.4f} | "
          f"{g['st_mae_noisy']:>10.4f} | {g['st_mae_denoised']:>10.4f}")
# ─────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ── 路径检查 ──────────────────────────────────────
    print("\n[路径检查]")
    for k, p in [("event_h5", CONFIG["event_h5"]),
                 ("val_csv",  CONFIG["val_csv"]),
                 ("noise_h5", CONFIG["noise_h5"]),
                 ("noise_csv",CONFIG["noise_csv"])]:
        print(f"  [{'✅' if os.path.exists(p) else '❌'}] {k}: {p}")
    for v in ABLATION_VARIANTS:
        ckpt = os.path.join(CONFIG["ablation_root"],
                            v["exp_name"], v["ckpt_name"])
        print(f"  [{'✅' if os.path.exists(ckpt) else '❌'}] "
              f"{v['display']}: {ckpt}")

    # ── 数据集 ────────────────────────────────────────
    print(f"\n[数据集] 加载: {CONFIG['val_csv']}")
    val_ds = STEADDatasetV3(
        event_h5_path  = CONFIG["event_h5"],
        event_csv_path = CONFIG["val_csv"],
        noise_h5_path  = CONFIG["noise_h5"],
        noise_csv_path = CONFIG["noise_csv"],
        signal_len     = CONFIG["signal_len"],
        cond_len       = CONFIG["cond_len"],
        snr_range      = (0.1, 20.0),
        clean_prob     = 0.0,
        part_b_ratio   = 0.0,
        seed           = 42,
    )
    loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=CONFIG["num_workers"])
    print(f"  样本总数: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device  : {device}")

    all_group_rows   = []
    all_global_stats = []
    all_pick_stats   = []

    for variant in ABLATION_VARIANTS:
        ckpt_path = os.path.join(CONFIG["ablation_root"],
                                 variant["exp_name"], variant["ckpt_name"])
        if not os.path.exists(ckpt_path):
            print(f"\n⚠️  跳过 [{variant['display']}]，权重不存在")
            continue

        print(f"\n{'─'*60}")
        print(f"🎯 评估: {variant['display']}")

        model = NoiseAwareDenoiserV3(
            z_dim=CONFIG["z_dim"], cond_len=CONFIG["cond_len"],
            num_heads=CONFIG["num_heads"],
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # ── 评估 ──────────────────────────────────────
        (results_by_group, all_snr_in, all_snr_out,
         all_groups, pick_results) = evaluate_one_variant(
            model, loader, device, variant
        )

        # ── 去噪指标聚合 ──────────────────────────────
        group_rows, global_summary = aggregate_group_rows(
            results_by_group, variant["display"]
        )
        # evaluate_baseline.py 和 evaluate_ablation_v3.py 都改这一行
        print_group_table(group_rows, variant["display"], global_summary)

        # ── P波拾取统计 ───────────────────────────────
        pick_stats = print_pick_stats(pick_results, variant["display"])

        # 将拾取成功率写入 global_summary（用于对比柱状图）
        global_summary["pick_success_rate"] = \
            pick_stats["pick_success_rate_denoise"]

        # ── 保存 CSV ──────────────────────────────────
        df_group = pd.DataFrame(group_rows)
        csv_path = os.path.join(CONFIG["output_dir"],
                                f"results_{variant['exp_name']}.csv")
        df_group.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"\n  ✅ 分组结果: {csv_path}")

        # ── 保存 P波拾取详细 CSV ──────────────────────
        pick_df = pd.DataFrame({
            "quality":            pick_results["quality"],
            "pick_error_noisy":   pick_results["pick_error_noisy"],
            "pick_error_denoise": pick_results["pick_error_denoise"],
            "pick_success_noisy": pick_results["pick_success_noisy"],
            "pick_success_denoise": pick_results["pick_success_denoise"],
        })
        pick_csv = os.path.join(CONFIG["output_dir"],
                                f"pick_{variant['exp_name']}.csv")
        pick_df.to_csv(pick_csv, index=False, float_format="%.4f")
        print(f"  ✅ P波拾取详情: {pick_csv}")

        # ── 图：SNR summary ───────────────────────────
        if all_snr_in:
            plot_snr_summary(
                all_snr_in, all_snr_out, all_groups,
                save_path=os.path.join(CONFIG["output_dir"],
                    f"snr_summary_{variant['exp_name']}.png"),
                title=f"Ablation: {variant['display']}",
            )

        # ── 图：Quality vs 拾取误差 ───────────────────
        plot_quality_vs_pick_error(
            quality_list      = pick_results["quality"].tolist(),
            pick_error_list   = pick_results["pick_error_denoise"].tolist(),
            pick_success_list = pick_results["pick_success_denoise"].tolist(),
            save_path         = os.path.join(CONFIG["output_dir"],
                f"quality_pick_{variant['exp_name']}.png"),
            title = f"Quality vs P-pick Error [{variant['display']}]",
        )

        all_group_rows.extend(group_rows)
        all_global_stats.append(global_summary)
        all_pick_stats.append({**pick_stats, "display": variant["display"]})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 全局对比 ──────────────────────────────────────
    if len(all_global_stats) > 1:
        print(f"\n{'='*90}")
        print("🏆 消融实验全局对比")
        print(f"{'='*90}")
        for g in all_global_stats:
            print(f"  {g['display']:<25} "
                  f"ΔSNR={g['delta_snr']:+.3f}  "
                  f"CC={g['cc']:.4f}  "
                  f"RMSE={g['rmse']:.4f}  "
                  f"Pick={g.get('pick_success_rate', float('nan')):.3f}  "
                  f"Quality={g['quality']:.4f}")

        # P波拾取汇总表
        print(f"\n{'─'*70}")
        print("📍 P波拾取汇总")
        print(f"{'─'*70}")
        print(f"  {'Model':<25} | {'Rate(noisy)':>11} | "
              f"{'Rate(denoise)':>13} | {'MAE_n':>7} | "
              f"{'MAE_d':>7} | {'Pearson':>8} | {'Spearman':>9}")
        print(f"  {'-'*90}")
        for ps in all_pick_stats:
            print(f"  {ps['display']:<25} | "
                  f"{ps['pick_success_rate_noisy']:>11.4f} | "
                  f"{ps['pick_success_rate_denoise']:>13.4f} | "
                  f"{str(ps['pick_mae_noisy']):>7} | "
                  f"{str(ps['pick_mae_denoise']):>7} | "
                  f"{str(ps['pearson_r']):>8} | "
                  f"{str(ps['spearman_r']):>9}")

        # 保存 CSV
        pd.DataFrame(all_global_stats).to_csv(
            os.path.join(CONFIG["output_dir"], "ablation_global_compare.csv"),
            index=False, float_format="%.4f"
        )
        pd.DataFrame(all_pick_stats).to_csv(
            os.path.join(CONFIG["output_dir"], "ablation_pick_summary.csv"),
            index=False, float_format="%.4f"
        )
        pd.DataFrame(all_group_rows).to_csv(
            os.path.join(CONFIG["output_dir"], "ablation_all_groups.csv"),
            index=False, float_format="%.4f"
        )

        plot_multi_model_compare(all_global_stats, CONFIG["output_dir"])

    print("\n[✅ 评估完成]")
    print(f"结果目录: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()

