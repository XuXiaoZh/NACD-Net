# -*- coding: utf-8 -*-
"""
迁移学习可视化脚本 v3（与 xiaorong_v2 预处理对齐，输出 JPG + SVG）
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import platform
from torch.utils.data import random_split

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [THIS_DIR, os.path.abspath(os.path.join(THIS_DIR, ".."))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_v3 import NoiseAwareDenoiserV3

# ============================================================
# 中文字体配置
# ============================================================
def _set_font():
    system = platform.system()
    if system == "Windows":
        candidates = ["Microsoft YaHei", "SimHei", "SimSun"]
    elif system == "Darwin":
        candidates = ["PingFang SC", "Heiti TC", "STHeiti"]
    else:
        candidates = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字体] 使用: {font}")
            return
    print("[字体] 未找到中文字体，使用默认字体")

_set_font()

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "pretrain_ckpt": r"D:/X/denoise/part1/v3/checkpoints_v3/best_model_v3.pth",
    "finetune_h5":   r"D:/X/p_wave/data/non_naturaldata.hdf5",
    "finetune_csv":  r"D:/X/p_wave/data/non_naturaldata.csv",
    "noise_h5":      r"D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv":     r"D:/X/p_wave/data/chunk1.csv",
    "output_dir":    r"D:/X/denoise/part1/v3/exp2_freeze_v3/val_plots",
    "exp2_dir":      r"D:/X/denoise/part1/v3/exp2_freeze_fixed",
    "z_dim":         128,
    "cond_len":      400,
    "num_heads":     8,
    "signal_len":    6000,
    "snr_db_range":  (-15.0, 10.0),
    "noise_boost":   1.0,
    "max_samples":   5000,
    "val_ratio":     0.15,
    "batch_size":    1,
    "num_workers":   0,
    "seed":          42,
    "fs":            100,
    "num_plots":     50,
}

EPS = 1e-10

FREEZE_STRATEGIES = [
    {"name": "full_freeze",         "display": "全冻结（只训练输出层）"},
    {"name": "freeze_encoder",      "display": "冻结Encoder（训练FiLM+Decoder）"},
    {"name": "freeze_noise_encoder","display": "冻结NoiseEncoder（训练主干）"},
    {"name": "full_unfreeze",       "display": "全解冻（整体微调）"},
]

# ============================================================
# Dataset（与 xiaorong_v2 完全一致的预处理）
# ============================================================
class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, event_h5_path, event_csv_path,
                 noise_h5_path, noise_csv_path,
                 signal_len=6000, cond_len=400,
                 snr_db_range=(-15.0, 10.0),
                 noise_boost=1.0, max_samples=None, seed=42):
        self.event_h5_path = event_h5_path
        self.noise_h5_path = noise_h5_path
        self.signal_len    = signal_len
        self.cond_len      = cond_len
        self.snr_db_range  = snr_db_range
        self.noise_boost   = float(noise_boost)
        self.seed          = int(seed)
        self._ev_h5        = None
        self._no_h5        = None

        df = pd.read_csv(event_csv_path, low_memory=False)
        if max_samples is not None:
            df = df.iloc[:int(max_samples)].reset_index(drop=True)
        self.event_df = df
        self.noise_df = pd.read_csv(noise_csv_path, low_memory=False)

    @property
    def ev_h5(self):
        if self._ev_h5 is None:
            self._ev_h5 = h5py.File(self.event_h5_path, "r")
        return self._ev_h5

    @property
    def no_h5(self):
        if self._no_h5 is None:
            self._no_h5 = h5py.File(self.noise_h5_path, "r")
        return self._no_h5

    def __len__(self):
        return len(self.event_df)

    def _get_p_sample(self, row):
        for col in ["trace_P_arrival_sample", "Pg"]:
            if col in row.index and not pd.isna(row[col]):
                try:
                    return int(float(row[col]))
                except Exception:
                    pass
        return 2250

    def _load_event(self, h5f, name, p_sample, pre_p=500):
        x = h5f["data"][name][:]
        x = x.T.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        T = x.shape[1]
        start = max(0, p_sample - pre_p)
        end   = start + self.signal_len
        if end > T:
            end   = T
            start = max(0, end - self.signal_len)
        seg = x[:, start:end]
        if seg.shape[1] < self.signal_len:
            out = np.zeros((3, self.signal_len), dtype=np.float32)
            out[:, :seg.shape[1]] = seg
            seg = out
        p_rel = int(np.clip(p_sample - start, 0, self.signal_len - 1))
        return seg, p_rel

    def _load_noise(self, h5f, name):
        x = h5f["data"][name][:]
        x = x.T.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        T = x.shape[1]
        if T >= self.signal_len:
            return x[:, :self.signal_len]
        out = np.zeros((3, self.signal_len), dtype=np.float32)
        out[:, :T] = x
        return out

    @staticmethod
    def _norm_peak(x):
        # 与 xiaorong_v2 一致：直接峰值归一化，不去直流偏置
        m = np.abs(x).max()
        return x / m if m > 1e-10 else x

    @staticmethod
    def _mix_snr(clean, noise, snr_db):
        snr_lin = 10.0 ** (snr_db / 10.0)
        ps = np.mean(clean ** 2)
        pn = np.mean(noise ** 2)
        if ps < 1e-12 or pn < 1e-12:
            return clean.copy()
        scale = float(np.clip(np.sqrt(ps / (snr_lin * pn)), 0.0, 10.0))
        return clean + scale * noise

    def __getitem__(self, idx):
        row        = self.event_df.iloc[idx]
        trace_name = str(row["trace_name"])
        rng        = np.random.default_rng(self.seed + idx)
        p_sample   = self._get_p_sample(row)

        clean, p_rel = self._load_event(self.ev_h5, trace_name, p_sample)
        clean = self._norm_peak(clean)

        ni         = int(rng.integers(0, len(self.noise_df)))
        noise_name = str(self.noise_df.iloc[ni]["trace_name"])
        noise      = self._norm_peak(self._load_noise(self.no_h5, noise_name))

        snr_db_val = float(rng.uniform(self.snr_db_range[0], self.snr_db_range[1]))
        noisy_base = self._mix_snr(clean, noise, snr_db_val)
        # 与 xiaorong_v2 一致：noisy 不做额外去偏置/归一化
        noisy      = clean + self.noise_boost * (noisy_base - clean)

        z_cond = noise[:, :self.cond_len].copy()
        m = np.abs(z_cond).max()
        if m > 1e-10:
            z_cond = z_cond / m

        return {
            "clean":   torch.from_numpy(np.clip(clean, -10, 10).astype(np.float32)),
            "noisy":   torch.from_numpy(np.clip(noisy, -10, 10).astype(np.float32)),
            "z_cond":  torch.from_numpy(np.clip(z_cond, -10, 10).astype(np.float32)),
            "p_onset": p_rel,
            "snr_db":  snr_db_val,
        }

# ============================================================
# 指标函数
# ============================================================
def snr_db_fn(clean, test):
    sig = np.sum(clean ** 2)
    noi = np.sum((test - clean) ** 2)
    return float(10.0 * np.log10((sig + EPS) / (noi + EPS)))

def rmse_fn(clean, test):
    return float(np.sqrt(np.mean((clean - test) ** 2)))

def prd_fn(clean, test):
    return float(np.sqrt(
        np.sum((clean - test) ** 2) / (np.sum(clean ** 2) + EPS)
    ))

def compute_st_mae(clean, test, fs=100, win_ms=100.0, overlap=0.5):
    win_len = int(fs * win_ms / 1000)
    hop_len = max(1, int(win_len * (1 - overlap)))
    T = min(len(clean), len(test))
    times, st_mae = [], []
    start = 0
    while start + win_len <= T:
        end = start + win_len
        st_mae.append(np.abs(clean[start:end] - test[start:end]).mean())
        times.append((start + end) / 2 / fs)
        start += hop_len
    return np.array(times), np.array(st_mae)

# ============================================================
# 坐标轴样式
# ============================================================
def _style_ax(ax, ylabel="", title="", label_color="#333333",
              xlim=None, ylim=None, show_xlabel=False):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=label_color,
                      labelpad=4, rotation=90, va="center")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#aaaaaa")
        ax.spines[sp].set_linewidth(0.7)

# ============================================================
# 单样本绘图（同时保存 JPG 和 SVG）
# ============================================================
def plot_one_sample(x_noisy, y_clean, y_pred, p_onset, fs, save_stem,
                    sample_idx, snr_in, snr_out, strategy_display):
    """
    save_stem: 不含扩展名的完整路径，例如
               '.../full_freeze/sample_001_snrin-3.2dB'
    会自动保存为 .jpg 和 .svg
    """
    T = x_noisy.shape[1]
    t_axis = np.arange(T) / fs
    xlim = (0, T / fs)
    p_t  = p_onset / fs
    CH_NAMES = ["Z", "N", "E"]
    COLORS = {
        "noisy":    "#2ca02c",
        "clean":    "#d62728",
        "denoised": "#1a1a1a",
        "error":    "#1f77b4",
        "stmae_d":  "#ff7f0e",
        "stmae_n":  "#2ca02c",
    }
    ROW_LABELS = [
        "Noisy signal",
        "Clean signal",
        "Denoised signal",
        "Error (Denoised - Clean)",
        "ST-MAE",
    ]
    ROW_LABEL_COLORS = [
        COLORS["noisy"], COLORS["clean"], COLORS["denoised"],
        COLORS["error"], COLORS["stmae_d"],
    ]

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        f"Qualitative Denoising Performance — Sample #{sample_idx}"
        f"  [{strategy_display}]\n"
        f"Input SNR = {snr_in:.2f} dB   →   "
        f"Output SNR = {snr_out:.2f} dB   "
        f"(Gain = {snr_out - snr_in:+.2f} dB)",
        fontsize=12, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.42, wspace=0.22,
                           top=0.93, bottom=0.05, left=0.08, right=0.98)

    for col, ch in enumerate(CH_NAMES):
        noisy_ch    = x_noisy[col].astype(np.float64)
        clean_ch    = y_clean[col].astype(np.float64)
        denoised_ch = y_pred[col].astype(np.float64)
        error_ch    = denoised_ch - clean_ch

        # Row 0: Noisy
        ax0 = fig.add_subplot(gs[0, col])
        ax0.plot(t_axis, noisy_ch, color=COLORS["noisy"], linewidth=0.5, alpha=0.9)
        ax0.axvline(p_t, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
        _style_ax(ax0, ylabel=ROW_LABELS[0] if col == 0 else "",
                  title=f"Channel {ch}", label_color=ROW_LABEL_COLORS[0],
                  xlim=xlim, ylim=(-1.1, 1.1))

        # Row 1: Clean
        ax1 = fig.add_subplot(gs[1, col])
        ax1.plot(t_axis, clean_ch, color=COLORS["clean"], linewidth=0.5, alpha=0.9)
        ax1.axvline(p_t, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
        _style_ax(ax1, ylabel=ROW_LABELS[1] if col == 0 else "",
                  label_color=ROW_LABEL_COLORS[1], xlim=xlim, ylim=(-1.1, 1.1))

        # Row 2: Denoised
        ax2 = fig.add_subplot(gs[2, col])
        ax2.plot(t_axis, denoised_ch, color=COLORS["denoised"], linewidth=0.5, alpha=0.9)
        ax2.axvline(p_t, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
        snr_ch  = snr_db_fn(clean_ch, denoised_ch)
        rmse_ch = rmse_fn(clean_ch, denoised_ch)
        prd_ch  = prd_fn(clean_ch, denoised_ch)
        ax2.set_title(f"SNR={snr_ch:.1f}dB  RMSE={rmse_ch:.4f}  PRD={prd_ch:.3f}",
                      fontsize=7.5, color="#333333", pad=3)
        _style_ax(ax2, ylabel=ROW_LABELS[2] if col == 0 else "",
                  label_color=ROW_LABEL_COLORS[2], xlim=xlim, ylim=(-1.1, 1.1))

        # Row 3: Error
        ax3 = fig.add_subplot(gs[3, col])
        ax3.plot(t_axis, error_ch, color=COLORS["error"], linewidth=0.45, alpha=0.85)
        ax3.axhline(0, color="#aaaaaa", linewidth=0.6)
        ax3.axvline(p_t, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
        _style_ax(ax3, ylabel=ROW_LABELS[3] if col == 0 else "",
                  label_color=ROW_LABEL_COLORS[3], xlim=xlim, ylim=(-0.25, 0.25))

        # Row 4: ST-MAE
        ax4 = fig.add_subplot(gs[4, col])
        t_n, stmae_n = compute_st_mae(clean_ch, noisy_ch, fs)
        t_d, stmae_d = compute_st_mae(clean_ch, denoised_ch, fs)
        ax4.plot(t_n, stmae_n, color=COLORS["stmae_n"], linewidth=0.6,
                 alpha=0.75, label="Noisy")
        ax4.plot(t_d, stmae_d, color=COLORS["stmae_d"], linewidth=0.8,
                 alpha=0.95, label="Denoised")
        ax4.axvline(p_t, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
        ymax = max(
            stmae_n.max() if len(stmae_n) > 0 else 0.06,
            stmae_d.max() if len(stmae_d) > 0 else 0.06,
        ) * 1.15
        ymax = max(ymax, 0.06)
        if col == 0:
            ax4.legend(fontsize=7, loc="upper right", framealpha=0.5, handlelength=1.5)
        _style_ax(ax4, ylabel=ROW_LABELS[4] if col == 0 else "",
                  label_color=ROW_LABEL_COLORS[4], xlim=xlim,
                  ylim=(0, ymax), show_xlabel=True)

    # ── 同时保存 JPG 和 SVG ──────────────────────────────────
    path_jpg = save_stem + ".jpg"
    path_svg = save_stem + ".svg"
    plt.savefig(path_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.savefig(path_svg, bbox_inches="tight", format="svg")
    plt.close(fig)
    print(f"  [✓] JPG: {path_jpg}")
    print(f"  [✓] SVG: {path_svg}")

# ============================================================
# 主函数
# ============================================================
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    full_ds = FinetuneDataset(
        event_h5_path  = CONFIG["finetune_h5"],
        event_csv_path = CONFIG["finetune_csv"],
        noise_h5_path  = CONFIG["noise_h5"],
        noise_csv_path = CONFIG["noise_csv"],
        signal_len     = CONFIG["signal_len"],
        cond_len       = CONFIG["cond_len"],
        snr_db_range   = CONFIG["snr_db_range"],
        noise_boost    = CONFIG["noise_boost"],
        max_samples    = CONFIG["max_samples"],
        seed           = CONFIG["seed"],
    )

    n_val   = max(1, int(len(full_ds) * CONFIG["val_ratio"]))
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )
    print(f"[INFO] 验证集: {len(val_ds)} 条")

    # 预先缓存固定波形
    num_plots = CONFIG["num_plots"]
    fixed_samples = []
    for idx in range(min(num_plots, len(val_ds))):
        batch = val_ds[idx]
        fixed_samples.append({
            "noisy":   batch["noisy"].unsqueeze(0),
            "clean":   batch["clean"].unsqueeze(0),
            "z_cond":  batch["z_cond"].unsqueeze(0),
            "p_onset": int(batch["p_onset"]),
        })
    print(f"[INFO] 已缓存 {len(fixed_samples)} 条固定波形")

    for strategy in FREEZE_STRATEGIES:
        sname   = strategy["name"]
        display = strategy["display"]
        ckpt    = os.path.join(CONFIG["exp2_dir"], f"best_{sname}.pth")

        print(f"\n{'='*60}")
        print(f"[策略] {display}")

        if not os.path.exists(ckpt):
            print(f"  ⚠ 权重不存在: {ckpt}，跳过")
            continue

        model = NoiseAwareDenoiserV3(
            z_dim     = CONFIG["z_dim"],
            cond_len  = CONFIG["cond_len"],
            num_heads = CONFIG["num_heads"],
        ).to(device)

        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()

        # 分别建 jpg/ 和 svg/ 子目录
        out_dir_jpg = os.path.join(CONFIG["output_dir"], sname, "jpg")
        out_dir_svg = os.path.join(CONFIG["output_dir"], sname, "svg")
        os.makedirs(out_dir_jpg, exist_ok=True)
        os.makedirs(out_dir_svg, exist_ok=True)

        snr_in_list, snr_out_list = [], []

        with torch.no_grad():
            for i, sample in enumerate(fixed_samples):
                noisy   = sample["noisy"].to(device)
                clean   = sample["clean"].to(device)
                z_cond  = sample["z_cond"].to(device)
                p_onset = sample["p_onset"]

                if not all(torch.isfinite(t).all() for t in [noisy, clean, z_cond]):
                    print(f"  ⚠ 样本 #{i+1} 含 NaN/Inf，跳过")
                    continue

                try:
                    pred, _, _ = model(noisy, z_cond)
                except Exception as e:
                    print(f"  ⚠ 推理异常 #{i+1}: {e}")
                    continue

                if not torch.isfinite(pred).all():
                    print(f"  ⚠ 输出含 NaN/Inf，跳过 #{i+1}")
                    continue

                x_np    = noisy[0].cpu().numpy()
                y_np    = clean[0].cpu().numpy()
                pred_np = pred[0].cpu().numpy()

                snr_in  = snr_db_fn(y_np[2], x_np[2])
                snr_out = snr_db_fn(y_np[2], pred_np[2])
                snr_in_list.append(snr_in)
                snr_out_list.append(snr_out)

                # stem 放在 jpg 目录下，svg 目录单独构建
                fname = f"sample_{i+1:03d}_snrin{snr_in:.1f}dB"
                # 传入不含扩展名的 stem，函数内部自动保存两份
                # 两种格式放同一子目录（sname/）
                save_stem = os.path.join(
                    CONFIG["output_dir"], sname, fname
                )
                plot_one_sample(
                    x_noisy          = x_np,
                    y_clean          = y_np,
                    y_pred           = pred_np,
                    p_onset          = p_onset,
                    fs               = CONFIG["fs"],
                    save_stem        = save_stem,
                    sample_idx       = i + 1,
                    snr_in           = snr_in,
                    snr_out          = snr_out,
                    strategy_display = display,
                )

        if snr_in_list:
            gains = np.array(snr_out_list) - np.array(snr_in_list)
            print(f"  生成图数:    {len(snr_in_list)}")
            print(f"  Input  SNR:  {np.mean(snr_in_list):.2f} dB")
            print(f"  Output SNR:  {np.mean(snr_out_list):.2f} dB")
            print(f"  SNR Gain:    {gains.mean():+.2f} dB")

    print(f"\n[✅ 完成] 图像保存至: {CONFIG['output_dir']}")
    print(f"  每个策略目录下同时包含 .jpg 和 .svg 文件")

if __name__ == "__main__":
    main()