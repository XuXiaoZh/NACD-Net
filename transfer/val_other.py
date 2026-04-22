# -*- coding: utf-8 -*-
# D:\X\denoise\part1\v3\transfer\val_baselines.py
"""
baseline 模型验证脚本：DeepDenoiser / DPRNN
先加噪再降噪，输出 metrics.csv + SVG 三列对比图
用chunk2后100条数据
用法：
    python val_baselines.py --model deepdenoiser
    python val_baselines.py --model dprnn
"""

import os
import sys
import json
import glob
import argparse
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.serif"]  = ["Times New Roman"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["svg.fonttype"] = "none"

# ---------------- 路径 ----------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR   = os.path.abspath(os.path.join(THIS_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
for p in [V3_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from baeslines.deep_denoiser import DeepDenoiser
from baeslines.dprnn import DPRNN

# ============================================================
# 配置
# ============================================================
CONFIG = {
    # 数据路径
    "event_h5":  r"D:/X/p_wave/data/chunk2.hdf5",
    "event_csv": r"D:/X/p_wave/data/chunk2.csv",
    "noise_h5":  r"D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv": r"D:/X/p_wave/data/chunk1.csv",

    # 测试样本范围
    "use_tail_n":  1000,
    "max_samples": 200,

    # 加噪参数
    "snr_db_range": (-10.0, 0.0),
    "fixed_snr_db": None,
    "noise_boost":  1.0,

    # 模型 checkpoint（key = model_name）
    "ckpt_paths": {
        "deepdenoiser": r"D:/X/denoise/part1/v3/baeslines/checkpoints/deep_denoiser/best_model.pth",
        "dprnn":        r"D:/X/denoise/part1/v3/baeslines/checkpoints/dprnn/best_model.pth",
    },

    # 信号参数
    "signal_len": 6000,
    "cond_len":   400,

    # 推理
    "batch_size":  8,
    "num_workers": 0,
    "seed":        42,

    # 输出根目录（子目录按模型名自动创建）
    "out_root": r"D:/X/denoise/part1/v3/baeslines/val_baseline_outputs",

    # SVG
    "save_svg_num": 50,
    "fs": None,

    "save_fmt": "png",   # "svg" 或 "png"
}

# ============================================================
# Dataset（与 val_addnoise.py 完全一致）
# ============================================================
class AddNoiseEvalDataset(Dataset):
    def __init__(
        self,
        event_h5_path, event_csv_path,
        noise_h5_path,  noise_csv_path,
        signal_len=6000, cond_len=400,
        snr_db_range=(-10.0, 0.0), fixed_snr_db=None,
        noise_boost=1.0, use_tail_n=None, max_samples=None, seed=42,
    ):
        super().__init__()
        self.event_h5_path = event_h5_path
        self.noise_h5_path = noise_h5_path
        self.signal_len    = signal_len
        self.cond_len      = cond_len
        self.snr_db_range  = snr_db_range
        self.fixed_snr_db  = fixed_snr_db
        self.noise_boost   = float(noise_boost)
        self.seed          = int(seed)

        self.event_df = pd.read_csv(event_csv_path, low_memory=False)
        self.noise_df = pd.read_csv(noise_csv_path,  low_memory=False)

        if use_tail_n is not None:
            self.event_df = self.event_df.tail(int(use_tail_n)).reset_index(drop=True)
        if max_samples is not None:
            self.event_df = self.event_df.iloc[:int(max_samples)].reset_index(drop=True)

        self._event_h5 = None
        self._noise_h5 = None
        print(f"[Dataset] events={len(self.event_df)}, noises={len(self.noise_df)}")

    @property
    def event_h5(self):
        if self._event_h5 is None:
            self._event_h5 = h5py.File(self.event_h5_path, "r")
        return self._event_h5

    @property
    def noise_h5(self):
        if self._noise_h5 is None:
            self._noise_h5 = h5py.File(self.noise_h5_path, "r")
        return self._noise_h5

    def __len__(self):
        return len(self.event_df)

    def _load(self, h5f, trace_name):
        x = h5f["data"][trace_name][:]
        x = x.T.astype(np.float32)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _pad_or_crop(self, x):
        T = x.shape[1]
        if T >= self.signal_len:
            return x[:, :self.signal_len]
        out = np.zeros((3, self.signal_len), dtype=np.float32)
        out[:, :T] = x
        return out

    @staticmethod
    def _normalize_peak(x):
        m = np.abs(x).max()
        return (x / m, m) if m > 1e-10 else (x, 1.0)

    @staticmethod
    def _mix_snr_db(clean_n, noise_n, snr_db):
        snr_lin = 10.0 ** (snr_db / 10.0)
        ps = np.mean(clean_n ** 2)
        pn = np.mean(noise_n ** 2)
        if ps < 1e-12 or pn < 1e-12:
            return clean_n.copy(), 0.0
        scale = float(np.clip(np.sqrt(ps / (snr_lin * pn)), 0.0, 10.0))
        return clean_n + scale * noise_n, scale

    def _get_p_onset(self, row):
        for col in ["p_arrival_sample", "p_onset", "itp"]:
            if col in row.index and not pd.isna(row[col]):
                try:
                    v = int(row[col])
                    if 0 <= v < self.signal_len:
                        return v
                except Exception:
                    pass
        return self.signal_len // 10

    def __getitem__(self, idx):
        row = self.event_df.iloc[idx]
        trace_name = str(row["trace_name"])
        rng = np.random.default_rng(self.seed + idx)

        clean   = self._pad_or_crop(self._load(self.event_h5, trace_name))
        clean_n, _ = self._normalize_peak(clean)

        ni = int(rng.integers(0, len(self.noise_df)))
        noise_name = str(self.noise_df.iloc[ni]["trace_name"])
        noise   = self._pad_or_crop(self._load(self.noise_h5, noise_name))
        noise_n, _ = self._normalize_peak(noise)

        snr_db = float(self.fixed_snr_db) if self.fixed_snr_db is not None \
                 else float(rng.uniform(self.snr_db_range[0], self.snr_db_range[1]))

        noisy_base, _ = self._mix_snr_db(clean_n, noise_n, snr_db)
        noisy_n = clean_n + self.noise_boost * (noisy_base - clean_n)

        z_cond = noise_n[:, :self.cond_len].copy()
        zc_m = np.abs(z_cond).max()
        if zc_m > 1e-10:
            z_cond = z_cond / zc_m

        p_onset = self._get_p_onset(row)
        valid_mask = np.zeros(self.signal_len, dtype=np.float32)
        valid_mask[p_onset:min(p_onset + 4000, self.signal_len)] = 1.0

        return {
            "clean":      torch.from_numpy(np.clip(clean_n, -10, 10).astype(np.float32)),
            "noisy":      torch.from_numpy(np.clip(noisy_n, -10, 10).astype(np.float32)),
            "z_cond":     torch.from_numpy(np.clip(z_cond,  -10, 10).astype(np.float32)),
            "valid_mask": torch.from_numpy(valid_mask),
            "trace_name": trace_name,
            "noise_name": noise_name,
            "snr_db":     torch.tensor(snr_db, dtype=torch.float32),
        }

# ============================================================
# 工具函数
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def find_checkpoint(ckpt_path: str) -> str:
    if os.path.exists(ckpt_path):
        return ckpt_path
    ckpt_dir = os.path.dirname(ckpt_path)
    cands = glob.glob(os.path.join(ckpt_dir, "ckpt_epoch*.pth"))
    if not cands:
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    cands.sort(key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or "-1"))
    return cands[-1]

def safe_unique_name(name: str, used: set):
    base = str(name).replace("/", "_").replace("\\", "_")
    if base not in used:
        used.add(base)
        return base
    i = 1
    while f"{base}_{i}" in used:
        i += 1
    used.add(f"{base}_{i}")
    return f"{base}_{i}"

def compute_snr_db(clean, residual, valid_mask):
    """clean/residual: [B,3,T], valid_mask: [B,T]"""
    mask = valid_mask.unsqueeze(1)
    n    = mask.sum(dim=[1, 2]) * clean.shape[1] + 1e-10
    sig  = ((clean ** 2) * mask).sum(dim=[1, 2]) / n
    noi  = ((residual ** 2) * mask).sum(dim=[1, 2]) / n + 1e-10
    return torch.clamp(10.0 * torch.log10(sig / noi), -50, 50)

def save_triplet_svg(clean_3t, noisy_3t, deno_3t, out_svg, title="", fs=None):
    ensure_dir(os.path.dirname(out_svg))
    ch_names = ["E", "N", "Z"]
    T = clean_3t.shape[-1]
    t = np.arange(T) if fs is None else np.arange(T) / float(fs)
    xlab = "sample" if fs is None else "time (s)"

    fig, axes = plt.subplots(3, 3, figsize=(16, 8), sharex=True)
    for j, col_title in enumerate(["Clean", "Noisy", "Denoised"]):
        axes[0, j].set_title(col_title, fontsize=11)

    for i in range(3):
        ymax = max(np.max(np.abs(clean_3t[i])), np.max(np.abs(noisy_3t[i])),
                   np.max(np.abs(deno_3t[i])), 1e-6)
        for j, sig in enumerate([clean_3t[i], noisy_3t[i], deno_3t[i]]):
            axes[i, j].plot(t, sig, lw=0.8, color="#1f77b4")
            axes[i, j].set_ylim(-ymax, ymax)
            axes[i, j].grid(alpha=0.25, linestyle="--")
            if j == 0:
                axes[i, j].set_ylabel(ch_names[i])
    for j in range(3):
        axes[2, j].set_xlabel(xlab)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)


def save_triplet_figure(clean_3t, noisy_3t, deno_3t, out_path, title="", fs=None, fmt="svg"):
    ensure_dir(os.path.dirname(out_path))
    ch_names = ["E", "N", "Z"]
    T = clean_3t.shape[-1]
    t = np.arange(T) if fs is None else np.arange(T) / float(fs)
    xlab = "sample" if fs is None else "time (s)"

    fig, axes = plt.subplots(3, 3, figsize=(16, 8), sharex=True)
    for j, col_title in enumerate(["Clean", "Noisy", "Denoised"]):
        axes[0, j].set_title(col_title, fontsize=11)

    for i in range(3):
        ymax = max(np.max(np.abs(clean_3t[i])), np.max(np.abs(noisy_3t[i])),
                   np.max(np.abs(deno_3t[i])), 1e-6)
        for j, sig in enumerate([clean_3t[i], noisy_3t[i], deno_3t[i]]):
            axes[i, j].plot(t, sig, lw=0.8, color="#1f77b4")
            axes[i, j].set_ylim(-ymax, ymax)
            axes[i, j].grid(alpha=0.25, linestyle="--")
            if j == 0:
                axes[i, j].set_ylabel(ch_names[i])
    for j in range(3):
        axes[2, j].set_xlabel(xlab)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, format=fmt, dpi=150 if fmt == "png" else None)
    plt.close(fig)


# ============================================================
# 模型工厂
# ============================================================
def build_model(model_name: str):
    if model_name == "deepdenoiser":
        return DeepDenoiser(in_ch=3)
    elif model_name == "dprnn":
        return DPRNN(in_ch=3)
    else:
        raise ValueError(f"未知模型: {model_name}，可选 deepdenoiser / dprnn")

# ============================================================
# 主流程
# ============================================================
def main(model_name: str):
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # 输出目录
    out_dir  = os.path.join(CONFIG["out_root"], model_name)
    svg_dir  = os.path.join(out_dir, "svg_triplets")
    out_h5   = os.path.join(out_dir, "clean_noisy_denoised.hdf5")
    metrics_csv  = os.path.join(out_dir, "metrics.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    ensure_dir(out_dir)
    ensure_dir(svg_dir)

    for k in ["event_h5", "event_csv", "noise_h5", "noise_csv"]:
        if not os.path.exists(CONFIG[k]):
            raise FileNotFoundError(f"{k} not found: {CONFIG[k]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] model={model_name}  device={device}")

    # 模型加载
    model = build_model(model_name).to(device)
    ckpt_path = find_checkpoint(CONFIG["ckpt_paths"][model_name])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[INFO] Loaded ckpt: {ckpt_path}")

    # 数据
    ds = AddNoiseEvalDataset(
        event_h5_path=CONFIG["event_h5"],
        event_csv_path=CONFIG["event_csv"],
        noise_h5_path=CONFIG["noise_h5"],
        noise_csv_path=CONFIG["noise_csv"],
        signal_len=CONFIG["signal_len"],
        cond_len=CONFIG["cond_len"],
        snr_db_range=CONFIG["snr_db_range"],
        fixed_snr_db=CONFIG["fixed_snr_db"],
        noise_boost=CONFIG["noise_boost"],
        use_tail_n=CONFIG["use_tail_n"],
        max_samples=CONFIG["max_samples"],
        seed=CONFIG["seed"],
    )
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=CONFIG["num_workers"])

    used_names = set()
    metrics    = []
    svg_saved  = 0

    with h5py.File(out_h5, "w") as f:
        g_clean = f.create_group("clean")
        g_noisy = f.create_group("noisy")
        g_deno  = f.create_group("denoised")

        with torch.no_grad():
            for batch in loader:
                clean      = batch["clean"].to(device)       # [B,3,T]
                noisy      = batch["noisy"].to(device)
                z_cond     = batch["z_cond"].to(device)
                valid_mask = batch["valid_mask"].to(device)
                names      = batch["trace_name"]
                noise_names= batch["noise_name"]
                snr_db_arr = batch["snr_db"].cpu().numpy()

                # 两个模型都接受 z_cond 但会忽略它
                denoised, _, _ = model(noisy, z_cond)

                snr_in  = compute_snr_db(clean, noisy - clean,    valid_mask).cpu().numpy()
                snr_out = compute_snr_db(clean, denoised - clean, valid_mask).cpu().numpy()
                snr_gain = snr_out - snr_in

                clean_np = clean.cpu().numpy()
                noisy_np = noisy.cpu().numpy()
                deno_np  = denoised.cpu().numpy()

                for i in range(clean_np.shape[0]):
                    uname = safe_unique_name(str(names[i]), used_names)

                    g_clean.create_dataset(uname, data=clean_np[i].T.astype(np.float32), compression="gzip")
                    g_noisy.create_dataset(uname, data=noisy_np[i].T.astype(np.float32), compression="gzip")
                    g_deno.create_dataset( uname, data=deno_np[i].T.astype(np.float32),  compression="gzip")

                    metrics.append({
                        "trace_name":      str(names[i]),
                        "noise_trace_name":str(noise_names[i]),
                        "snr_set_db":      float(snr_db_arr[i]),
                        "input_snr_db":    float(snr_in[i]),
                        "output_snr_db":   float(snr_out[i]),
                        "snr_gain_db":     float(snr_gain[i]),
                    })

                    if svg_saved < CONFIG["save_svg_num"]:
                        title_str = f"{names[i]} | SNR_set={snr_db_arr[i]:.2f} dB | SNR_gain={snr_gain[i]:.2f} dB"

                        # 原有 svg
                        save_triplet_svg(
                            clean_np[i], noisy_np[i], deno_np[i],
                            out_svg=os.path.join(svg_dir, f"{uname}_triplet.svg"),
                            title=title_str,
                            fs=CONFIG["fs"],
                        )

                        # 额外保存 png
                        save_triplet_figure(
                            clean_np[i], noisy_np[i], deno_np[i],
                            out_path=os.path.join(svg_dir, f"{uname}_triplet.png"),
                            title=title_str,
                            fs=CONFIG["fs"],
                            fmt="png",
                        )

                        svg_saved += 1  # 只加一次
    mdf = pd.DataFrame(metrics)
    mdf.to_csv(metrics_csv, index=False)

    summary = {
        "model":               model_name,
        "n_samples":           int(len(mdf)),
        "snr_set_db_mean":     float(mdf["snr_set_db"].mean()),
        "input_snr_db_mean":   float(mdf["input_snr_db"].mean()),
        "output_snr_db_mean":  float(mdf["output_snr_db"].mean()),
        "snr_gain_db_mean":    float(mdf["snr_gain_db"].mean()),
        "snr_gain_db_median":  float(mdf["snr_gain_db"].median()),
        "fixed_snr_db":        CONFIG["fixed_snr_db"],
        "snr_db_range":        list(CONFIG["snr_db_range"]),
        "noise_boost":         CONFIG["noise_boost"],
        "svg_saved":           int(svg_saved),
        "ckpt_used":           ckpt_path,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n========== {model_name} 验证完成 ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["deepdenoiser", "dprnn"],
                        help="选择要验证的模型")
    args = parser.parse_args()
    main(args.model)