# -*- coding: utf-8 -*-
"""
迁移学习评估脚本 evaluate_transfer.py
"""

import os, sys, h5py
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.signal import resample_poly
from torch.utils.data import Dataset, DataLoader, random_split

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [THIS_DIR,
          os.path.abspath(os.path.join(THIS_DIR, "..")),
          os.path.abspath(os.path.join(THIS_DIR, "..", ".."))]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from model_v3 import NoiseAwareDenoiserV3
except ModuleNotFoundError:
    from model_v3 import NoiseAwareDenoiserV3

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "ckpt_path":    r"D:/X/denoise/part1/v3/checkpoints_transfer_15k/best_transfer_15k.pth",
    "finetune_h5":  r"D:/X/p_wave/data/non_naturaldata.hdf5",
    "finetune_csv": r"D:/X/p_wave/data/non_naturaldata.csv",
    "noise_h5":     r"D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv":    r"D:/X/p_wave/data/chunk1.csv",
    "output_dir":   r"D:/X/denoise/part1/v3/eval_transfer_15k",
    "max_samples":  15000,
    "val_ratio":    0.1,
    "signal_len":   6000,
    "cond_len":     400,
    "z_dim":        128,
    "num_heads":    8,
    "snr_db_range": (-15.0, 10.0),
    "noise_boost":  1.0,
    "fs":           100,
    "sta_len":      0.5,
    "lta_len":      10.0,
    "stalta_thr":   2.0,
    "pick_tol":     50,
    "batch_size":   16,
    "num_workers":  0,
    "seed":         42,
    "save_fig_num": 12,
}

EPS        = 1e-10
SNR_BINS   = [-np.inf, -5, 0, 5, 10, 15, np.inf]
SNR_LABELS = ["<-5", "-5~0", "0~5", "5~10", "10~15", ">15"]

# ============================================================
# Dataset
# ============================================================
class EvalDataset(Dataset):
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
        self._last_offset  = 0

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

    def _load_event(self, h5f, name):
        x = h5f["data"][name][:]
        x = x.T.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = resample_poly(x, up=2, down=1, axis=1).astype(np.float32)
        T, start, end = x.shape[1], 4000, 10000
        if T >= end:
            self._last_offset = start
            return x[:, start:end]
        elif T > start:
            self._last_offset = start
            seg = x[:, start:T]
            out = np.zeros((3, self.signal_len), dtype=np.float32)
            out[:, :seg.shape[1]] = seg
            return out
        else:
            self._last_offset = 0
            out = np.zeros((3, self.signal_len), dtype=np.float32)
            out[:, :min(T, self.signal_len)] = x[:, :self.signal_len]
            return out

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
        # 不重采样
        T = x.shape[1]
        start = max(0, p_sample - pre_p)
        end = start + self.signal_len
        if end > T:
            end = T
            start = max(0, end - self.signal_len)
        seg = x[:, start:end]
        if seg.shape[1] < self.signal_len:
            out = np.zeros((3, self.signal_len), dtype=np.float32)
            out[:, :seg.shape[1]] = seg
            seg = out
        p_rel = int(np.clip(p_sample - start, 0, self.signal_len - 1))
        self._last_p_rel = p_rel
        return seg

    def __getitem__(self, idx):
        row = self.event_df.iloc[idx]
        trace_name = str(row["trace_name"])
        rng = np.random.default_rng(self.seed + idx)

        p_sample = self._get_p_sample(row)
        clean = self._norm_peak(self._load_event(self.ev_h5, trace_name, p_sample))
        p_rel = self._last_p_rel  # 截取后的相对位置，这才是正确的 p_onset

        ni = int(rng.integers(0, len(self.noise_df)))
        noise_name = str(self.noise_df.iloc[ni]["trace_name"])
        noise = self._norm_peak(self._load_noise(self.no_h5, noise_name))

        snr_db_val = float(rng.uniform(self.snr_db_range[0], self.snr_db_range[1]))
        noisy_base = self._mix_snr(clean, noise, snr_db_val)
        noisy = clean + self.noise_boost * (noisy_base - clean)

        z_cond = noise[:, :self.cond_len].copy()
        m = np.abs(z_cond).max()
        if m > 1e-10:
            z_cond = z_cond / m

        return {
            "clean": torch.from_numpy(np.clip(clean, -10, 10).astype(np.float32)),
            "noisy": torch.from_numpy(np.clip(noisy, -10, 10).astype(np.float32)),
            "z_cond": torch.from_numpy(np.clip(z_cond, -10, 10).astype(np.float32)),
            "p_onset": p_rel,
            "trace_name": trace_name,
            "snr_db": snr_db_val,
        }
# ============================================================
# 指标函数
# ============================================================
def snr_db(clean, test):
    sig = np.sum(clean ** 2)
    noi = np.sum((test - clean) ** 2)
    return float(10.0 * np.log10((sig + EPS) / (noi + EPS)))

def cc_fn(clean, test):
    c = clean - clean.mean()
    t = test  - test.mean()
    d = np.sqrt(np.sum(c**2) * np.sum(t**2))
    return float(np.sum(c * t) / d) if d > EPS else 0.0

def rmse_fn(clean, test):
    return float(np.sqrt(np.mean((clean - test) ** 2)))

def prd_fn(clean, test):
    return float(np.sqrt(np.sum((clean - test) ** 2) / (np.sum(clean ** 2) + EPS)))

def st_mae_mean(clean, test, fs=100, win_ms=100, overlap=0.5):
    n   = min(len(clean), len(test))
    win = min(int(fs * win_ms / 1000), n)
    if win <= 0:
        return float("nan")
    hop  = max(1, int(win * (1.0 - overlap)))
    vals = [np.abs(clean[s:s+win] - test[s:s+win]).mean()
            for s in range(0, n - win + 1, hop)]
    return float(np.mean(vals)) if vals else float("nan")

def assign_snr_group(snr_val):
    for i in range(len(SNR_BINS) - 1):
        if SNR_BINS[i] <= snr_val < SNR_BINS[i + 1]:
            return SNR_LABELS[i]
    return SNR_LABELS[-1]

# ============================================================
# STA/LTA
# ============================================================
def stalta_pick(wave, fs, sta_len, lta_len, threshold):
    x    = wave[2].astype(np.float64) if wave.ndim == 2 else wave.astype(np.float64)
    nsta = int(sta_len * fs)
    nlta = int(lta_len * fs)
    T    = len(x)
    if T < nlta + nsta:
        return -1
    cf  = x ** 2
    cs  = np.cumsum(np.concatenate([[0.0], cf]))
    i0, i1 = nlta, T - nsta
    if i0 >= i1:
        return -1
    idx   = np.arange(i0, i1)
    lta   = (cs[idx]        - cs[idx - nlta]) / nlta
    sta   = (cs[idx + nsta] - cs[idx])        / nsta
    valid = lta > EPS
    ratio = np.zeros(len(idx))
    ratio[valid] = sta[valid] / lta[valid]
    trig  = np.where(ratio > threshold)[0]
    return int(trig[0] + i0) if len(trig) > 0 else -1

# ============================================================
# 评估主循环
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    records          = []
    quality_list     = []
    pick_err_noisy   = []
    pick_err_denoise = []
    pick_suc_noisy   = []
    pick_suc_denoise = []
    skip             = 0

    for batch in loader:
        noisy   = batch["noisy"].to(device)
        clean   = batch["clean"].to(device)
        z_cond  = batch["z_cond"].to(device)
        p_onset = batch["p_onset"]
        snr_in  = batch["snr_db"]

        if not all(torch.isfinite(t).all() for t in [noisy, clean, z_cond]):
            skip += 1
            continue

        try:
            pred, quality, _ = model(noisy, z_cond)
        except Exception as e:
            skip += 1
            if skip <= 3:
                print(f"  ⚠ forward error: {e}")
            continue

        if not torch.isfinite(pred).all():
            skip += 1
            continue

        for i in range(noisy.shape[0]):
            x_np    = noisy[i].cpu().numpy()
            y_np    = clean[i].cpu().numpy()
            p_np    = pred[i].cpu().numpy()
            qual    = float(quality[i].item())
            p_true  = int(p_onset[i].item())
            snr_i   = float(snr_in[i].item())

            x_z = x_np[2].astype(np.float64)
            y_z = y_np[2].astype(np.float64)
            p_z = p_np[2].astype(np.float64)

            snr_o = snr_db(y_z, p_z)
            group = assign_snr_group(snr_i)

            records.append({
                "snr_in":          snr_i,
                "snr_out":         snr_o,
                "delta_snr":       snr_o - snr_i,
                "cc":              cc_fn(y_z, p_z),
                "rmse":            rmse_fn(y_z, p_z),
                "prd":             prd_fn(y_z, p_z),
                "st_mae_noisy":    st_mae_mean(y_z, x_z,  CONFIG["fs"]),
                "st_mae_denoised": st_mae_mean(y_z, p_z,  CONFIG["fs"]),
                "quality":         qual,
                "snr_group":       group,
            })

            tol    = CONFIG["pick_tol"]
            fs     = CONFIG["fs"]
            kw     = dict(fs=fs, sta_len=CONFIG["sta_len"],
                          lta_len=CONFIG["lta_len"],
                          threshold=CONFIG["stalta_thr"])

            pick_n = stalta_pick(x_np, **kw)
            pred_pick = p_np.copy()
            peak = np.abs(pred_pick).max()
            if peak > 1e-10:
                pred_pick = pred_pick / peak
            pick_d = stalta_pick(pred_pick, **kw)
            # pick_d = stalta_pick(p_np, **kw)

            if pick_n >= 0:
                err_n = abs(pick_n - p_true)
                pick_err_noisy.append(float(err_n))
                pick_suc_noisy.append(err_n <= tol)
            else:
                pick_err_noisy.append(float("nan"))
                pick_suc_noisy.append(False)

            if pick_d >= 0:
                err_d = abs(pick_d - p_true)
                pick_err_denoise.append(float(err_d))
                pick_suc_denoise.append(err_d <= tol)
            else:
                pick_err_denoise.append(float("nan"))
                pick_suc_denoise.append(False)

            quality_list.append(qual)

    if skip > 0:
        print(f"  ⚠ 跳过 {skip} 个异常样本")

    pick_results = {
        "quality":            np.array(quality_list),
        "pick_err_noisy":     np.array(pick_err_noisy),
        "pick_err_denoise":   np.array(pick_err_denoise),
        "pick_suc_noisy":     np.array(pick_suc_noisy,  dtype=bool),
        "pick_suc_denoise":   np.array(pick_suc_denoise, dtype=bool),
    }
    return records, pick_results

# ============================================================
# 统计 & 打印
# ============================================================
def summarize(records):
    df = pd.DataFrame(records)
    summary = {}
    for col in ["delta_snr", "cc", "rmse", "prd",
                "st_mae_noisy", "st_mae_denoised", "quality"]:
        summary[col] = float(df[col].mean())
    summary["total"] = len(df)

    print(f"\n{'='*65}")
    print("  迁移学习模型评估结果")
    print(f"{'='*65}")
    print(f"  总样本数   : {summary['total']}")
    print(f"  ΔSNR (dB)  : {summary['delta_snr']:+.4f}")
    print(f"  CC         : {summary['cc']:.4f}")
    print(f"  RMSE       : {summary['rmse']:.4f}")
    print(f"  PRD        : {summary['prd']:.4f}")
    print(f"  ST-MAE(n)  : {summary['st_mae_noisy']:.4f}")
    print(f"  ST-MAE(d)  : {summary['st_mae_denoised']:.4f}")
    print(f"  Quality    : {summary['quality']:.4f}")

    print(f"\n  {'Group':>7} | {'N':>4} | {'SNR_in':>7} → {'SNR_out':>7} "
          f"({'ΔSNR':>6}) | {'CC':>6} | {'RMSE':>7} | {'Quality':>7}")
    print(f"  {'-'*75}")
    for label in SNR_LABELS:
        sub = df[df["snr_group"] == label]
        if sub.empty:
            continue
        print(f"  {label:>7} | {len(sub):>4} | "
              f"{sub['snr_in'].mean():>7.2f} → {sub['snr_out'].mean():>7.2f} "
              f"({sub['delta_snr'].mean():>+6.2f}) | "
              f"{sub['cc'].mean():>6.4f} | {sub['rmse'].mean():>7.4f} | "
              f"{sub['quality'].mean():>7.4f}")
    return df, summary

def print_pick_stats(pick_results):
    q   = pick_results["quality"]
    en  = pick_results["pick_err_noisy"]
    ed  = pick_results["pick_err_denoise"]
    sn  = pick_results["pick_suc_noisy"]
    sd  = pick_results["pick_suc_denoise"]
    n   = len(q)
    tol = CONFIG["pick_tol"]

    rate_n = sn.sum() / n if n > 0 else 0.0
    rate_d = sd.sum() / n if n > 0 else 0.0
    mae_n  = float(en[np.isfinite(en) & sn].mean()) if (np.isfinite(en) & sn).sum() > 0 else float("nan")
    mae_d  = float(ed[np.isfinite(ed) & sd].mean()) if (np.isfinite(ed) & sd).sum() > 0 else float("nan")

    valid = np.isfinite(ed) & sd
    if valid.sum() > 5:
        pr, pp = pearsonr(q[valid], ed[valid])
        sr, sp = spearmanr(q[valid], ed[valid])
    else:
        pr = pp = sr = sp = float("nan")

    print(f"\n  {'─'*55}")
    print(f"  P波拾取统计")
    print(f"  {'─'*55}")
    print(f"  总样本数              : {n}")
    print(f"  含噪输入拾取成功率     : {rate_n:.3f}  ({sn.sum()}/{n})")
    print(f"  去噪后拾取成功率       : {rate_d:.3f}  ({sd.sum()}/{n})")
    if not np.isnan(mae_n):
        print(f"  含噪输入平均拾取误差   : {mae_n:.1f} samples ({mae_n/CONFIG['fs']*1000:.0f} ms)")
    if not np.isnan(mae_d):
        print(f"  去噪后平均拾取误差     : {mae_d:.1f} samples ({mae_d/CONFIG['fs']*1000:.0f} ms)")
    print(f"  Quality vs 拾取误差相关性:")
    print(f"    Pearson  r = {pr:.4f}  (p={pp:.3e})")
    print(f"    Spearman ρ = {sr:.4f}  (p={sp:.3e})")
    if not np.isnan(pr):
        if pr < -0.1:
            print("    ✅ 负相关：Quality↑ → 拾取误差↓（符合预期）")
        elif pr > 0.1:
            print("    ⚠️  正相关：Quality↑ → 拾取误差↑（不符合预期）")
        else:
            print("    ➖ 弱相关：Quality 与拾取误差关系不显著")

    return {
        "pick_success_rate_noisy":   round(rate_n, 4),
        "pick_success_rate_denoise": round(rate_d, 4),
        "pick_mae_noisy":            round(mae_n, 2) if not np.isnan(mae_n) else None,
        "pick_mae_denoise":          round(mae_d, 2) if not np.isnan(mae_d) else None,
        "pearson_r":                 round(pr, 4)    if not np.isnan(pr)    else None,
        "spearman_r":                round(sr, 4)    if not np.isnan(sr)    else None,
    }

# ============================================================
# 图表
# ============================================================
def plot_snr_summary(df, output_dir):
    color_map = {
        "<-5": "#d62728", "-5~0": "#ff7f0e", "0~5": "#2ca02c",
        "5~10": "#1f77b4", "10~15": "#9467bd", ">15": "#8c564b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transfer Model — SNR Analysis", fontsize=12, fontweight="bold")

    ax = axes[0]
    for label in SNR_LABELS:
        sub = df[df["snr_group"] == label]
        if sub.empty:
            continue
        ax.scatter(sub["snr_in"], sub["snr_out"],
                   label=label, color=color_map[label], alpha=0.6, s=20)
    lo = df[["snr_in", "snr_out"]].min().min() - 2
    hi = df[["snr_in", "snr_out"]].max().max() + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("Output SNR (dB)")
    ax.set_title("Input vs Output SNR")
    ax.legend(title="Group", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    means  = [df[df["snr_group"] == l]["delta_snr"].mean() for l in SNR_LABELS
              if not df[df["snr_group"] == l].empty]
    stds   = [df[df["snr_group"] == l]["delta_snr"].std()  for l in SNR_LABELS
              if not df[df["snr_group"] == l].empty]
    labels = [l for l in SNR_LABELS if not df[df["snr_group"] == l].empty]
    colors = [color_map[l] for l in labels]
    bars   = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.8, capsize=5, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("SNR Group (Input)")
    ax.set_ylabel("Mean SNR Gain (dB)")
    ax.set_title("Per-Group SNR Gain")
    ax.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                m + (0.1 if m >= 0 else -0.4),
                f"{m:+.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transfer_snr_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [图] transfer_snr_summary.png")

def plot_quality_pick(pick_results, output_dir):
    q   = pick_results["quality"]
    ed  = pick_results["pick_err_denoise"]
    sd  = pick_results["pick_suc_denoise"]
    tol = CONFIG["pick_tol"]

    valid = sd & np.isfinite(ed)
    q_v   = q[valid]
    e_v   = ed[valid]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Transfer Model — Quality vs P-pick Error", fontsize=12, fontweight="bold")

    ax = axes[0]
    sc = ax.scatter(q_v, e_v, c=e_v, cmap="RdYlGn_r",
                    alpha=0.6, s=20, vmin=0, vmax=tol * 2)
    plt.colorbar(sc, ax=ax, label="Pick Error (samples)")
    if len(q_v) > 2:
        pr, pp = pearsonr(q_v, e_v)
        sr, sp = spearmanr(q_v, e_v)
        ax.set_title(f"Quality vs Pick Error\nPearson r={pr:.3f}  Spearman ρ={sr:.3f}", fontsize=9)
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Pick Error (samples)")
    ax.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins    = np.linspace(0, 1, 11)
    labels  = [f"{bins[i]:.1f}~{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    bin_idx = np.clip(np.digitize(q_v, bins) - 1, 0, len(labels) - 1)
    means, stds, counts = [], [], []
    for b in range(len(labels)):
        mask = bin_idx == b
        means.append(e_v[mask].mean() if mask.sum() > 0 else np.nan)
        stds.append(e_v[mask].std()   if mask.sum() > 0 else 0)
        counts.append(mask.sum())
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(labels)))
    bars   = ax.bar(np.arange(len(labels)), means, yerr=stds,
                    color=colors, alpha=0.85, capsize=4, width=0.7)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.set_xlabel("Quality Score Bin")
    ax.set_ylabel("Mean Pick Error (samples)")
    ax.set_title("Quality Bin → Mean Pick Error")
    ax.grid(axis="y", alpha=0.3)
    for bar, cnt, m in zip(bars, counts, means):
        if cnt > 0 and not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width()/2, m + 0.5,
                    f"n={cnt}", ha="center", fontsize=6)

    ax = axes[2]
    thresholds    = np.linspace(0.0, 0.95, 20)
    success_rates = []
    sample_ratios = []
    for thr in thresholds:
        mask = q >= thr
        if mask.sum() == 0:
            success_rates.append(np.nan)
            sample_ratios.append(0)
            continue
        good = sd[mask] & (ed[mask] <= tol)
        success_rates.append(good.sum() / mask.sum())
        sample_ratios.append(mask.sum() / len(q))
    ax2 = ax.twinx()
    ax.plot(thresholds, success_rates, "b-o", ms=4, label=f"Pick Success (err≤{tol})")
    ax2.plot(thresholds, sample_ratios, "r--s", ms=4, label="Sample Ratio", alpha=0.7)
    ax.set_xlabel("Quality Score Threshold")
    ax.set_ylabel("Pick Success Rate", color="blue")
    ax2.set_ylabel("Remaining Sample Ratio", color="red")
    ax.set_title("Quality Threshold → Pick Success Rate")
    ax.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transfer_quality_pick.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [图] transfer_quality_pick.png")

def plot_waveform_samples(model, loader, device, output_dir, n=12):
    model.eval()
    saved = 0
    fig, axes = plt.subplots(n, 3, figsize=(18, n * 2))
    fig.suptitle("Transfer Model — Waveform Samples (Z channel)", fontsize=11, fontweight="bold")

    with torch.no_grad():
        for batch in loader:
            if saved >= n:
                break
            noisy  = batch["noisy"].to(device)
            clean  = batch["clean"].to(device)
            z_cond = batch["z_cond"].to(device)
            try:
                pred, quality, _ = model(noisy, z_cond)
            except Exception:
                continue
            for i in range(noisy.shape[0]):
                if saved >= n:
                    break
                x_z = noisy[i, 2].cpu().numpy()
                y_z = clean[i, 2].cpu().numpy()
                p_z = pred[i,  2].cpu().numpy()
                q   = float(quality[i].item())
                t   = np.arange(len(x_z)) / CONFIG["fs"]

                ax = axes[saved]
                ax[0].plot(t, x_z, lw=0.6, color="#d62728")
                ax[0].set_title(f"Noisy  (sample {saved})", fontsize=8)
                ax[1].plot(t, y_z, lw=0.6, color="#2ca02c")
                ax[1].set_title("Clean", fontsize=8)
                ax[2].plot(t, p_z, lw=0.6, color="#1f77b4")
                ax[2].set_title(f"Denoised  Q={q:.3f}", fontsize=8)
                for a in ax:
                    a.set_xlabel("Time (s)", fontsize=7)
                    a.tick_params(labelsize=7)
                    a.grid(alpha=0.3)
                saved += 1

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transfer_waveform_samples.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  [图] transfer_waveform_samples.png")

# ============================================================
# main
# ============================================================
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("\n[路径检查]")
    for k in ["ckpt_path", "finetune_h5", "finetune_csv", "noise_h5", "noise_csv"]:
        p = CONFIG[k]
        print(f"  [{'✅' if os.path.exists(p) else '❌'}] {k}: {p}")

    print("\n[数据集] 加载...")
    full_ds = EvalDataset(
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
        generator=torch.Generator().manual_seed(CONFIG["seed"])
    )
    loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=CONFIG["num_workers"])
    print(f"  验证集大小: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device    : {device}")

    model = NoiseAwareDenoiserV3(
        z_dim=CONFIG["z_dim"],
        cond_len=CONFIG["cond_len"],
        num_heads=CONFIG["num_heads"],
    ).to(device)
    ckpt = torch.load(CONFIG["ckpt_path"], map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print(f"  Epoch     : {ckpt.get('epoch', 'N/A')}")
        print(f"  Val Loss  : {ckpt.get('val_loss', 'N/A')}")
        print(f"  Val SNR   : {ckpt.get('val_snr_db', 'N/A')}")
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  权重加载  : {CONFIG['ckpt_path']}")

    print("\n[评估中...]")
    records, pick_results = evaluate(model, loader, device)

    df, summary = summarize(records)
    pick_stats  = print_pick_stats(pick_results)

    # 保存 CSV
    df.to_csv(os.path.join(CONFIG["output_dir"], "transfer_results.csv"),
              index=False, float_format="%.4f")
    pd.DataFrame([{**summary, **pick_stats}]).to_csv(
        os.path.join(CONFIG["output_dir"], "transfer_summary.csv"),
        index=False, float_format="%.4f"
    )
    pd.DataFrame({
        "quality":          pick_results["quality"],
        "pick_err_noisy":   pick_results["pick_err_noisy"],
        "pick_err_denoise": pick_results["pick_err_denoise"],
        "pick_suc_noisy":   pick_results["pick_suc_noisy"],
        "pick_suc_denoise": pick_results["pick_suc_denoise"],
    }).to_csv(os.path.join(CONFIG["output_dir"], "transfer_pick_detail.csv"),
              index=False, float_format="%.4f")

    # 图表
    plot_snr_summary(df, CONFIG["output_dir"])
    plot_quality_pick(pick_results, CONFIG["output_dir"])
    plot_waveform_samples(model, loader, device,
                          CONFIG["output_dir"], n=CONFIG["save_fig_num"])

    print(f"\n[✅ 完成] 结果目录: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()