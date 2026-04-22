# -*- coding: utf-8 -*-
"""
实验3：源域 vs 目标域分布对比
- 预训练模型直接评估 non_naturaldata（零样本迁移）
- 微调后评估 non_naturaldata
- 量化域偏移改善幅度
"""

import os, sys, h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [THIS_DIR, os.path.abspath(os.path.join(THIS_DIR, ".."))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_v3 import NoiseAwareDenoiserV3

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "pretrain_ckpt":  r"D:/X/denoise/part1/v3/checkpoints_v3/best_model_v3.pth",
    "finetune_ckpt":  r"D:/X/denoise/part1/v3/checkpoints_transfer_15k/best_transfer_15k.pth",
    "finetune_h5":    r"D:/X/p_wave/data/non_naturaldata.hdf5",
    "finetune_csv":   r"D:/X/p_wave/data/non_naturaldata.csv",
    "source_h5":      r"D:/X/p_wave/data/chunk2.hdf5",
    "source_csv":     r"D:/X/p_wave/data/chunk2_val.csv",
    "noise_h5":       r"D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv":      r"D:/X/p_wave/data/chunk1.csv",
    "output_dir":     r"D:/X/denoise/part1/v3/exp3_domain_shift",
    "max_samples":    3000,
    "val_ratio":      0.15,
    "signal_len":     6000,
    "cond_len":       400,
    "z_dim":          128,
    "num_heads":      8,
    "snr_db_range":   (-15.0, 10.0),
    "noise_boost":    1.0,
    "batch_size":     16,
    "num_workers":    0,
    "seed":           42,
    "sta_len":        0.5,
    "lta_len":        4.0,
    "stalta_thr":     2.5,
    "pick_tol":       50,
    "fs":             100,
}

EPS        = 1e-10
SNR_BINS   = [-np.inf, -5, 0, 5, 10, 15, np.inf]
SNR_LABELS = ["<-5", "-5~0", "0~5", "5~10", "10~15", ">15"]

# ============================================================
# Dataset（复用）
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
        for col in ["trace_P_arrival_sample", "p_arrival_sample", "Pg"]:
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
# 指标
# ============================================================
def snr_db_fn(clean, test):
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

def stalta_pick(wave, fs, sta_len, lta_len, threshold):
    x    = wave[2].astype(np.float64) if wave.ndim == 2 else wave.astype(np.float64)
    nsta = int(sta_len * fs)
    nlta = int(lta_len * fs)
    T    = len(x)
    if T < nlta + nsta:
        return -1
    cf    = x ** 2
    cs    = np.cumsum(np.concatenate([[0.0], cf]))
    i0, i1 = nlta, T - nsta
    if i0 >= i1:
        return -1
    idx   = np.arange(i0, i1)
    lta   = (cs[idx] - cs[idx - nlta]) / nlta
    sta   = (cs[idx + nsta] - cs[idx]) / nsta
    valid = lta > EPS
    ratio = np.zeros(len(idx))
    ratio[valid] = sta[valid] / lta[valid]
    trig  = np.where(ratio > threshold)[0]
    return int(trig[0] + i0) if len(trig) > 0 else -1

def assign_snr_group(v):
    for i in range(len(SNR_BINS) - 1):
        if SNR_BINS[i] <= v < SNR_BINS[i + 1]:
            return SNR_LABELS[i]
    return SNR_LABELS[-1]

# ============================================================
# 评估一个模型在一个数据集上
# ============================================================
@torch.no_grad()
def evaluate_model(model, loader, device, tag=""):
    model.eval()
    records  = []
    pick_suc = []
    kw = dict(fs=CONFIG["fs"], sta_len=CONFIG["sta_len"],
              lta_len=CONFIG["lta_len"], threshold=CONFIG["stalta_thr"])

    for batch in loader:
        noisy  = batch["noisy"].to(device)
        clean  = batch["clean"].to(device)
        z_cond = batch["z_cond"].to(device)
        p_true = batch["p_onset"]
        snr_in = batch["snr_db"]

        if not all(torch.isfinite(t).all() for t in [noisy, clean, z_cond]):
            continue
        try:
            pred, quality, _ = model(noisy, z_cond)
        except Exception:
            continue
        if not torch.isfinite(pred).all():
            continue

        for i in range(noisy.shape[0]):
            x_np = noisy[i].cpu().numpy()
            y_np = clean[i].cpu().numpy()
            p_np = pred[i].cpu().numpy()
            pt   = int(p_true[i].item())
            si   = float(snr_in[i].item())

            y_z = y_np[2]; p_z = p_np[2]; x_z = x_np[2]
            snr_i = snr_db_fn(y_z, x_z)
            snr_o = snr_db_fn(y_z, p_z)

            records.append({
                "snr_in":    si,
                "snr_out":   snr_o,
                "delta_snr": snr_o - si,
                "cc":        cc_fn(y_z, p_z),
                "rmse":      rmse_fn(y_z, p_z),
                "quality":   float(quality[i].item()),
                "snr_group": assign_snr_group(si),
            })

            pick = stalta_pick(p_np, **kw)
            pick_suc.append(pick >= 0 and abs(pick - pt) <= CONFIG["pick_tol"])

    df = pd.DataFrame(records)
    result = {
        "tag":              tag,
        "total":            len(df),
        "delta_snr":        float(df["delta_snr"].mean()),
        "cc":               float(df["cc"].mean()),
        "rmse":             float(df["rmse"].mean()),
        "quality":          float(df["quality"].mean()),
        "pick_success_rate":float(np.mean(pick_suc)) if pick_suc else 0.0,
    }
    return df, result

# ============================================================
# 主函数
# ============================================================
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 构建目标域数据集（non_naturaldata）──────────────────
    target_ds = EvalDataset(
        event_h5_path  = CONFIG["finetune_h5"],
        event_csv_path = CONFIG["finetune_csv"],
        noise_h5_path  = CONFIG["noise_h5"],
        noise_csv_path = CONFIG["noise_csv"],
        max_samples    = CONFIG["max_samples"],
        signal_len     = CONFIG["signal_len"],
        cond_len       = CONFIG["cond_len"],
        snr_db_range   = CONFIG["snr_db_range"],
        noise_boost    = CONFIG["noise_boost"],
        seed           = CONFIG["seed"],
    )
    n_val   = max(1, int(len(target_ds) * CONFIG["val_ratio"]))
    n_train = len(target_ds) - n_val
    _, target_val = random_split(
        target_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(CONFIG["seed"])
    )
    target_loader = DataLoader(target_val, batch_size=CONFIG["batch_size"],
                               shuffle=False, num_workers=CONFIG["num_workers"])

    # ── 构建源域数据集（chunk2）────────────────────────────
    source_ds = EvalDataset(
        event_h5_path  = CONFIG["source_h5"],
        event_csv_path = CONFIG["source_csv"],
        noise_h5_path  = CONFIG["noise_h5"],
        noise_csv_path = CONFIG["noise_csv"],
        max_samples    = CONFIG["max_samples"],
        signal_len     = CONFIG["signal_len"],
        cond_len       = CONFIG["cond_len"],
        snr_db_range   = CONFIG["snr_db_range"],
        noise_boost    = CONFIG["noise_boost"],
        seed           = CONFIG["seed"],
    )
    n_val_s   = max(1, int(len(source_ds) * CONFIG["val_ratio"]))
    n_train_s = len(source_ds) - n_val_s
    _, source_val = random_split(
        source_ds, [n_train_s, n_val_s],
        generator=torch.Generator().manual_seed(CONFIG["seed"])
    )
    source_loader = DataLoader(source_val, batch_size=CONFIG["batch_size"],
                               shuffle=False, num_workers=CONFIG["num_workers"])

    def load_model(ckpt_path):
        m = NoiseAwareDenoiserV3(
            z_dim=CONFIG["z_dim"],
            cond_len=CONFIG["cond_len"],
            num_heads=CONFIG["num_heads"],
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        m.load_state_dict(state)
        m.eval()
        return m

    # ── 4组评估 ─────────────────────────────────────────────
    EVALS = [
        {
            "ckpt":    CONFIG["pretrain_ckpt"],
            "dataset": "source",
            "loader":  source_loader,
            "tag":     "预训练模型 → 源域(STEAD)",
        },
        {
            "ckpt":    CONFIG["pretrain_ckpt"],
            "dataset": "target",
            "loader":  target_loader,
            "tag":     "预训练模型 → 目标域(non_natural)",
        },
        {
            "ckpt":    CONFIG["finetune_ckpt"],
            "dataset": "source",
            "loader":  source_loader,
            "tag":     "微调模型 → 源域(STEAD)",
        },
        {
            "ckpt":    CONFIG["finetune_ckpt"],
            "dataset": "target",
            "loader":  target_loader,
            "tag":     "微调模型 → 目标域(non_natural)",
        },
    ]

    all_results = []
    all_dfs     = {}

    for ev in EVALS:
        print(f"\n[评估] {ev['tag']}")
        model = load_model(ev["ckpt"])
        df, result = evaluate_model(model, ev["loader"], device, tag=ev["tag"])
        all_results.append(result)
        all_dfs[ev["tag"]] = df

        df.to_csv(
            os.path.join(CONFIG["output_dir"],
                         f"detail_{ev['tag'].replace(' ', '_').replace('/', '')}.csv"),
            index=False, float_format="%.4f"
        )
        del model

    # ── 打印汇总 ────────────────────────────────────────────
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(os.path.join(CONFIG["output_dir"], "exp3_summary.csv"),
                      index=False, float_format="%.4f")

    print(f"\n{'='*75}")
    print("实验3：域偏移分析汇总")
    print(f"{'='*75}")
    print(f"  {'场景':<35} {'ΔSNR':>8} {'CC':>8} {'RMSE':>8} {'Pick':>8}")
    print(f"  {'-'*70}")
    for r in all_results:
        print(f"  {r['tag']:<35} "
              f"{r['delta_snr']:>+8.4f} "
              f"{r['cc']:>8.4f} "
              f"{r['rmse']:>8.4f} "
              f"{r['pick_success_rate']:>8.4f}")

    # ── 域偏移改善量 ────────────────────────────────────────
    r = {x["tag"]: x for x in all_results}
    pre_target  = r.get("预训练模型 → 目标域(non_natural)", {})
    fine_target = r.get("微调模型 → 目标域(non_natural)", {})
    if pre_target and fine_target:
        improve_snr  = fine_target["delta_snr"]  - pre_target["delta_snr"]
        improve_pick = fine_target["pick_success_rate"] - pre_target["pick_success_rate"]
        print(f"\n  域偏移改善:")
        print(f"    ΔSNR 提升  : {improve_snr:+.4f} dB")
        print(f"    Pick 提升  : {improve_pick:+.4f}")

    # ── 可视化 ──────────────────────────────────────────────
    metrics_plot = [
        ("delta_snr",        "ΔSNR (dB)",    True),
        ("cc",               "CC",            True),
        ("rmse",             "RMSE",          False),
        ("pick_success_rate","P波拾取成功率", True),
    ]
    tags_short = [
        "预训练\n→源域", "预训练\n→目标域",
        "微调\n→源域",   "微调\n→目标域",
    ]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("实验3：预训练 vs 微调 × 源域 vs 目标域", fontsize=13, fontweight="bold")

    for ax, (key, label, higher) in zip(axes, metrics_plot):
        vals = [r.get(key, float("nan")) for r in all_results]
        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85, width=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(tags_short, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        # 标注预训练→目标域 和 微调→目标域 用红框区分
        for j in [1, 3]:
            bars[j].set_edgecolor("black")
            bars[j].set_linewidth(2)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        v + abs(v)*0.01,
                        f"{v:.3f}", ha="center", fontsize=8)
        ax.text(0.98, 0.98, "↑ better" if higher else "↓ better",
                transform=ax.transAxes, fontsize=8, ha="right", va="top", color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "exp3_domain_compare.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── SNR 分组细粒度对比（预训练 vs 微调，目标域）────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("实验3：目标域 SNR 分组对比（预训练 vs 微调）", fontsize=12, fontweight="bold")
    for ax, tag, title in zip(axes,
                               ["预训练模型 → 目标域(non_natural)",
                                "微调模型 → 目标域(non_natural)"],
                               ["预训练模型（域偏移）", "微调模型（迁移后）"]):
        df_sub = all_dfs.get(tag, pd.DataFrame())
        if df_sub.empty:
            continue
        means = [df_sub[df_sub["snr_group"] == l]["delta_snr"].mean()
                 for l in SNR_LABELS if not df_sub[df_sub["snr_group"] == l].empty]
        labs  = [l for l in SNR_LABELS
                 if not df_sub[df_sub["snr_group"] == l].empty]
        ax.bar(labs, means, color="#1f77b4" if "微调" in title else "#d62728",
               alpha=0.8, width=0.6)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("SNR Group (Input)")
        ax.set_ylabel("Mean ΔSNR (dB)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for i, (l, m) in enumerate(zip(labs, means)):
            if not np.isnan(m):
                ax.text(i, m + 0.1, f"{m:+.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "exp3_snr_group_compare.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[✅] 结果目录: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()