# -*- coding: utf-8 -*-
"""
实验6：迁移学习 vs 从头训练（相同数据量）
在 non_naturaldata 上分别：
  1. 预训练权重 + 微调
  2. 随机初始化从头训练
对比各数据量下的性能，验证迁移学习的小样本优势
"""

import os, sys, h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "finetune_h5":    r"D:/X/p_wave/data/non_naturaldata.hdf5",
    "finetune_csv":   r"D:/X/p_wave/data/non_naturaldata.csv",
    "noise_h5":       r"D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv":      r"D:/X/p_wave/data/chunk1.csv",
    "output_dir":     r"D:/X/denoise/part1/v3/exp6_transfer_vs_scratch",
    "val_ratio":      0.15,
    "signal_len":     6000,
    "cond_len":       400,
    "z_dim":          128,
    "num_heads":      8,
    "snr_db_range":   (-15.0, 10.0),
    "noise_boost":    1.0,
    "batch_size":     16,
    "num_workers":    0,
    "epochs":         40,
    "lr_finetune":    1e-4,    # 微调用较小学习率
    "lr_scratch":     3e-4,    # 从头训练用正常学习率
    "seed":           42,
    "sta_len":        0.5,
    "lta_len":        4.0,
    "stalta_thr":     2.5,
    "pick_tol":       50,
    "fs":             100,
    # 不同数据量实验点
    "sample_sizes":   [300, 500, 1000, 2000, 5000, 10000],
}

EPS = 1e-10

# ============================================================
# Dataset
# ============================================================
class FinetuneDataset(Dataset):
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
# 指标 & STA/LTA
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

# ============================================================
# 训练 + 验证
# ============================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        noisy  = batch["noisy"].to(device)
        clean  = batch["clean"].to(device)
        z_cond = batch["z_cond"].to(device)
        if not all(torch.isfinite(t).all() for t in [noisy, clean, z_cond]):
            continue
        optimizer.zero_grad()
        try:
            pred, _, _ = model(noisy, z_cond)
        except Exception:
            continue
        loss = nn.functional.l1_loss(pred, clean)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); n += 1
    return total_loss / n if n > 0 else float("nan")

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    delta_snrs, ccs, pick_suc = [], [], []
    kw = dict(fs=CONFIG["fs"], sta_len=CONFIG["sta_len"],
              lta_len=CONFIG["lta_len"], threshold=CONFIG["stalta_thr"])

    for batch in loader:
        noisy  = batch["noisy"].to(device)
        clean  = batch["clean"].to(device)
        z_cond = batch["z_cond"].to(device)
        p_true = batch["p_onset"]
        snr_in = batch["snr_db"]

        try:
            pred, _, _ = model(noisy, z_cond)
        except Exception:
            continue

        for i in range(noisy.shape[0]):
            x_np = noisy[i].cpu().numpy()
            y_np = clean[i].cpu().numpy()
            p_np = pred[i].cpu().numpy()
            pt   = int(p_true[i].item())
            si   = float(snr_in[i].item())

            y_z = y_np[2]; p_z = p_np[2]; x_z = x_np[2]
            delta_snrs.append(snr_db_fn(y_z, p_z) - snr_db_fn(y_z, x_z))
            ccs.append(cc_fn(y_z, p_z))

            pick = stalta_pick(p_np, **kw)
            pick_suc.append(pick >= 0 and abs(pick - pt) <= CONFIG["pick_tol"])

    return {
        "delta_snr":         float(np.mean(delta_snrs)) if delta_snrs else float("nan"),
        "cc":                float(np.mean(ccs))        if ccs        else float("nan"),
        "pick_success_rate": float(np.mean(pick_suc))   if pick_suc   else 0.0,
    }

# ============================================================
# 单次训练流程（微调 or 从头）
# ============================================================
def run_training(train_loader, val_loader, device,
                 mode="finetune", n_samples=1000):
    """
    mode: 'finetune' 或 'scratch'
    """
    model = NoiseAwareDenoiserV3(
        z_dim=CONFIG["z_dim"],
        cond_len=CONFIG["cond_len"],
        num_heads=CONFIG["num_heads"],
    ).to(device)

    if mode == "finetune":
        ckpt  = torch.load(CONFIG["pretrain_ckpt"], map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        lr = CONFIG["lr_finetune"]
        print(f"    [微调] 加载预训练权重，lr={lr}")
    else:
        lr = CONFIG["lr_scratch"]
        print(f"    [从头] 随机初始化，lr={lr}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"]
    )

    best_snr   = -999
    best_metrics = {}
    history    = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "mode":       mode,
            "n_samples":  n_samples,
            "train_loss": train_loss,
            **val_metrics,
        })

        if val_metrics["delta_snr"] > best_snr:
            best_snr     = val_metrics["delta_snr"]
            best_metrics = val_metrics.copy()

        if epoch % 10 == 0:
            print(f"      Epoch {epoch:>3} | loss={train_loss:.4f} | "
                  f"ΔSNR={val_metrics['delta_snr']:+.3f} | "
                  f"Pick={val_metrics['pick_success_rate']:.3f}")

    return best_metrics, history

# ============================================================
# 主函数
# ============================================================
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_records  = []
    all_histories= []

    for n_samples in CONFIG["sample_sizes"]:
        print(f"\n{'='*65}")
        print(f"数据量: {n_samples} 条")

        # 构建数据集
        ds = FinetuneDataset(
            event_h5_path  = CONFIG["finetune_h5"],
            event_csv_path = CONFIG["finetune_csv"],
            noise_h5_path  = CONFIG["noise_h5"],
            noise_csv_path = CONFIG["noise_csv"],
            signal_len     = CONFIG["signal_len"],
            cond_len       = CONFIG["cond_len"],
            snr_db_range   = CONFIG["snr_db_range"],
            noise_boost    = CONFIG["noise_boost"],
            max_samples    = n_samples,
            seed           = CONFIG["seed"],
        )
        n_val   = max(1, int(len(ds) * CONFIG["val_ratio"]))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(
            ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(CONFIG["seed"])
        )
        train_loader = DataLoader(train_ds, batch_size=min(CONFIG["batch_size"], n_train),
                                  shuffle=True,  num_workers=CONFIG["num_workers"])
        val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                                  shuffle=False, num_workers=CONFIG["num_workers"])
        print(f"  训练: {len(train_ds)}  验证: {len(val_ds)}")

        for mode in ["finetune", "scratch"]:
            print(f"\n  ── {mode} ──")
            best, history = run_training(
                train_loader, val_loader, device, mode=mode, n_samples=n_samples
            )
            all_records.append({
                "n_samples": n_samples,
                "mode":      mode,
                **best,
            })
            all_histories.extend(history)

            # 保存每条曲线
            pd.DataFrame(history).to_csv(
                os.path.join(CONFIG["output_dir"],
                             f"history_{mode}_n{n_samples}.csv"),
                index=False, float_format="%.4f"
            )

    # ── 汇总 CSV ────────────────────────────────────────────
    df_result = pd.DataFrame(all_records)
    df_result.to_csv(os.path.join(CONFIG["output_dir"], "exp6_summary.csv"),
                     index=False, float_format="%.4f")

    print(f"\n{'='*75}")
    print("实验6：迁移学习 vs 从头训练 汇总")
    print(f"{'='*75}")
    print(f"  {'N':>6} | {'Mode':<10} | {'ΔSNR':>8} | {'CC':>8} | {'Pick':>8}")
    print(f"  {'-'*55}")
    for _, row in df_result.iterrows():
        print(f"  {int(row['n_samples']):>6} | {row['mode']:<10} | "
              f"{row['delta_snr']:>+8.4f} | "
              f"{row['cc']:>8.4f} | "
              f"{row['pick_success_rate']:>8.4f}")

    # ── 可视化：数据量 vs 指标曲线 ─────────────────────────
    df_ft  = df_result[df_result["mode"] == "finetune"].sort_values("n_samples")
    df_sc  = df_result[df_result["mode"] == "scratch"].sort_values("n_samples")

    metrics_plot = [
        ("delta_snr",         "ΔSNR (dB)",    True),
        ("cc",                "CC",            True),
        ("pick_success_rate", "P波拾取成功率", True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("实验6：迁移学习 vs 从头训练（不同数据量）",
                 fontsize=13, fontweight="bold")

    for ax, (key, label, higher) in zip(axes, metrics_plot):
        ax.plot(df_ft["n_samples"], df_ft[key],
                "b-o", ms=6, lw=2, label="迁移学习（预训练+微调）")
        ax.plot(df_sc["n_samples"], df_sc[key],
                "r--s", ms=6, lw=2, label="从头训练（随机初始化）")
        ax.set_xscale("log")
        ax.set_xlabel("训练样本数", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 标注差值
        for n, v_ft, v_sc in zip(df_ft["n_samples"].tolist(),
                                   df_ft[key].tolist(),
                                   df_sc[key].tolist()):
            diff = v_ft - v_sc
            if abs(diff) > 0.01:
                y_mid = (v_ft + v_sc) / 2
                ax.annotate(f"Δ={diff:+.2f}", xy=(n, y_mid),
                            fontsize=7, color="green",
                            xytext=(5, 0), textcoords="offset points")

        ax.text(0.98, 0.02, "↑ better" if higher else "↓ better",
                transform=ax.transAxes, fontsize=8,
                ha="right", va="bottom", color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "exp6_transfer_vs_scratch.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 训练收敛曲线（以某一数据量为例）──────────────────
    example_n = 1000
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"实验6：训练收敛曲线（N={example_n}）",
                 fontsize=12, fontweight="bold")

    for csv_name, color, label in [
        (f"history_finetune_n{example_n}.csv", "blue",  "迁移学习"),
        (f"history_scratch_n{example_n}.csv",  "red",   "从头训练"),
    ]:
        path = os.path.join(CONFIG["output_dir"], csv_name)
        if not os.path.exists(path):
            continue
        h = pd.read_csv(path)
        axes[0].plot(h["epoch"], h["delta_snr"],
                     color=color, lw=1.5, label=label)
        axes[1].plot(h["epoch"], h["pick_success_rate"],
                     color=color, lw=1.5, label=label)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("ΔSNR (dB)")
    axes[0].set_title("ΔSNR 收敛曲线"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("P波拾取成功率")
    axes[1].set_title("P波拾取收敛曲线"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "exp6_convergence.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[✅] 结果目录: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()