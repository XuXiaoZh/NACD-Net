
# -*- coding: utf-8 -*-
# D:\X\denoise\part1\v3\transfer\val_addnoise.py
"""
先加噪再降噪的验证脚本（chunk2 + chunk1）

功能：
1) 从 chunk2 读取原始事件波形 clean
2) 从 chunk1 随机采样噪声 noise，并按随机 SNR(dB) 混合得到 noisy
3) 用 V3 模型对 noisy 去噪，得到 denoised
4) 保存 HDF5（三组数据）：
   - /clean/<trace>
   - /noisy/<trace>
   - /denoised/<trace>
5) 每个样本保存 3 张图（不叠加）：
   - clean.png
   - noisy.png
   - denoised.png
6) 保存 metrics.csv（input_snr / output_snr / snr_gain）

注意：
- 模型输入使用归一化域（与训练一致）
- HDF5里保存的是“归一化域波形 [T,3]”
"""
# -*- coding: utf-8 -*-
# D:\X\denoise\part1\v3\transfer\val_addnoise.py
# -*- coding: utf-8 -*-
# D:\X\denoise\part1\v3\transfer\val_addnoise.py
# -*- coding: utf-8 -*-
# D:\X\denoise\part1\v3\transfer\val.py
# -*- coding: utf-8 -*-
# D:\X\denoise\part1\v3\transfer\val.py
# -*- coding: utf-8 -*-
# D:\X\denoise\part1\v3\transfer\val.py

import os
import sys
import json
import glob
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 全局字体：Times New Roman（新罗马）
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.serif"] = ["Times New Roman"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["svg.fonttype"] = "none"   # SVG中保留文字，不转路径
# ---------------- 路径 ----------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))                 # ...\v3\transfer
V3_DIR   = os.path.abspath(os.path.join(THIS_DIR, ".."))              # ...\v3
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))        # ...\part1

for p in [V3_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from model_v3 import NoiseAwareDenoiserV3
except ModuleNotFoundError:
    from v3.model_v3 import NoiseAwareDenoiserV3

# ============================================================
# 配置（强噪 + SVG）
# ============================================================
CONFIG = {
    # 数据路径
    "event_h5":  r"D:/X/p_wave/data/chunk2.hdf5",   # clean事件
    "event_csv": r"D:/X/p_wave/data/chunk2.csv",
    "noise_h5":  r"D:/X/p_wave/data/chunk1.hdf5",   # 噪声库
    "noise_csv": r"D:/X/p_wave/data/chunk1.csv",

    # 测试样本范围
    "use_tail_n": 1000,      # 从chunk2后N条取样；None=全量
    "max_samples": 200,      # 最多跑多少条；None=不限制

    # 强噪参数
    "snr_db_range": (-10.0, 0.0),  # 随机强噪区间（dB）
    "fixed_snr_db": None,          # 例如 -5.0 固定强噪；None=随机
    "noise_boost": 1.0,            # >1 额外放大噪声分量

    # 模型
    "ckpt_path": r"D:/X/denoise/part1/v3/checkpoints_v3/best_model_v3.pth",
    "z_dim": 128,
    "num_heads": 8,
    "cond_len": 400,
    "signal_len": 6000,

    # 推理
    "batch_size": 8,
    "num_workers": 0,
    "seed": 42,

    # 输出
    "out_dir": r"v3/val_addnoise_outputs_strong",
    "out_h5": r"v3/val_addnoise_outputs_strong/clean_noisy_denoised.hdf5",
    "metrics_csv": r"v3/val_addnoise_outputs_strong/metrics.csv",
    "summary_json": r"v3/val_addnoise_outputs_strong/summary.json",

    # SVG 对比图
    "save_svg_num": 50,  # 前N条保存SVG
    "svg_dir": r"v3/val_addnoise_outputs_strong/svg_triplets",
    "fs": None,          # 如100Hz可填100
}

# ============================================================
# 数据集：clean + sampled noise -> strong noisy
# ============================================================
class AddNoiseEvalDataset(Dataset):
    def __init__(
        self,
        event_h5_path,
        event_csv_path,
        noise_h5_path,
        noise_csv_path,
        signal_len=6000,
        cond_len=400,
        snr_db_range=(-10.0, 0.0),
        fixed_snr_db=None,
        noise_boost=2.0,
        use_tail_n=1000,
        max_samples=None,
        seed=42,
    ):
        super().__init__()
        self.event_h5_path = event_h5_path
        self.noise_h5_path = noise_h5_path
        self.signal_len = signal_len
        self.cond_len = cond_len
        self.snr_db_range = snr_db_range
        self.fixed_snr_db = fixed_snr_db
        self.noise_boost = float(noise_boost)
        self.seed = int(seed)

        self.event_df = pd.read_csv(event_csv_path, low_memory=False)
        self.noise_df = pd.read_csv(noise_csv_path, low_memory=False)

        if use_tail_n is not None:
            self.event_df = self.event_df.tail(int(use_tail_n)).reset_index(drop=True)
        if max_samples is not None:
            self.event_df = self.event_df.iloc[:int(max_samples)].reset_index(drop=True)

        self._event_h5 = None
        self._noise_h5 = None

        print(f"[Dataset] events={len(self.event_df)}, noises={len(self.noise_df)}")
        print(f"[Dataset] snr_db_range={self.snr_db_range}, fixed_snr_db={self.fixed_snr_db}, noise_boost={self.noise_boost}")

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
        x = x.T.astype(np.float32)  # [3, T]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

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
        if m > 1e-10:
            return x / m, m
        return x, 1.0

    @staticmethod
    def _mix_snr_db(clean_n, noise_n, snr_db):
        """
        snr_db = 10log10(Ps/Pn)
        """
        snr_lin = 10.0 ** (snr_db / 10.0)
        ps = np.mean(clean_n ** 2)
        pn = np.mean(noise_n ** 2)
        if ps < 1e-12 or pn < 1e-12:
            return clean_n.copy(), 0.0
        scale = float(np.clip(np.sqrt(ps / (snr_lin * pn)), 0.0, 10.0))
        noisy = clean_n + scale * noise_n
        return noisy, scale

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

        # clean
        clean = self._pad_or_crop(self._load(self.event_h5, trace_name))
        clean_n, _ = self._normalize_peak(clean)

        # noise 随机采样
        ni = int(rng.integers(0, len(self.noise_df)))
        noise_name = str(self.noise_df.iloc[ni]["trace_name"])
        noise = self._pad_or_crop(self._load(self.noise_h5, noise_name))
        noise_n, _ = self._normalize_peak(noise)

        # snr
        if self.fixed_snr_db is None:
            snr_db = float(rng.uniform(self.snr_db_range[0], self.snr_db_range[1]))
        else:
            snr_db = float(self.fixed_snr_db)

        # 按SNR混合 + 强噪增强
        noisy_base, noise_scale = self._mix_snr_db(clean_n, noise_n, snr_db)
        noisy_n = clean_n + self.noise_boost * (noisy_base - clean_n)

        # z_cond（噪声前段）
        z_cond = noise_n[:, :self.cond_len].copy()
        zc_m = np.abs(z_cond).max()
        if zc_m > 1e-10:
            z_cond = z_cond / zc_m

        p_onset = self._get_p_onset(row)
        valid_mask = np.zeros(self.signal_len, dtype=np.float32)
        valid_mask[p_onset:min(p_onset + 4000, self.signal_len)] = 1.0

        clean_n = np.clip(clean_n, -10, 10).astype(np.float32)
        noisy_n = np.clip(noisy_n, -10, 10).astype(np.float32)
        z_cond  = np.clip(z_cond,  -10, 10).astype(np.float32)

        return {
            "clean": torch.from_numpy(clean_n),      # [3,T]
            "noisy": torch.from_numpy(noisy_n),      # [3,T]
            "z_cond": torch.from_numpy(z_cond),      # [3,C]
            "valid_mask": torch.from_numpy(valid_mask),
            "trace_name": trace_name,
            "noise_name": noise_name,
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            "noise_scale": torch.tensor(noise_scale, dtype=torch.float32),
        }

# ============================================================
# 工具函数
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def find_checkpoint(prefer_ckpt_path: str) -> str:
    if os.path.exists(prefer_ckpt_path):
        return prefer_ckpt_path

    ckpt_dir = os.path.dirname(prefer_ckpt_path)
    cands = glob.glob(os.path.join(ckpt_dir, "ckpt_epoch*.pth"))
    if not cands:
        raise FileNotFoundError(f"找不到 checkpoint: {prefer_ckpt_path}")

    def _epoch_num(p):
        name = os.path.basename(p)  # e.g. ckpt_epoch10.pth
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else -1

    cands.sort(key=_epoch_num)
    return cands[-1]

def safe_unique_name(name: str, used: set):
    base = str(name).replace("/", "_").replace("\\", "_")
    if base not in used:
        used.add(base)
        return base
    i = 1
    while f"{base}_{i}" in used:
        i += 1
    new_name = f"{base}_{i}"
    used.add(new_name)
    return new_name

def compute_snr_db(clean, residual, valid_mask):
    # clean/residual: [B,3,T], valid_mask:[B,T]
    mask = valid_mask.unsqueeze(1)  # [B,1,T]
    n = mask.sum(dim=[1, 2]) * clean.shape[1] + 1e-10
    sig = ((clean ** 2) * mask).sum(dim=[1, 2]) / n
    noi = ((residual ** 2) * mask).sum(dim=[1, 2]) / n + 1e-10
    snr = 10.0 * torch.log10(sig / noi)
    return torch.clamp(snr, -50, 50)

def save_triplet_svg(clean_3t, noisy_3t, deno_3t, out_svg, title="", fs=None):
    """
    clean_3t/noisy_3t/deno_3t: np.ndarray [3,T]
    输出：3行x3列 SVG
      col1=正常 col2=加噪 col3=降噪
      row = E/N/Z
    """
    ensure_dir(os.path.dirname(out_svg))

    ch_names = ["E", "N", "Z"]
    T = clean_3t.shape[-1]
    t = np.arange(T) if fs is None else np.arange(T) / float(fs)
    xlab = "sample" if fs is None else "time (s)"

    fig, axes = plt.subplots(3, 3, figsize=(16, 8), sharex=True)
    col_titles = ["Clean", "Noisy", "Denoised"]
    for j in range(3):
        axes[0, j].set_title(col_titles[j], fontsize=11)

    for i in range(3):
        c = clean_3t[i]
        n = noisy_3t[i]
        d = deno_3t[i]

        # 同一行统一Y范围，便于直观比较
        ymax = max(np.max(np.abs(c)), np.max(np.abs(n)), np.max(np.abs(d)), 1e-6)
        ymin = -ymax

        axes[i, 0].plot(t, c, lw=0.8, color="#1f77b4")
        axes[i, 1].plot(t, n, lw=0.8, color="#1f77b4")
        axes[i, 2].plot(t, d, lw=0.8, color="#1f77b4")

        for j in range(3):
            axes[i, j].set_ylim(ymin, ymax)
            axes[i, j].grid(alpha=0.25, linestyle="--")
            if j == 0:
                axes[i, j].set_ylabel(ch_names[i])

    for j in range(3):
        axes[2, j].set_xlabel(xlab)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)

# ============================================================
# 主流程
# ============================================================
def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    ensure_dir(CONFIG["out_dir"])
    ensure_dir(CONFIG["svg_dir"])

    for k in ["event_h5", "event_csv", "noise_h5", "noise_csv"]:
        if not os.path.exists(CONFIG[k]):
            raise FileNotFoundError(f"{k} not found: {CONFIG[k]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 模型
    model = NoiseAwareDenoiserV3(
        z_dim=CONFIG["z_dim"],
        cond_len=CONFIG["cond_len"],
        num_heads=CONFIG["num_heads"],
    ).to(device)

    ckpt_path = find_checkpoint(CONFIG["ckpt_path"])
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
    loader = DataLoader(
        ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=False,
    )

    used_names = set()
    metrics = []
    svg_saved = 0

    with h5py.File(CONFIG["out_h5"], "w") as f:
        g_clean = f.create_group("clean")
        g_noisy = f.create_group("noisy")
        g_deno  = f.create_group("denoised")

        with torch.no_grad():
            for batch in loader:
                clean = batch["clean"].to(device)           # [B,3,T]
                noisy = batch["noisy"].to(device)           # [B,3,T]
                z_cond = batch["z_cond"].to(device)         # [B,3,C]
                valid_mask = batch["valid_mask"].to(device)
                names = batch["trace_name"]
                noise_names = batch["noise_name"]
                snr_db = batch["snr_db"].cpu().numpy()

                denoised, quality, _ = model(noisy, z_cond)

                # SNR评估
                snr_in = compute_snr_db(clean, noisy - clean, valid_mask).cpu().numpy()
                snr_out = compute_snr_db(clean, denoised - clean, valid_mask).cpu().numpy()
                snr_gain = snr_out - snr_in

                clean_np = clean.cpu().numpy()
                noisy_np = noisy.cpu().numpy()
                deno_np = denoised.cpu().numpy()
                q_np = quality.cpu().numpy().reshape(-1)

                B = clean_np.shape[0]
                for i in range(B):
                    raw_name = str(names[i])
                    uname = safe_unique_name(raw_name, used_names)

                    c = clean_np[i]  # [3,T]
                    n = noisy_np[i]
                    d = deno_np[i]

                    # H5保存 [T,3]
                    g_clean.create_dataset(uname, data=c.T.astype(np.float32), compression="gzip")
                    g_noisy.create_dataset(uname, data=n.T.astype(np.float32), compression="gzip")
                    g_deno.create_dataset(uname, data=d.T.astype(np.float32), compression="gzip")

                    metrics.append({
                        "trace_name": raw_name,
                        "noise_trace_name": str(noise_names[i]),
                        "snr_set_db": float(snr_db[i]),
                        "input_snr_db": float(snr_in[i]),
                        "output_snr_db": float(snr_out[i]),
                        "snr_gain_db": float(snr_gain[i]),
                        "quality": float(q_np[i]),
                    })

                    # 保存SVG三列图
                    if svg_saved < CONFIG["save_svg_num"]:
                        out_svg = os.path.join(CONFIG["svg_dir"], f"{uname}_triplet.svg")
                        save_triplet_svg(
                            c, n, d,
                            out_svg=out_svg,
                            title=f"{raw_name} | SNR_set={snr_db[i]:.2f} dB | SNR_gain={snr_gain[i]:.2f} dB",
                            fs=CONFIG["fs"],
                        )
                        svg_saved += 1

    # 输出统计
    mdf = pd.DataFrame(metrics)
    mdf.to_csv(CONFIG["metrics_csv"], index=False)

    summary = {
        "n_samples": int(len(mdf)),
        "snr_set_db_mean": float(mdf["snr_set_db"].mean()) if len(mdf) else float("nan"),
        "input_snr_db_mean": float(mdf["input_snr_db"].mean()) if len(mdf) else float("nan"),
        "output_snr_db_mean": float(mdf["output_snr_db"].mean()) if len(mdf) else float("nan"),
        "snr_gain_db_mean": float(mdf["snr_gain_db"].mean()) if len(mdf) else float("nan"),
        "snr_gain_db_median": float(mdf["snr_gain_db"].median()) if len(mdf) else float("nan"),
        "quality_mean": float(mdf["quality"].mean()) if len(mdf) else float("nan"),
        "fixed_snr_db": CONFIG["fixed_snr_db"],
        "snr_db_range": list(CONFIG["snr_db_range"]),
        "noise_boost": CONFIG["noise_boost"],
        "svg_saved": int(svg_saved),
        "ckpt_used": ckpt_path,
        "out_h5": CONFIG["out_h5"],
        "metrics_csv": CONFIG["metrics_csv"],
        "svg_dir": CONFIG["svg_dir"],
    }

    with open(CONFIG["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== 强噪验证完成 ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()