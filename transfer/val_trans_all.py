# -*- coding: utf-8 -*-
# D:\X\denoise\part1\v3\transfer\val_baselines_addnoise.py
"""
deep_denoiser / dprnn / v3 三模型加噪验证脚本
对标 val_wavelet_addnoise.py 的加噪流程（从头取样，固定种子）
"""

import os
import sys
import json
import glob
import importlib
import inspect
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
# 路径
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR   = os.path.abspath(os.path.join(THIS_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
for p in [THIS_DIR, V3_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    "event_h5":  r"D:/X/denoise/part1/v3/transfer/data/event_100hz_6000.hdf5",
    "event_csv": r"D:/X/denoise/part1/v3/transfer/data/event_100hz_6000.csv",
    "noise_h5":  r"D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv": r"D:/X/p_wave/data/chunk1.csv",

    "max_samples":   200,
    "snr_db_range":  (-10.0, 0.0),
    "fixed_snr_db":  None,
    "noise_boost":   1.0,

    "signal_len": 6000,
    "cond_len":   400,
    "seed":       42,
    "fs":         100,

    "batch_size":  8,
    "num_workers": 0,

    "out_root":    r"D:/X/denoise/part1/v3/baeslines/eval_outputs",
    "save_fig_num": 50,
    "image_formats": ["svg", "png"],
    "image_dpi":   220,

    "models": [
        {
            "name": "deep_denoiser",
            "type": "baseline",
            "ckpt_dir": r"D:/X/denoise/part1/v3/baeslines/checkpoints/deep_denoiser",
            "ckpt": None,
            "model_module": "v3.baeslines.deep_denoiser",
            "model_class": "DeepDenoiser",
            "model_kwargs": {},
        },
        {
            "name": "dprnn",
            "type": "baseline",
            "ckpt_dir": r"D:/X/denoise/part1/v3/baeslines/checkpoints/dprnn",
            "ckpt": None,
            "model_module": "v3.baeslines.dprnn",
            "model_class": "DPRNN",
            "model_kwargs": {},
        },
        {
            "name": "v3_best",
            "type": "v3",
            "ckpt": r"D:/X/denoise/part1/v3/checkpoints_v3/best_model_v3.pth",
            "ckpt_dir": None,
            "z_dim": 128,
            "num_heads": 8,
            "cond_len": 400,
        },
    ],
}

# ============================================================
# 数据集
# ============================================================
class AddNoiseDataset(Dataset):
    def __init__(self, event_df, noise_df, event_h5_path, noise_h5_path,
                 signal_len, cond_len, snr_db_range, fixed_snr_db, noise_boost, seed):
        self.event_df = event_df
        self.noise_df = noise_df
        self.event_h5_path = event_h5_path
        self.noise_h5_path = noise_h5_path
        self.signal_len = signal_len
        self.cond_len = cond_len
        self.snr_db_range = snr_db_range
        self.fixed_snr_db = fixed_snr_db
        self.noise_boost = float(noise_boost)
        self.seed = int(seed)
        self._ev_h5 = None
        self._no_h5 = None

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

    def _load(self, h5f, name):
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

        clean = self._norm_peak(self._load(self.ev_h5, trace_name))

        ni = int(rng.integers(0, len(self.noise_df)))
        noise_name = str(self.noise_df.iloc[ni]["trace_name"])
        noise = self._norm_peak(self._load(self.no_h5, noise_name))

        snr_db = float(self.fixed_snr_db) if self.fixed_snr_db is not None \
            else float(rng.uniform(self.snr_db_range[0], self.snr_db_range[1]))

        noisy_base = self._mix_snr(clean, noise, snr_db)
        noisy = clean + self.noise_boost * (noisy_base - clean)

        # z_cond：噪声前段归一化
        z_cond = noise[:, :self.cond_len].copy()
        m = np.abs(z_cond).max()
        if m > 1e-10:
            z_cond = z_cond / m

        p_onset = self._get_p_onset(row)
        valid_mask = np.zeros(self.signal_len, dtype=np.float32)
        valid_mask[p_onset:min(p_onset + 4000, self.signal_len)] = 1.0

        return {
            "clean":      torch.from_numpy(np.clip(clean, -10, 10).astype(np.float32)),
            "noisy":      torch.from_numpy(np.clip(noisy, -10, 10).astype(np.float32)),
            "z_cond":     torch.from_numpy(np.clip(z_cond, -10, 10).astype(np.float32)),
            "valid_mask": torch.from_numpy(valid_mask),
            "trace_name": trace_name,
            "noise_name": noise_name,
            "snr_set_db": torch.tensor(snr_db, dtype=torch.float32),
        }

# ============================================================
# 工具
# ============================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_uname(name, used):
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
    m = valid_mask.unsqueeze(1)
    n = m.sum(dim=[1, 2]) * clean.shape[1] + 1e-10
    sig = ((clean ** 2) * m).sum(dim=[1, 2]) / n
    noi = ((residual ** 2) * m).sum(dim=[1, 2]) / n + 1e-10
    return torch.clamp(10.0 * torch.log10(sig / noi), -50, 50)

def resolve_ckpt(mc):
    ckpt = mc.get("ckpt")
    if ckpt and os.path.exists(ckpt):
        return ckpt
    ckpt_dir = mc.get("ckpt_dir")
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"[{mc['name']}] ckpt_dir 不存在: {ckpt_dir}")
    cands = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not cands:
        raise FileNotFoundError(f"[{mc['name']}] 目录下无 .pth: {ckpt_dir}")
    best = [p for p in cands if "best" in os.path.basename(p).lower()]
    if best:
        return best[0]
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def build_model(mc, device):
    if mc["type"] == "v3":
        try:
            from model_v3 import NoiseAwareDenoiserV3
        except ModuleNotFoundError:
            from v3.model_v3 import NoiseAwareDenoiserV3
        return NoiseAwareDenoiserV3(
            z_dim=mc.get("z_dim", 128),
            cond_len=mc.get("cond_len", 400),
            num_heads=mc.get("num_heads", 8),
        ).to(device)

    mod = importlib.import_module(mc["model_module"])
    cls = getattr(mod, mc["model_class"])
    kwargs = mc.get("model_kwargs", {})
    sig = inspect.signature(cls.__init__)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    try:
        return cls(**filtered).to(device)
    except TypeError:
        return cls().to(device)

def load_weights(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        sd = obj.get("state_dict") or obj.get("model_state_dict") or obj
    else:
        sd = obj
    new_sd = {}
    for k, v in sd.items():
        #兼容 Python 3.8 及以下
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"  ckpt: {os.path.basename(ckpt_path)} | missing={len(missing)} unexpected={len(unexpected)}")
def infer(model, mtype, noisy, z_cond):
    if mtype == "v3":
        den, q, _ = model(noisy, z_cond)
        return den, q.reshape(-1)
    # baseline：先试带 z_cond，再试不带
    try:
        out = model(noisy, z_cond)
    except Exception:
        out = model(noisy)
    den = out[0] if isinstance(out, (tuple, list)) else out
    q = torch.full((noisy.size(0),), float("nan"), device=noisy.device)
    return den, q

# ============================================================
# 绘图
# ============================================================
def save_triplet(clean_3t, noisy_3t, deno_3t, out_base, title="", fs=None,
                 formats=("svg",), dpi=220):
    ensure_dir(os.path.dirname(out_base))
    ch = ["E", "N", "Z"]
    T = clean_3t.shape[-1]
    t = np.arange(T) if fs is None else np.arange(T) / float(fs)
    xlab = "sample" if fs is None else "time (s)"

    fig, axes = plt.subplots(3, 3, figsize=(16, 8), sharex=True)
    for j, ct in enumerate(["Clean", "Noisy", "Denoised"]):
        axes[0, j].set_title(ct, fontsize=11)

    for i in range(3):
        c, n, d = clean_3t[i], noisy_3t[i], deno_3t[i]
        ymax = max(np.max(np.abs(c)), np.max(np.abs(n)), np.max(np.abs(d)), 1e-6)
        for j, sig in enumerate([c, n, d]):
            axes[i, j].plot(t, sig, lw=0.8, color="#1f77b4")
            axes[i, j].set_ylim(-ymax, ymax)
            axes[i, j].grid(alpha=0.25, linestyle="--")
            if j == 0:
                axes[i, j].set_ylabel(ch[i])
    for j in range(3):
        axes[2, j].set_xlabel(xlab)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    for fmt in formats:
        fmt = fmt.lower().strip(".")
        kw = {"dpi": dpi} if fmt in ["png", "jpeg"] else {}
        fig.savefig(f"{out_base}.{fmt}", format=fmt, **kw)
    plt.close(fig)

# ============================================================
# 单模型主循环
# ============================================================
def run_model(mc, loader, device):
    name = mc["name"]
    out_dir = os.path.join(CONFIG["out_root"], name)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "figs"))

    print(f"\n{'='*50}\n[Model] {name}\n{'='*50}")
    ckpt = resolve_ckpt(mc)
    model = build_model(mc, device)
    load_weights(model, ckpt, device)
    model.eval()

    rows = []
    used = set()
    fig_saved = 0

    with h5py.File(os.path.join(out_dir, "clean_noisy_denoised.hdf5"), "w") as hf:
        gc = hf.create_group("clean")
        gn = hf.create_group("noisy")
        gd = hf.create_group("denoised")

        with torch.no_grad():
            for batch in loader:
                clean     = batch["clean"].to(device)
                noisy     = batch["noisy"].to(device)
                z_cond    = batch["z_cond"].to(device)
                vmask     = batch["valid_mask"].to(device)
                tnames    = batch["trace_name"]
                nnames    = batch["noise_name"]
                snr_set   = batch["snr_set_db"].cpu().numpy()

                denoised, quality = infer(model, mc["type"], noisy, z_cond)

                # 长度对齐（baseline 输出长度可能不同）
                if denoised.shape[-1] != clean.shape[-1]:
                    denoised = torch.nn.functional.interpolate(
                        denoised, size=clean.shape[-1], mode="linear", align_corners=False
                    )

                snr_in  = compute_snr_db(clean, noisy - clean, vmask).cpu().numpy()
                snr_out = compute_snr_db(clean, denoised - clean, vmask).cpu().numpy()

                c_np = clean.cpu().numpy()
                n_np = noisy.cpu().numpy()
                d_np = denoised.cpu().numpy()
                q_np = quality.detach().cpu().numpy()

                for i in range(c_np.shape[0]):
                    raw   = str(tnames[i])
                    uname = safe_uname(raw, used)

                    gc.create_dataset(uname, data=c_np[i].T.astype(np.float32), compression="gzip")
                    gn.create_dataset(uname, data=n_np[i].T.astype(np.float32), compression="gzip")
                    gd.create_dataset(uname, data=d_np[i].T.astype(np.float32), compression="gzip")

                    gain = float(snr_out[i] - snr_in[i])
                    rows.append({
                        "trace_name":    raw,
                        "noise_trace":   str(nnames[i]),
                        "snr_set_db":    float(snr_set[i]),
                        "input_snr_db":  float(snr_in[i]),
                        "output_snr_db": float(snr_out[i]),
                        "snr_gain_db":   gain,
                        "quality":       float(q_np[i]) if np.isfinite(q_np[i]) else np.nan,
                    })

                    if fig_saved < CONFIG["save_fig_num"]:
                        out_base = os.path.join(out_dir, "figs", uname)
                        save_triplet(
                            c_np[i], n_np[i], d_np[i], out_base,
                            title=f"{raw} | SNR_set={snr_set[i]:.2f} dB | Gain={gain:.2f} dB",
                            fs=CONFIG["fs"],
                            formats=CONFIG["image_formats"],
                            dpi=CONFIG["image_dpi"],
                        )
                        fig_saved += 1

    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    summary = {
        "model":              name,
        "ckpt":               ckpt,
        "n_samples":          int(len(mdf)),
        "snr_set_db_mean":    float(mdf["snr_set_db"].mean()),
        "input_snr_db_mean":  float(mdf["input_snr_db"].mean()),
        "output_snr_db_mean": float(mdf["output_snr_db"].mean()),
        "snr_gain_db_mean":   float(mdf["snr_gain_db"].mean()),
        "snr_gain_db_median": float(mdf["snr_gain_db"].median()),
        "fig_saved":          fig_saved,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

# ============================================================
# 主函数
# ============================================================
def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    ensure_dir(CONFIG["out_root"])

    for k in ["event_h5", "event_csv", "noise_h5", "noise_csv"]:
        if not os.path.exists(CONFIG[k]):
            raise FileNotFoundError(f"{k}: {CONFIG[k]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    event_df = pd.read_csv(CONFIG["event_csv"], low_memory=False)
    noise_df = pd.read_csv(CONFIG["noise_csv"], low_memory=False)
    if CONFIG["max_samples"] is not None:
        event_df = event_df.iloc[:CONFIG["max_samples"]].reset_index(drop=True)
    print(f"[INFO] events={len(event_df)}, noises={len(noise_df)}")

    ds = AddNoiseDataset(
        event_df=event_df, noise_df=noise_df,
        event_h5_path=CONFIG["event_h5"],
        noise_h5_path=CONFIG["noise_h5"],
        signal_len=CONFIG["signal_len"],
        cond_len=CONFIG["cond_len"],
        snr_db_range=CONFIG["snr_db_range"],
        fixed_snr_db=CONFIG["fixed_snr_db"],
        noise_boost=CONFIG["noise_boost"],
        seed=CONFIG["seed"],
    )
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=CONFIG["num_workers"])

    all_summary = []
    failed = []
    for mc in CONFIG["models"]:
        try:
            s = run_model(mc, loader, device)
            all_summary.append(s)
        except Exception as e:
            print(f"[FAILED] {mc['name']}: {e}")
            failed.append({"model": mc["name"], "error": str(e)})

    pd.DataFrame(all_summary).to_csv(
        os.path.join(CONFIG["out_root"], "all_models_summary.csv"), index=False
    )
    if failed:
        with open(os.path.join(CONFIG["out_root"], "failed.json"), "w") as f:
            json.dump(failed, f, indent=2)

    print(f"\n========== 完成 ==========")
    print(f"成功: {len(all_summary)} | 失败: {len(failed)}")

if __name__ == "__main__":
    main()