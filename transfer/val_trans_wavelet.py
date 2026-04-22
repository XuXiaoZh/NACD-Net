# -*- coding: utf-8 -*-
# D:\X\denoise\part1\v3\transfer\val_wavelet_addnoise.py
"""
传统小波去噪 - 加噪版（从头取样，不用尾部）
对标 val_addnoise.py 流程
"""

import os
import json
import h5py
import numpy as np
import pandas as pd
import pywt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
# 配置
# ============================================================
CONFIG = {
    "event_h5":  r"D:/X/denoise/part1/v3/transfer/data/event_100hz_6000.hdf5",
    "event_csv": r"D:/X/denoise/part1/v3/transfer/data/event_100hz_6000.csv",
    "noise_h5":  r"D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv": r"D:/X/p_wave/data/chunk1.csv",

    "max_samples": 200,

    "snr_db_range": (-10.0, 0.0),
    "fixed_snr_db": None,
    "noise_boost":  1.0,

    "signal_len": 6000,
    "cond_len":   400,
    "seed":       42,
    "fs":         100,

    # 小波参数
    "wavelet":        "db4",
    "level":          5,
    "threshold_mode": "soft",

    # 输出
    "out_dir":      r"v3/val_wavelet_addnoise_outputs",
    "out_h5":       r"v3/val_wavelet_addnoise_outputs/clean_noisy_denoised.hdf5",
    "metrics_csv":  r"v3/val_wavelet_addnoise_outputs/metrics.csv",
    "summary_json": r"v3/val_wavelet_addnoise_outputs/summary.json",
    "save_svg_num": 50,
    "svg_dir":      r"v3/val_wavelet_addnoise_outputs/svg_triplets",
}

# ============================================================
# 小波去噪
# ============================================================
def wavelet_denoise_1d(signal: np.ndarray) -> np.ndarray:
    coeffs = pywt.wavedec(signal, CONFIG["wavelet"], level=CONFIG["level"])
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(max(len(signal), 2)))
    coeffs_thresh = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=CONFIG["threshold_mode"])
        for c in coeffs[1:]
    ]
    return pywt.waverec(coeffs_thresh, CONFIG["wavelet"])[:len(signal)].astype(np.float32)

def wavelet_denoise_3ch(wave: np.ndarray) -> np.ndarray:
    return np.stack([wavelet_denoise_1d(wave[c]) for c in range(3)])

# ============================================================
# 数据工具
# ============================================================
def load_wave(h5f, trace_name, signal_len):
    x = h5f["data"][trace_name][:]
    x = x.T.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    T = x.shape[1]
    if T >= signal_len:
        return x[:, :signal_len]
    out = np.zeros((3, signal_len), dtype=np.float32)
    out[:, :T] = x
    return out

def normalize_peak(x):
    m = np.abs(x).max()
    return (x / m, m) if m > 1e-10 else (x, 1.0)

def mix_snr_db(clean_n, noise_n, snr_db):
    snr_lin = 10.0 ** (snr_db / 10.0)
    ps = np.mean(clean_n ** 2)
    pn = np.mean(noise_n ** 2)
    if ps < 1e-12 or pn < 1e-12:
        return clean_n.copy(), 0.0
    scale = float(np.clip(np.sqrt(ps / (snr_lin * pn)), 0.0, 10.0))
    return clean_n + scale * noise_n, scale

def get_p_onset(row, signal_len):
    for col in ["p_arrival_sample", "p_onset", "itp"]:
        if col in row.index and not pd.isna(row[col]):
            try:
                v = int(row[col])
                if 0 <= v < signal_len:
                    return v
            except Exception:
                pass
    return signal_len // 10

def compute_snr_db(clean, residual, p_onset, signal_len):
    ev_l = p_onset
    ev_r = min(signal_len, p_onset + 4000)
    if ev_r - ev_l < 10:
        ev_r = min(signal_len, ev_l + 1000)
    sig = np.mean(clean[:, ev_l:ev_r] ** 2) + 1e-12
    noi = np.mean(residual[:, ev_l:ev_r] ** 2) + 1e-12
    return float(np.clip(10.0 * np.log10(sig / noi), -50, 50))

# ============================================================
# SVG 三列图
# ============================================================
def save_triplet_svg(clean_3t, noisy_3t, deno_3t, out_svg, title="", fs=None):
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    ch_names = ["E", "N", "Z"]
    T = clean_3t.shape[-1]
    t = np.arange(T) if fs is None else np.arange(T) / float(fs)
    xlab = "sample" if fs is None else "time (s)"

    fig, axes = plt.subplots(3, 3, figsize=(16, 8), sharex=True)
    for j, col_title in enumerate(["Clean", "Noisy", "Denoised (Wavelet)"]):
        axes[0, j].set_title(col_title, fontsize=11)

    for i in range(3):
        c, n, d = clean_3t[i], noisy_3t[i], deno_3t[i]
        ymax = max(np.max(np.abs(c)), np.max(np.abs(n)), np.max(np.abs(d)), 1e-6)
        for j, sig in enumerate([c, n, d]):
            axes[i, j].plot(t, sig, lw=0.8, color="#1f77b4")
            axes[i, j].set_ylim(-ymax, ymax)
            axes[i, j].grid(alpha=0.25, linestyle="--")
            if j == 0:
                axes[i, j].set_ylabel(ch_names[i])

    for j in range(3):
        axes[2, j].set_xlabel(xlab)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)

# ============================================================
# 主流程
# ============================================================
def main():
    np.random.seed(CONFIG["seed"])
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    os.makedirs(CONFIG["svg_dir"], exist_ok=True)

    for k in ["event_h5", "event_csv", "noise_h5", "noise_csv"]:
        if not os.path.exists(CONFIG[k]):
            raise FileNotFoundError(f"{k} not found: {CONFIG[k]}")

    event_df = pd.read_csv(CONFIG["event_csv"], low_memory=False)
    noise_df = pd.read_csv(CONFIG["noise_csv"], low_memory=False)

    if CONFIG["max_samples"] is not None:
        event_df = event_df.iloc[:int(CONFIG["max_samples"])].reset_index(drop=True)

    print(f"[INFO] events={len(event_df)}, noises={len(noise_df)}")
    print(f"[INFO] wavelet={CONFIG['wavelet']}, level={CONFIG['level']}, mode={CONFIG['threshold_mode']}")

    metrics = []
    svg_saved = 0
    used_names = set()

    with h5py.File(CONFIG["event_h5"], "r") as ev_h5, \
         h5py.File(CONFIG["noise_h5"], "r") as no_h5, \
         h5py.File(CONFIG["out_h5"], "w") as out_h5:

        g_clean = out_h5.create_group("clean")
        g_noisy = out_h5.create_group("noisy")
        g_deno  = out_h5.create_group("denoised")

        for idx, row in event_df.iterrows():
            trace_name = str(row["trace_name"])
            rng = np.random.default_rng(CONFIG["seed"] + idx)

            # clean
            clean = load_wave(ev_h5, trace_name, CONFIG["signal_len"])
            clean_n, _ = normalize_peak(clean)

            # noise 随机采样
            ni = int(rng.integers(0, len(noise_df)))
            noise_name = str(noise_df.iloc[ni]["trace_name"])
            noise = load_wave(no_h5, noise_name, CONFIG["signal_len"])
            noise_n, _ = normalize_peak(noise)

            # SNR 混合
            if CONFIG["fixed_snr_db"] is None:
                snr_db = float(rng.uniform(CONFIG["snr_db_range"][0], CONFIG["snr_db_range"][1]))
            else:
                snr_db = float(CONFIG["fixed_snr_db"])

            noisy_base, _ = mix_snr_db(clean_n, noise_n, snr_db)
            noisy_n = clean_n + CONFIG["noise_boost"] * (noisy_base - clean_n)
            noisy_n = np.clip(noisy_n, -10, 10).astype(np.float32)

            # 小波去噪
            deno = wavelet_denoise_3ch(noisy_n)

            # 指标
            p_onset = get_p_onset(row, CONFIG["signal_len"])
            snr_in  = compute_snr_db(clean_n, noisy_n - clean_n, p_onset, CONFIG["signal_len"])
            snr_out = compute_snr_db(clean_n, deno - clean_n,    p_onset, CONFIG["signal_len"])

            # 唯一名
            base = trace_name.replace("/", "_").replace("\\", "_")
            uname = base
            if uname in used_names:
                i = 1
                while f"{base}_{i}" in used_names:
                    i += 1
                uname = f"{base}_{i}"
            used_names.add(uname)

            # H5
            g_clean.create_dataset(uname, data=clean_n.T.astype(np.float32), compression="gzip")
            g_noisy.create_dataset(uname, data=noisy_n.T.astype(np.float32), compression="gzip")
            g_deno.create_dataset(uname,  data=deno.T.astype(np.float32),    compression="gzip")

            metrics.append({
                "trace_name":    trace_name,
                "noise_trace":   noise_name,
                "snr_set_db":    snr_db,
                "input_snr_db":  snr_in,
                "output_snr_db": snr_out,
                "snr_gain_db":   snr_out - snr_in,
            })

            # SVG
            if svg_saved < CONFIG["save_svg_num"]:
                out_svg = os.path.join(CONFIG["svg_dir"], f"{uname}_triplet.svg")
                save_triplet_svg(
                    clean_n, noisy_n, deno,
                    out_svg=out_svg,
                    title=f"{trace_name} | SNR_set={snr_db:.2f} dB | SNR_gain={snr_out - snr_in:.2f} dB",
                    fs=CONFIG["fs"],
                )
                svg_saved += 1

            if (idx + 1) % 50 == 0:
                print(f"  processed {idx + 1}/{len(event_df)}")

    mdf = pd.DataFrame(metrics)
    mdf.to_csv(CONFIG["metrics_csv"], index=False)

    summary = {
        "method":               f"wavelet ({CONFIG['wavelet']}, level={CONFIG['level']}, {CONFIG['threshold_mode']})",
        "n_samples":            int(len(mdf)),
        "snr_set_db_mean":      float(mdf["snr_set_db"].mean()),
        "input_snr_db_mean":    float(mdf["input_snr_db"].mean()),
        "output_snr_db_mean":   float(mdf["output_snr_db"].mean()),
        "snr_gain_db_mean":     float(mdf["snr_gain_db"].mean()),
        "snr_gain_db_median":   float(mdf["snr_gain_db"].median()),
        "svg_saved":            int(svg_saved),
        "out_h5":               CONFIG["out_h5"],
        "metrics_csv":          CONFIG["metrics_csv"],
    }

    with open(CONFIG["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== 小波去噪（加噪版）完成 ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()