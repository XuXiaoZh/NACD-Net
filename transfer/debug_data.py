# debug_data.py  放在 val_trans_nonnatural.py 同目录
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

EVENT_H5  = r"D:/X/p_wave/data/non_naturaldata.hdf5"
EVENT_CSV = r"D:/X/p_wave/data/non_naturaldata.csv"
OUT_PNG   = r"D:/X/denoise/part1/v3/debug_data.png"

df = pd.read_csv(EVENT_CSV, low_memory=False)
print("CSV 列名:", df.columns.tolist())
print("前5行:\n", df.head())
print("\nP波相关列:")
for col in ["p_arrival_sample", "p_onset", "itp", "Pg"]:
    if col in df.columns:
        print(f"  {col}: min={df[col].min()}, max={df[col].max()}, "
              f"null={df[col].isna().sum()}")

with h5py.File(EVENT_H5, "r") as f:
    keys = list(f["data"].keys())
    print(f"\nHDF5 总条数: {len(keys)}")

    # 取前12条看原始波形
    fig, axes = plt.subplots(12, 1, figsize=(16, 24))
    for i, name in enumerate(keys[:12]):
        raw = f["data"][name][:]
        print(f"  [{i}] {name}: shape={raw.shape}, "
              f"min={raw.min():.4f}, max={raw.max():.4f}")

        # 重采样
        x = raw.T.astype(np.float32)
        x = resample_poly(x, up=2, down=1, axis=1)
        T = x.shape[1]

        # 找 P 波标注
        row = df[df["trace_name"] == name]
        p_sample = None
        if not row.empty:
            for col in ["p_arrival_sample", "p_onset", "itp", "Pg"]:
                if col in row.columns and not row[col].isna().all():
                    p_sample = int(row[col].iloc[0]) * 2  # 50→100Hz
                    break

        # 画 Z 分量（重采样后全段）
        z = x[2]
        axes[i].plot(z, lw=0.5, color="#1f77b4")
        axes[i].set_title(
            f"{name} | T={T} | p_sample(×2)={p_sample}",
            fontsize=7
        )
        # 标注截取窗口
        axes[i].axvspan(4000, min(10000, T), alpha=0.15, color="green",
                        label="截取窗口[4000:10000]")
        if p_sample is not None:
            axes[i].axvline(p_sample, color="red", lw=1.2, label=f"P={p_sample}")
            # 截取后的相对位置
            p_rel = p_sample - 4000
            if 0 <= p_rel < 6000:
                axes[i].axvline(p_sample, color="red", lw=1.2)
                axes[i].set_title(
                    f"{name} | T={T} | P(原)={p_sample} | P(截取后)={p_rel}",
                    fontsize=7
                )
        axes[i].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\n图已保存: {OUT_PNG}")