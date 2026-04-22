# -*- coding: utf-8 -*-
# D:\X\denoise\part1\v3\transfer\val_trans.py
"""
三阶段渐进式迁移学习（200条小数据集 + 新噪声类型适配）

阶段1 [epoch 1  ~ phase1_end]: 只训练 FiLM 层（噪声条件适配）
阶段2 [epoch phase1_end+1 ~ phase2_end]: 解冻 Decoder（重建质量提升）
阶段3 [epoch phase2_end+1 ~ epochs]: 解冻 Denoiser Encoder（极低lr细化）
NoiseEncoder: 全程冻结
"""

import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
# 路径
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR= os.path.abspath(os.path.join(THIS_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
for p in [THIS_DIR, V3_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from model_v3 import NoiseAwareDenoiserV3
except ModuleNotFoundError:
    from v3.model_v3 import NoiseAwareDenoiserV3

# ============================================================
# 配置
# ============================================================
CONFIG = {
    # 预训练权重
    "pretrain_ckpt": r"D:/X/denoise/part1/v3/checkpoints_v3/best_model_v3.pth",

    # 小数据集（200条，自动切train/val）
    "finetune_h5":  r"D:/X/denoise/part1/v3/transfer/data/event_100hz_6000.hdf5",
    "finetune_csv": r"D:/X/denoise/part1/v3/transfer/data/event_100hz_6000.csv",

    "noise_h5":     r"D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv":    r"D:/X/p_wave/data/chunk1.csv",

    # 数据量控制
    "max_samples": 200,
    "val_ratio":   0.2,     # 后20%做验证（40条）

    # 加噪参数
    "snr_db_range": (-10.0, 5.0),
    "noise_boost":  1.0,

    # 模型参数
    "z_dim":128,
    "num_heads":  8,
    "cond_len":   400,
    "signal_len": 6000,

    # 三阶段 epoch 边界
    "phase1_end": 5,# epoch 1~5:只训练 FiLM
    "phase2_end": 15,   # epoch 6~15: FiLM + Decoder
    "epochs":     30,   # epoch 16~30:FiLM + Decoder + Enc(极低lr)

    # 各阶段学习率
    "lr_film": 5e-4,
    "lr_dec":  1e-4,
    "lr_enc":  5e-6,

    # 损失权重
    "lambda_wave":1.0,
    "lambda_freq":     0.5,
    "lambda_envelope": 0.3,
    "lambda_quality":  0.1,

    # 训练参数
    "batch_size":4,
    "num_workers":  0,
    "weight_decay": 1e-5,
    "seed":         42,

    # 输出
    "out_dir":      r"D:/X/denoise/part1/v3/checkpoints_transfer_v2",
    "fig_dir":      r"D:/X/denoise/part1/v3/checkpoints_transfer_v2/val_figs",
    "log_json":     r"D:/X/denoise/part1/v3/checkpoints_transfer_v2/train_log.json",

    # 验证时保存波形图数量
    "save_fig_num": 5,
}

# ============================================================
# 数据集
# ============================================================
class FinetuneDataset(Dataset):
    def __init__(
        self,
        event_h5_path,
        event_csv_path,
        noise_h5_path,
        noise_csv_path,
        signal_len=6000,
        cond_len=400,
        snr_db_range=(-10.0, 5.0),
        noise_boost=1.0,
        max_samples=None,
        seed=42,
    ):
        self.event_h5_path = event_h5_path
        self.noise_h5_path = noise_h5_path
        self.signal_len    = signal_len
        self.cond_len      = cond_len
        self.snr_db_range  = snr_db_range
        self.noise_boost   = float(noise_boost)
        self.seed          = int(seed)

        df = pd.read_csv(event_csv_path, low_memory=False)
        if max_samples is not None:
            df = df.iloc[:int(max_samples)].reset_index(drop=True)
        self.event_df = df
        self.noise_df = pd.read_csv(noise_csv_path, low_memory=False)

        self._ev_h5 = None
        self._no_h5 = None
        print(f"[Dataset] events={len(self.event_df)}, noises={len(self.noise_df)}")

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
        if ps < 1e-12or pn < 1e-12:
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
        row        = self.event_df.iloc[idx]
        trace_name = str(row["trace_name"])
        rng= np.random.default_rng(self.seed + idx)

        clean = self._norm_peak(self._load(self.ev_h5, trace_name))

        ni= int(rng.integers(0, len(self.noise_df)))
        noise_name = str(self.noise_df.iloc[ni]["trace_name"])
        noise      = self._norm_peak(self._load(self.no_h5, noise_name))

        snr_db     = float(rng.uniform(self.snr_db_range[0], self.snr_db_range[1]))
        noisy_base = self._mix_snr(clean, noise, snr_db)
        noisy      = clean + self.noise_boost * (noisy_base - clean)

        z_cond = noise[:, :self.cond_len].copy()
        m = np.abs(z_cond).max()
        if m > 1e-10:
            z_cond = z_cond / m

        p_onset    = self._get_p_onset(row)
        valid_mask = np.zeros(self.signal_len, dtype=np.float32)
        valid_mask[p_onset:min(p_onset + 4000, self.signal_len)] = 1.0

        return {
            "clean":      torch.from_numpy(np.clip(clean, -10, 10).astype(np.float32)),
            "noisy":      torch.from_numpy(np.clip(noisy, -10, 10).astype(np.float32)),
            "z_cond":     torch.from_numpy(np.clip(z_cond, -10, 10).astype(np.float32)),
            "valid_mask": torch.from_numpy(valid_mask),
            "trace_name": trace_name,
            "snr_db":     float(snr_db),
        }

# ============================================================
# 损失函数
# ============================================================
def hilbert_envelope(x):
    """
    近似 Hilbert 包络（FFT 实现）
    x: [B, C, T] -> envelope: [B, C, T]
    """
    N= x.shape[-1]
    Xf = torch.fft.rfft(x, dim=-1)

    h = torch.zeros(Xf.shape[-1], device=x.device, dtype=x.dtype)
    h[0] = 1.0
    if N % 2 == 0:
        h[1:N // 2] = 2.0
        h[N // 2]   = 1.0
    else:
        h[1:(N + 1) // 2] = 2.0

    analytic = torch.fft.irfft(Xf * h, n=N, dim=-1)
    envelope = torch.sqrt(x ** 2 + analytic ** 2 + 1e-8)
    return envelope

class TransferLossV2(nn.Module):
    def __init__(self, lw=1.0, lf=0.5, le=0.3, lq=0.1):
        super().__init__()
        self.lw = lw
        self.lf = lf
        self.le = le
        self.lq = lq

    def forward(self, pred, clean, quality, valid_mask):
        mask= valid_mask.unsqueeze(1)# [B,1,T]
        weight = 1.0 + mask#事件窗口权重2倍，背景1倍

        # 波形 L1
        wave_loss = (torch.abs(pred - clean) * weight).mean()

        # 频域 L1
        pred_f= torch.abs(torch.fft.rfft(pred,  dim=-1))
        clean_f = torch.abs(torch.fft.rfft(clean, dim=-1))
        freq_loss = torch.abs(pred_f - clean_f).mean()

        # 包络 L1（对新噪声类型更鲁棒）
        pred_env= hilbert_envelope(pred)
        clean_env = hilbert_envelope(clean)
        env_loss  = (torch.abs(pred_env - clean_env) * weight).mean()

        # quality 正则
        quality_loss = (1.0 - quality).abs().mean()

        total = (
            self.lw * wave_loss
            + self.lf * freq_loss
            + self.le * env_loss
            + self.lq * quality_loss
        )

        return total, {
            "wave":wave_loss.item(),
            "freq":     freq_loss.item(),
            "envelope": env_loss.item(),
            "quality":  quality_loss.item(),
        }

# ============================================================
# 三阶段参数管理
# ============================================================
class PhaseManager:
    def __init__(self, model, cfg):
        self.model         = model
        self.cfg           = cfg
        self.current_phase = 0# 全部先冻结
        for p in model.parameters():
            p.requires_grad = False

        # 按名称分组
        self.film_params = []
        self.dec_params  = []
        self.enc_params  = []

        film_prefixes = ("film",)
        dec_prefixes  = ("dec", "out_conv", "out_act", "quality_head")
        enc_prefixes  = ("enc", "ref", "bn")

        for name, param in model.denoiser.named_parameters():
            prefix = name.split(".")[0]
            if any(prefix.startswith(p) for p in film_prefixes):
                self.film_params.append(param)
            elif any(prefix.startswith(p) for p in dec_prefixes):
                self.dec_params.append(param)
            elif any(prefix.startswith(p) for p in enc_prefixes):
                self.enc_params.append(param)

        # NoiseEncoder 永远冻结
        for p in model.noise_encoder.parameters():
            p.requires_grad = False

        print(f"[PhaseManager] film={len(self.film_params)} | "
              f"dec={len(self.dec_params)} | enc={len(self.enc_params)}")

    def _set_grad(self, params, flag):
        for p in params:
            p.requires_grad = flag

    def enter_phase1(self):
        self._set_grad(self.film_params, True)
        self._set_grad(self.dec_params,  False)
        self._set_grad(self.enc_params,  False)
        self.current_phase = 1
        print("\n" + "=" * 50)
        print(">>> Phase 1：只训练 FiLM 层（噪声条件适配）")
        print("=" * 50)

    def enter_phase2(self, optimizer):
        self._set_grad(self.dec_params, True)
        optimizer.add_param_group({
            "params": self.dec_params,
            "lr":self.cfg["lr_dec"],
            "name":   "dec",
        })
        self.current_phase = 2
        print("\n" + "=" * 50)
        print(">>> Phase 2：解冻 Decoder（重建质量提升）")
        print("=" * 50)

    def enter_phase3(self, optimizer):
        self._set_grad(self.enc_params, True)
        optimizer.add_param_group({
            "params": self.enc_params,
            "lr":     self.cfg["lr_enc"],
            "name":   "enc",
        })
        self.current_phase = 3
        print("\n" + "=" * 50)
        print(">>> Phase 3：解冻 Denoiser Encoder（极低lr细化）")
        print("=" * 50)

    def get_phase(self, epoch):
        if epoch <= self.cfg["phase1_end"]:
            return 1
        elif epoch <= self.cfg["phase2_end"]:
            return 2
        else:
            return 3

# ============================================================
# 工具函数
# ============================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_pretrain(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        sd = (obj.get("state_dict")
              or obj.get("model_state_dict")
              or obj)
    else:
        sd = obj
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[Pretrain] {ckpt_path}")
    print(f"  missing={len(missing)}, unexpected={len(unexpected)}")

def compute_snr(clean, pred, valid_mask):
    m= valid_mask.unsqueeze(1)
    n   = m.sum(dim=[1, 2]) * clean.shape[1] +1e-10
    sig = ((clean ** 2) * m).sum(dim=[1, 2]) / n
    noi = (((pred - clean) ** 2) * m).sum(dim=[1, 2]) / n + 1e-10
    snr = 10.0 * torch.log10(sig / noi)
    return torch.clamp(snr, -50, 50).mean().item()

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cos_lr_scale(phase_epoch, phase_total):
    return 0.5 * (1 + np.cos(np.pi * (phase_epoch - 1) / max(phase_total, 1)))

# ============================================================
# 验证 + 保存波形图
# ============================================================
@torch.no_grad()
def validate(model, loader, criterion, device, epoch, fig_dir, save_n=5):
    model.eval()
    total_loss = 0.0
    total_snr  = 0.0
    n_batch= 0
    fig_saved  = 0

    for batch in loader:
        clean  = batch["clean"].to(device)
        noisy  = batch["noisy"].to(device)
        z_cond = batch["z_cond"].to(device)
        vmask  = batch["valid_mask"].to(device)

        pred, quality, _ = model(noisy, z_cond)
        loss, _ = criterion(pred, clean, quality, vmask)

        total_loss += loss.item()
        total_snr  += compute_snr(clean, pred, vmask)
        n_batch    += 1

        if fig_saved < save_n:
            c_np= clean.cpu().numpy()
            n_np  = noisy.cpu().numpy()
            d_np  = pred.cpu().numpy()
            names = batch["trace_name"]
            snrs  = batch["snr_db"]
            for i in range(min(c_np.shape[0], save_n - fig_saved)):
                _save_wave_fig(
                    c_np[i], n_np[i], d_np[i],
                    name=str(names[i]),
                    snr_db=float(snrs[i]),
                    epoch=epoch,
                    out_dir=fig_dir,
                )
                fig_saved += 1

    return total_loss / max(n_batch, 1), total_snr / max(n_batch, 1)

def _save_wave_fig(clean, noisy, deno, name, snr_db, epoch, out_dir):
    ensure_dir(out_dir)
    safe_name = name.replace("/", "_").replace("\\", "_")
    T= clean.shape[-1]
    t  = np.arange(T)
    ch = ["E", "N", "Z"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 7), sharex=True)
    for j, (data, title) in enumerate(
        zip([clean, noisy, deno], ["Clean", "Noisy", "Denoised"])
    ):
        axes[0, j].set_title(title, fontsize=10)
        for i in range(3):
            ymax = max(np.abs(clean[i]).max(), np.abs(noisy[i]).max(), 1e-6)
            axes[i, j].plot(t, data[i], lw=0.7, color="#1f77b4")
            axes[i, j].set_ylim(-ymax, ymax)
            axes[i, j].grid(alpha=0.2, linestyle="--")
            if j == 0:
                axes[i, j].set_ylabel(ch[i])

    fig.suptitle(
        f"Epoch {epoch} | {safe_name} | SNR_set={snr_db:.1f}dB", fontsize=10
    )
    fig.tight_layout()
    out = os.path.join(out_dir, f"ep{epoch:03d}_{safe_name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)

# ============================================================
# 主函数
# ============================================================
def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    ensure_dir(CONFIG["out_dir"])
    ensure_dir(CONFIG["fig_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── 数据集
    full_ds = FinetuneDataset(
        event_h5_path=CONFIG["finetune_h5"],
        event_csv_path=CONFIG["finetune_csv"],
        noise_h5_path=CONFIG["noise_h5"],
        noise_csv_path=CONFIG["noise_csv"],
        signal_len=CONFIG["signal_len"],
        cond_len=CONFIG["cond_len"],
        snr_db_range=CONFIG["snr_db_range"],
        noise_boost=CONFIG["noise_boost"],
        max_samples=CONFIG["max_samples"],
        seed=CONFIG["seed"],
    )

    n_total = len(full_ds)
    n_val= max(1, int(n_total * CONFIG["val_ratio"]))
    n_train = n_total - n_val
    print(f"[INFO] total={n_total} | train={n_train} | val={n_val}")

    train_ds = Subset(full_ds, list(range(n_train)))
    val_ds   = Subset(full_ds, list(range(n_train, n_total)))

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    # ── 模型 + 预训练权重
    model = NoiseAwareDenoiserV3(
        z_dim=CONFIG["z_dim"],
        cond_len=CONFIG["cond_len"],
        num_heads=CONFIG["num_heads"],
    ).to(device)
    load_pretrain(model, CONFIG["pretrain_ckpt"], device)

    # ── 阶段管理器
    pm = PhaseManager(model, CONFIG)
    pm.enter_phase1()

    # ── 初始 optimizer（只含 FiLM 参数）
    optimizer = torch.optim.AdamW(
        [{"params": pm.film_params, "lr": CONFIG["lr_film"], "name": "film"}],
        weight_decay=CONFIG["weight_decay"],
    )

    criterion = TransferLossV2(
        lw=CONFIG["lambda_wave"],
        lf=CONFIG["lambda_freq"],
        le=CONFIG["lambda_envelope"],
        lq=CONFIG["lambda_quality"],
    )

    best_val_loss = float("inf")
    log = []

    for epoch in range(1, CONFIG["epochs"] + 1):

        # ── 阶段切换
        phase = pm.get_phase(epoch)
        if phase == 2and pm.current_phase == 1:
            pm.enter_phase2(optimizer)
        elif phase == 3 and pm.current_phase == 2:
            pm.enter_phase3(optimizer)

        # ── 当前阶段内余弦 lr衰减
        if phase == 1:
            phase_epoch = epoch
            phase_total = CONFIG["phase1_end"]
        elif phase == 2:
            phase_epoch = epoch - CONFIG["phase1_end"]
            phase_total = CONFIG["phase2_end"] - CONFIG["phase1_end"]
        else:
            phase_epoch = epoch - CONFIG["phase2_end"]
            phase_total = CONFIG["epochs"] - CONFIG["phase2_end"]

        scale = cos_lr_scale(phase_epoch, phase_total)
        for pg in optimizer.param_groups:
            name = pg.get("name", "")
            if name == "film":
                pg["lr"] = CONFIG["lr_film"] * scale
            elif name == "dec":
                pg["lr"] = CONFIG["lr_dec"] * scale
            elif name == "enc":
                pg["lr"] = CONFIG["lr_enc"] * scale

        # ── 训练
        model.train()
        model.noise_encoder.eval()   # NoiseEncoder BN 保持 eval

        tr_loss = 0.0
        tr_snr  = 0.0
        n_b     = 0

        for batch in train_loader:
            clean  = batch["clean"].to(device)
            noisy  = batch["noisy"].to(device)
            z_cond = batch["z_cond"].to(device)
            vmask  = batch["valid_mask"].to(device)

            pred, quality, _ = model(noisy, z_cond)
            loss, sub = criterion(pred, clean, quality, vmask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            tr_loss += loss.item()
            tr_snr  += compute_snr(clean, pred, vmask)
            n_b     += 1

        tr_loss /= max(n_b, 1)
        tr_snr  /= max(n_b, 1)

        # ── 验证
        va_loss, va_snr = validate(
            model, val_loader, criterion, device,
            epoch=epoch,
            fig_dir=CONFIG["fig_dir"],
            save_n=CONFIG["save_fig_num"] if epoch % 5 == 0
        else 0,
        )

        n_trainable = count_trainable(model)

        row = {
            "epoch":epoch,
            "phase":            phase,
            "tr_loss":          round(tr_loss, 6),
            "tr_snr_db":        round(tr_snr,  4),
            "va_loss":          round(va_loss, 6),
            "va_snr_db":        round(va_snr,  4),
            "trainable_params": n_trainable,
            "lr_film":          round(optimizer.param_groups[0]["lr"], 8),
        }
        log.append(row)

        print(
            f"Ep{epoch:03d} [Ph{phase}] "
            f"trainable={n_trainable /1e3:.1f}K | "
            f"tr={tr_loss:.5f}/{tr_snr:.2f}dB | "
            f"va={va_loss:.5f}/{va_snr:.2f}dB"
        )

        # ── 保存最优
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(
                {
                    "epoch":            epoch,
                    "phase":            phase,
                    "model_state_dict": model.state_dict(),
                    "val_loss":         va_loss,
                    "val_snr_db":       va_snr,
                    "config":           CONFIG,
                },
                os.path.join(CONFIG["out_dir"], "best_transfer_v2.pth"),
            )
            print(f"  ✅ Best (va_loss={va_loss:.5f}, va_snr={va_snr:.2f}dB)")

        # ── 阶段结束保存
        if epoch in [CONFIG["phase1_end"], CONFIG["phase2_end"], CONFIG["epochs"]]:
            torch.save(
                {
                    "epoch":            epoch,
                    "phase":            phase,
                    "model_state_dict": model.state_dict(),
                },
                os.path.join(CONFIG["out_dir"], f"phase{phase}_ep{epoch:03d}.pth"),
            )

    # ── 保存日志
    with open(CONFIG["log_json"], "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    # ── 绘制训练曲线
    _plot_log(log, CONFIG["out_dir"])

    print(f"\n========== 迁移微调完成 ==========")
    print(f"最优val_loss={best_val_loss:.5f}")
    print(f"输出目录: {CONFIG['out_dir']}")

def _plot_log(log, out_dir):
    df = pd.DataFrame(log)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Loss 曲线
    axes[0].plot(df["epoch"], df["tr_loss"], label="train loss")
    axes[0].plot(df["epoch"], df["va_loss"], label="val loss")
    for ep, label in [
        (CONFIG["phase1_end"], "Ph1→Ph2"),
        (CONFIG["phase2_end"], "Ph2→Ph3"),
    ]:
        axes[0].axvline(ep, color="gray", linestyle="--", alpha=0.6)
        axes[0].text(ep + 0.2, df["tr_loss"].max() * 0.95, label, fontsize=8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss Curve")
    axes[0].grid(alpha=0.3)

    # SNR 曲线
    axes[1].plot(df["epoch"], df["tr_snr_db"], label="train SNR")
    axes[1].plot(df["epoch"], df["va_snr_db"], label="val SNR")
    for ep in [CONFIG["phase1_end"], CONFIG["phase2_end"]]:
        axes[1].axvline(ep, color="gray", linestyle="--", alpha=0.6)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("SNR (dB)")
    axes[1].legend()
    axes[1].set_title("SNR Curve")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
    plt.close(fig)
    print("[INFO] 训练曲线已保存")

if __name__ == "__main__":
    main()