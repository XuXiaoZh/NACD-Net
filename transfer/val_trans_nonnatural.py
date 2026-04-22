# # -*- coding: utf-8 -*-
# """
# 大数据量迁移学习（15000条）能运行但是效果不好
# 策略：
# - 全量微调（不需要严格渐进冻结）
# - 阶段1[1~10]:冻结 NoiseEncoder + Encoder，只训练 FiLM + Decoder（热身）
# - 阶段2 [11~end]: 全部解冻，整体微调（极低lr保护预训练特征）
# - 数据增强：随机SNR + 随机噪声片段 + 随机翻转
# - 余弦退火 + warmup
# """
#
# import os
# import sys
# import json
# import h5py
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split
# from scipy.signal import resample_poly  # ← 新增
#
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# matplotlib.rcParams["font.family"] = "Times New Roman"
# matplotlib.rcParams["axes.unicode_minus"] = False
#
# # ============================================================
# # 路径
# # ============================================================
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# V3_DIR   = os.path.abspath(os.path.join(THIS_DIR, ".."))
# ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
# for p in [THIS_DIR, V3_DIR, ROOT_DIR]:
#     if p not in sys.path:
#         sys.path.insert(0, p)
#
# try:
#     from model_v3 import NoiseAwareDenoiserV3
# except ModuleNotFoundError:
#     from v3.model_v3 import NoiseAwareDenoiserV3
#
# # ============================================================
# # 配置
# # ============================================================
# CONFIG = {
#     "pretrain_ckpt": r"D:/X/denoise/part1/v3/checkpoints_v3/best_model_v3.pth",
#     "finetune_h5":  r"D:/X/p_wave/data/non_naturaldata.hdf5",
#     "finetune_csv": r"D:/X/p_wave/data/non_naturaldata.csv",
#     "noise_h5":     r"D:/X/p_wave/data/chunk1.hdf5",
#     "noise_csv":    r"D:/X/p_wave/data/chunk1.csv",
#     "max_samples": 15000,
#     "val_ratio":   0.1,
#     "snr_db_range": (-15.0, 10.0),
#     "noise_boost":  1.0,
#     "aug_flip_prob":0.3,
#     "aug_scale_prob":      0.3,
#     "aug_scale_range":     (0.7, 1.3),
#     "aug_noise_shift_prob":0.5,
#     "z_dim":      128,
#     "num_heads":  8,
#     "cond_len":   400,
#     "signal_len": 6000,
#     "warmup_epochs": 10,
#     "epochs":        60,
#     "lr_film":    2e-4,
#     "lr_dec":     1e-4,
#     "lr_enc":     5e-5,
#     "lr_min":     1e-6,
#     "lambda_wave":1.0,
#     "lambda_freq":     0.5,
#     "lambda_envelope": 0.3,
#     "lambda_quality":  0.1,
#     "batch_size":   32,
#     "num_workers":  4,
#     "weight_decay": 1e-5,
#     "grad_clip":    1.0,
#     "seed":         42,
#     "out_dir":      r"D:/X/denoise/part1/v3/checkpoints_transfer_15k",
#     "fig_dir":      r"D:/X/denoise/part1/v3/checkpoints_transfer_15k/val_figs",
#     "log_json":     r"D:/X/denoise/part1/v3/checkpoints_transfer_15k/train_log.json",
#     "save_fig_num": 8,
#     "save_fig_every": 5,
# }
#
# # ============================================================
# # 数据集（含数据增强）
# # ============================================================
# class FinetuneDataset(Dataset):
#     def __init__(
#         self,
#         event_h5_path,
#         event_csv_path,
#         noise_h5_path,
#         noise_csv_path,
#         signal_len=6000,
#         cond_len=400,
#         snr_db_range=(-15.0, 10.0),
#         noise_boost=1.0,
#         max_samples=None,
#         seed=42,
#         augment=True,
#         aug_flip_prob=0.3,
#         aug_scale_prob=0.3,
#         aug_scale_range=(0.7, 1.3),
#         aug_noise_shift_prob=0.5,
#     ):
#         self.event_h5_path = event_h5_path
#         self.noise_h5_path = noise_h5_path
#         self.signal_len    = signal_len
#         self.cond_len      = cond_len
#         self.snr_db_range  = snr_db_range
#         self.noise_boost   = float(noise_boost)
#         self.seed          = int(seed)
#         self.augment       = augment
#
#         self.aug_flip_prob        = aug_flip_prob
#         self.aug_scale_prob       = aug_scale_prob
#         self.aug_scale_range      = aug_scale_range
#         self.aug_noise_shift_prob = aug_noise_shift_prob
#
#         df = pd.read_csv(event_csv_path, low_memory=False)
#         if max_samples is not None:
#             df = df.iloc[:int(max_samples)].reset_index(drop=True)
#         self.event_df = df
#         self.noise_df = pd.read_csv(noise_csv_path, low_memory=False)
#
#         self._ev_h5 = None
#         self._no_h5 = None
#         print(f"[Dataset] events={len(self.event_df)}, noises={len(self.noise_df)}, augment={augment}")
#
#     @property
#     def ev_h5(self):
#         if self._ev_h5 is None:
#             self._ev_h5 = h5py.File(self.event_h5_path, "r")
#         return self._ev_h5
#
#     @property
#     def no_h5(self):
#         if self._no_h5 is None:
#             self._no_h5 = h5py.File(self.noise_h5_path, "r")
#         return self._no_h5
#
#     def __len__(self):
#         return len(self.event_df)
#
#     # ← 改动：原 _load 拆成两个方法
#     def _load_event(self, h5f, name):
#         """加载事件波形，50Hz重采样到100Hz，取第4000~10000个点"""
#         x = h5f["data"][name][:]
#         x = x.T.astype(np.float32)  # (3, T_orig)
#         x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
#
#         # 50Hz → 100Hz
#         x = resample_poly(x, up=2, down=1, axis=1).astype(np.float32)
#
#         T = x.shape[1]
#         start = 4000
#         end = 10000
#
#         if T >= end:
#             self._last_offset = start
#             return x[:, start:end]  # 直接取 [4000, 10000)
#         elif T > start:
#             self._last_offset = start
#             seg = x[:, start:T]  # 取到末尾
#             out = np.zeros((3, self.signal_len), dtype=np.float32)
#             out[:, :seg.shape[1]] = seg
#             return out
#         else:
#             # 信号太短，退回全段
#             self._last_offset = 0
#             out = np.zeros((3, self.signal_len), dtype=np.float32)
#             out[:, :min(T, self.signal_len)] = x[:, :self.signal_len]
#             return out
#     def _load_noise(self, h5f, name):
#         """加载噪声波形（已是100Hz，无需重采样）"""
#         x = h5f["data"][name][:]
#         x = x.T.astype(np.float32)  # (3, T)
#         x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
#
#         T = x.shape[1]
#         if T >= self.signal_len:
#             return x[:, :self.signal_len]
#         out = np.zeros((3, self.signal_len), dtype=np.float32)
#         out[:, :T] = x
#         return out
#
#     @staticmethod
#     def _norm_peak(x):
#         m = np.abs(x).max()
#         return x / m if m > 1e-10 else x
#
#     @staticmethod
#     def _mix_snr(clean, noise, snr_db):
#         snr_lin = 10.0 ** (snr_db / 10.0)
#         ps = np.mean(clean ** 2)
#         pn = np.mean(noise ** 2)
#         if ps < 1e-12 or pn < 1e-12:
#             return clean.copy()
#         scale = float(np.clip(np.sqrt(ps / (snr_lin * pn)), 0.0, 10.0))
#         return clean + scale * noise
#
#     def _get_p_onset(self, row):
#         # ← 改动：采样点×2，从50Hz标注转换到100Hz
#         for col in ["p_arrival_sample", "p_onset", "itp", "Pg"]:
#             if col in row.index and not pd.isna(row[col]):
#                 try:
#                     v = int(row[col]) * 2  # 50Hz → 100Hz
#                     if 0 <= v < self.signal_len:
#                         return v
#                 except Exception:
#                     pass
#         return self.signal_len // 10
#
#     def _augment(self, clean, noise, rng):
#         if self.augment and rng.random() < self.aug_flip_prob:
#             clean = clean[:, ::-1].copy()
#             noise = noise[:, ::-1].copy()
#
#         if self.augment and rng.random() < self.aug_scale_prob:
#             s = rng.uniform(*self.aug_scale_range)
#             clean = clean * s
#
#         if self.augment and rng.random() < self.aug_noise_shift_prob:
#             shift = int(rng.integers(0, max(1, self.signal_len // 4)))
#             noise = np.roll(noise, shift, axis=1)
#
#         return clean, noise
#
#     def __getitem__(self, idx):
#         row        = self.event_df.iloc[idx]
#         trace_name = str(row["trace_name"])
#         rng        = np.random.default_rng(self.seed + idx)
#
#         # ← 改动：分别调用对应的加载方法
#         clean = self._norm_peak(self._load_event(self.ev_h5, trace_name))
#
#         ni         = int(rng.integers(0, len(self.noise_df)))
#         noise_name = str(self.noise_df.iloc[ni]["trace_name"])
#         noise      = self._norm_peak(self._load_noise(self.no_h5, noise_name))
#
#         clean, noise = self._augment(clean, noise, rng)
#
#         snr_db     = float(rng.uniform(self.snr_db_range[0], self.snr_db_range[1]))
#         noisy_base = self._mix_snr(clean, noise, snr_db)
#         noisy      = clean + self.noise_boost * (noisy_base - clean)
#
#         z_cond = noise[:, :self.cond_len].copy()
#         m = np.abs(z_cond).max()
#         if m > 1e-10:
#             z_cond = z_cond / m
#
#         p_onset    = self._get_p_onset(row)
#         valid_mask = np.zeros(self.signal_len, dtype=np.float32)
#         valid_mask[p_onset:min(p_onset + 4000, self.signal_len)] = 1.0
#
#         return {
#             "clean":      torch.from_numpy(np.clip(clean, -10, 10).astype(np.float32)),
#             "noisy":      torch.from_numpy(np.clip(noisy, -10, 10).astype(np.float32)),
#             "z_cond":     torch.from_numpy(np.clip(z_cond, -10, 10).astype(np.float32)),
#             "valid_mask": torch.from_numpy(valid_mask),
#             "trace_name": trace_name,
#             "snr_db":     float(snr_db),
#         }
#
# # ============================================================
# # 损失函数
# # ============================================================
# def hilbert_envelope(x):
#     N  = x.shape[-1]
#     Xf = torch.fft.rfft(x, dim=-1)
#     h  = torch.zeros(Xf.shape[-1], device=x.device, dtype=x.dtype)
#     h[0] = 1.0
#     if N % 2 == 0:
#         h[1:N // 2] = 2.0
#         h[N // 2]   = 1.0
#     else:
#         h[1:(N + 1) // 2] = 2.0
#     analytic = torch.fft.irfft(Xf * h, n=N, dim=-1)
#     return torch.sqrt(x ** 2 + analytic ** 2 + 1e-8)
#
# class TransferLoss(nn.Module):
#     def __init__(self, lw=1.0, lf=0.5, le=0.3, lq=0.1):
#         super().__init__()
#         self.lw = lw
#         self.lf = lf
#         self.le = le
#         self.lq = lq
#
#     def forward(self, pred, clean, quality, valid_mask):
#         mask   = valid_mask.unsqueeze(1)
#         weight = 1.0 + mask
#
#         wave_loss = (torch.abs(pred - clean) * weight).mean()
#
#         freq_loss = torch.abs(
#             torch.abs(torch.fft.rfft(pred, dim=-1)) -
#             torch.abs(torch.fft.rfft(clean, dim=-1))
#         ).mean()
#
#         env_loss = (
#             torch.abs(hilbert_envelope(pred) - hilbert_envelope(clean)) * weight
#         ).mean()
#
#         quality_loss = (1.0 - quality).abs().mean()
#
#         total = (
#             self.lw * wave_loss
#             + self.lf * freq_loss
#             + self.le * env_loss
#             + self.lq * quality_loss
#         )
#         return total, {
#             "wave":     wave_loss.item(),
#             "freq":     freq_loss.item(),
#             "envelope": env_loss.item(),
#             "quality":  quality_loss.item(),
#         }
#
# # ============================================================
# # 工具函数
# # ============================================================
# def ensure_dir(p):
#     os.makedirs(p, exist_ok=True)
#
# def load_pretrain(model, ckpt_path, device):
#     obj = torch.load(ckpt_path, map_location=device)
#     sd  = (obj.get("state_dict") or obj.get("model_state_dict") or obj) \
#           if isinstance(obj, dict) else obj
#     new_sd = {}
#     for k, v in sd.items():
#         if k.startswith("module."):
#             k = k[len("module."):]
#         if k.startswith("model."):
#             k = k[len("model."):]
#         new_sd[k] = v
#     missing, unexpected = model.load_state_dict(new_sd, strict=False)
#     print(f"[Pretrain] missing={len(missing)}, unexpected={len(unexpected)}")
#
# def compute_snr(clean, pred, valid_mask):
#     m   = valid_mask.unsqueeze(1)
#     n   = m.sum(dim=[1, 2]) * clean.shape[1] + 1e-10
#     sig = ((clean ** 2) * m).sum(dim=[1, 2]) / n
#     noi = (((pred - clean) ** 2) * m).sum(dim=[1, 2]) / n + 1e-10
#     return torch.clamp(10.0 * torch.log10(sig / noi), -50, 50).mean().item()
#
# def count_trainable(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# def get_lr(optimizer):
#     return optimizer.param_groups[0]["lr"]
#
# # ============================================================
# # 两阶段参数管理
# # ============================================================
# def setup_phase1(model, cfg):
#     for p in model.parameters():
#         p.requires_grad = False
#
#     for p in model.noise_encoder.parameters():
#         p.requires_grad = False
#
#     film_params = []
#     dec_params  = []
#     enc_params  = []
#
#     film_prefixes = ("film",)
#     dec_prefixes  = ("dec", "out_conv", "out_act", "quality_head")
#     enc_prefixes  = ("enc", "ref", "bn")
#
#     for name, param in model.denoiser.named_parameters():
#         prefix = name.split(".")[0]
#         if any(prefix.startswith(p) for p in film_prefixes):
#             param.requires_grad = True
#             film_params.append(param)
#         elif any(prefix.startswith(p) for p in dec_prefixes):
#             param.requires_grad = True
#             dec_params.append(param)
#         elif any(prefix.startswith(p) for p in enc_prefixes):
#             enc_params.append(param)
#
#     print(f"[Phase1] film={len(film_params)} | dec={len(dec_params)} | "
#           f"enc={len(enc_params)}(冻结)")
#
#     optimizer = torch.optim.AdamW(
#         [
#             {"params": film_params, "lr": cfg["lr_film"], "name": "film"},
#             {"params": dec_params,  "lr": cfg["lr_dec"],  "name": "dec"},
#         ],
#         weight_decay=cfg["weight_decay"],
#     )
#     return optimizer, enc_params
#
# def enter_phase2(model, optimizer, enc_params, cfg):
#     for p in enc_params:
#         p.requires_grad = True
#     optimizer.add_param_group({
#         "params": enc_params,
#         "lr":     cfg["lr_enc"],
#         "name":   "enc",
#     })
#     print("\n" + "=" * 50)
#     print(">>> Phase 2：解冻 Denoiser Encoder，整体微调")
#     print("=" * 50)
#
# # ============================================================
# # 学习率调度
# # ============================================================
# def get_lr_scale(epoch, warmup_epochs, total_epochs, lr_min_ratio=0.01):
#     if epoch <= warmup_epochs:
#         return epoch / warmup_epochs
#     progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
#     cosine   = 0.5 * (1 + np.cos(np.pi * progress))
#     return lr_min_ratio + (1 - lr_min_ratio) * cosine
#
# def apply_lr_scale(optimizer, scale, cfg):
#     base_lrs = {"film": cfg["lr_film"], "dec": cfg["lr_dec"], "enc": cfg["lr_enc"]}
#     for pg in optimizer.param_groups:
#         name = pg.get("name", "")
#         if name in base_lrs:
#             pg["lr"] = base_lrs[name] * scale
#
# # ============================================================
# # 验证
# # ============================================================
# @torch.no_grad()
# def validate(model, loader, criterion, device, epoch, fig_dir, save_n=0):
#     model.eval()
#     total_loss = 0.0
#     total_snr  = 0.0
#     n_batch    = 0
#     fig_saved  = 0
#
#     for batch in loader:
#         clean  = batch["clean"].to(device)
#         noisy  = batch["noisy"].to(device)
#         z_cond = batch["z_cond"].to(device)
#         vmask  = batch["valid_mask"].to(device)
#
#         pred, quality, _ = model(noisy, z_cond)
#         loss, _          = criterion(pred, clean, quality, vmask)
#
#         total_loss += loss.item()
#         total_snr  += compute_snr(clean, pred, vmask)
#         n_batch    += 1
#
#         if fig_saved < save_n:
#             c_np  = clean.cpu().numpy()
#             n_np  = noisy.cpu().numpy()
#             d_np  = pred.cpu().numpy()
#             names = batch["trace_name"]
#             snrs  = batch["snr_db"]
#             for i in range(min(c_np.shape[0], save_n - fig_saved)):
#                 _save_wave_fig(
#                     c_np[i], n_np[i], d_np[i],
#                     name=str(names[i]),
#                     snr_db=float(snrs[i]),
#                     epoch=epoch,
#                     out_dir=fig_dir,
#                 )
#                 fig_saved += 1
#
#     return total_loss / max(n_batch, 1), total_snr / max(n_batch, 1)
#
# def _save_wave_fig(clean, noisy, deno, name, snr_db, epoch, out_dir):
#     ensure_dir(out_dir)
#     safe = name.replace("/", "_").replace("\\", "_")
#     T    = clean.shape[-1]
#     t    = np.arange(T)
#     ch   = ["E", "N", "Z"]
#
#     fig, axes = plt.subplots(3, 3, figsize=(16, 7), sharex=True)
#     for j, (data, title) in enumerate(
#         zip([clean, noisy, deno], ["Clean", "Noisy", "Denoised"])
#     ):
#         axes[0, j].set_title(title, fontsize=10)
#         for i in range(3):
#             ymax = max(np.abs(clean[i]).max(), np.abs(noisy[i]).max(), 1e-6)
#             axes[i, j].plot(t, data[i], lw=0.6, color="#1f77b4")
#             axes[i, j].set_ylim(-ymax, ymax)
#             axes[i, j].grid(alpha=0.2, linestyle="--")
#             if j == 0:
#                 axes[i, j].set_ylabel(ch[i])
#
#     fig.suptitle(f"Ep{epoch} | {safe} | SNR={snr_db:.1f}dB", fontsize=9)
#     fig.tight_layout()
#     fig.savefig(os.path.join(out_dir, f"ep{epoch:03d}_{safe}.png"), dpi=130)
#     plt.close(fig)
#
# # ============================================================
# # 主函数
# # ============================================================
# def main():
#     torch.manual_seed(CONFIG["seed"])
#     np.random.seed(CONFIG["seed"])
#     ensure_dir(CONFIG["out_dir"])
#     ensure_dir(CONFIG["fig_dir"])
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[INFO] Device: {device}")
#
#     full_ds = FinetuneDataset(
#         event_h5_path=CONFIG["finetune_h5"],
#         event_csv_path=CONFIG["finetune_csv"],
#         noise_h5_path=CONFIG["noise_h5"],
#         noise_csv_path=CONFIG["noise_csv"],
#         signal_len=CONFIG["signal_len"],
#         cond_len=CONFIG["cond_len"],
#         snr_db_range=CONFIG["snr_db_range"],
#         noise_boost=CONFIG["noise_boost"],
#         max_samples=CONFIG["max_samples"],
#         seed=CONFIG["seed"],
#         augment=True,
#         aug_flip_prob=CONFIG["aug_flip_prob"],
#         aug_scale_prob=CONFIG["aug_scale_prob"],
#         aug_scale_range=CONFIG["aug_scale_range"],
#         aug_noise_shift_prob=CONFIG["aug_noise_shift_prob"],
#     )
#
#     n_total = len(full_ds)
#     n_val   = max(1, int(n_total * CONFIG["val_ratio"]))
#     n_train = n_total - n_val
#     print(f"[INFO] total={n_total} | train={n_train} | val={n_val}")
#
#     train_ds, val_ds = random_split(
#         full_ds,
#         [n_train, n_val],
#         generator=torch.Generator().manual_seed(CONFIG["seed"]),
#     )
#     val_ds.dataset.augment = False
#
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=CONFIG["batch_size"],
#         shuffle=True,
#         num_workers=CONFIG["num_workers"],
#         pin_memory=True,
#         persistent_workers=CONFIG["num_workers"] > 0,
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=CONFIG["batch_size"],
#         shuffle=False,
#         num_workers=CONFIG["num_workers"],
#         pin_memory=True,
#         persistent_workers=CONFIG["num_workers"] > 0,
#     )
#
#     model = NoiseAwareDenoiserV3(
#         z_dim=CONFIG["z_dim"],
#         cond_len=CONFIG["cond_len"],
#         num_heads=CONFIG["num_heads"],
#     ).to(device)
#     load_pretrain(model, CONFIG["pretrain_ckpt"], device)
#
#     optimizer, enc_params = setup_phase1(model, CONFIG)
#
#     criterion = TransferLoss(
#         lw=CONFIG["lambda_wave"],
#         lf=CONFIG["lambda_freq"],
#         le=CONFIG["lambda_envelope"],
#         lq=CONFIG["lambda_quality"],
#     )
#
#     best_val_loss  = float("inf")
#     log            = []
#     phase2_entered = False
#
#     for epoch in range(1, CONFIG["epochs"] + 1):
#
#         if epoch > CONFIG["warmup_epochs"] and not phase2_entered:
#             enter_phase2(model, optimizer, enc_params, CONFIG)
#             phase2_entered = True
#
#         phase = 1 if epoch <= CONFIG["warmup_epochs"] else 2
#
#         lr_scale = get_lr_scale(
#             epoch,
#             warmup_epochs=CONFIG["warmup_epochs"],
#             total_epochs=CONFIG["epochs"],
#             lr_min_ratio=CONFIG["lr_min"] / CONFIG["lr_film"],
#         )
#         apply_lr_scale(optimizer, lr_scale, CONFIG)
#
#         model.train()
#         model.noise_encoder.eval()
#
#         tr_loss = 0.0
#         tr_snr  = 0.0
#         n_b     = 0
#
#         for batch in train_loader:
#             clean  = batch["clean"].to(device)
#             noisy  = batch["noisy"].to(device)
#             z_cond = batch["z_cond"].to(device)
#             vmask  = batch["valid_mask"].to(device)
#
#             pred, quality, _ = model(noisy, z_cond)
#             loss, sub        = criterion(pred, clean, quality, vmask)
#
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(
#                 [p for p in model.parameters() if p.requires_grad],
#                 max_norm=CONFIG["grad_clip"],
#             )
#             optimizer.step()
#
#             tr_loss += loss.item()
#             tr_snr  += compute_snr(clean, pred, vmask)
#             n_b     += 1
#
#         tr_loss /= max(n_b, 1)
#         tr_snr  /= max(n_b, 1)
#
#         save_n = CONFIG["save_fig_num"] if epoch % CONFIG["save_fig_every"] == 0 else 0
#         va_loss, va_snr = validate(
#             model, val_loader, criterion, device,
#             epoch=epoch, fig_dir=CONFIG["fig_dir"], save_n=save_n,
#         )
#
#         n_trainable = count_trainable(model)
#         cur_lr      = get_lr(optimizer)
#
#         row = {
#             "epoch":            epoch,
#             "phase":            phase,
#             "tr_loss":          round(tr_loss, 6),
#             "tr_snr_db":        round(tr_snr,  4),
#             "va_loss":          round(va_loss, 6),
#             "va_snr_db":        round(va_snr,  4),
#             "trainable_params": n_trainable,
#             "lr":               round(cur_lr, 8),
#         }
#         log.append(row)
#
#         print(
#             f"Ep{epoch:03d} [Ph{phase}] "
#             f"lr={cur_lr:.2e} | "
#             f"trainable={n_trainable / 1e6:.2f}M | "
#             f"tr={tr_loss:.5f}/{tr_snr:.2f}dB | "
#             f"va={va_loss:.5f}/{va_snr:.2f}dB"
#         )
#
#         if va_loss < best_val_loss:
#             best_val_loss = va_loss
#             torch.save(
#                 {
#                     "epoch":            epoch,
#                     "phase":            phase,
#                     "model_state_dict": model.state_dict(),
#                     "optimizer":        optimizer.state_dict(),
#                     "val_loss":         va_loss,
#                     "val_snr_db":       va_snr,
#                     "config":           CONFIG,
#                 },
#                 os.path.join(CONFIG["out_dir"], "best_transfer_15k.pth"),
#             )
#             print(f"  ✅ Best saved (va_loss={va_loss:.5f}, SNR={va_snr:.2f}dB)")
#
#         if epoch % 10 == 0:
#             torch.save(
#                 {
#                     "epoch":            epoch,
#                     "model_state_dict": model.state_dict(),
#                     "optimizer":        optimizer.state_dict(),
#                 },
#                 os.path.join(CONFIG["out_dir"], f"ckpt_ep{epoch:03d}.pth"),
#             )
#
#     with open(CONFIG["log_json"], "w", encoding="utf-8") as f:
#         json.dump(log, f, indent=2, ensure_ascii=False)
#
#     _plot_log(log, CONFIG["out_dir"])
#
#     print(f"\n========== 迁移微调完成 ==========")
#     print(f"最优val_loss = {best_val_loss:.5f}")
#     print(f"输出目录: {CONFIG['out_dir']}")
#
# def _plot_log(log, out_dir):
#     df = pd.DataFrame(log)
#     fig, axes = plt.subplots(1, 2, figsize=(14, 4))
#
#     axes[0].plot(df["epoch"], df["tr_loss"], label="train loss")
#     axes[0].plot(df["epoch"], df["va_loss"], label="val loss")
#     axes[0].axvline(CONFIG["warmup_epochs"], color="gray",
#                     linestyle="--", alpha=0.6, label="Ph1→Ph2")
#     axes[0].set_xlabel("Epoch")
#     axes[0].set_ylabel("Loss")
#     axes[0].legend()
#     axes[0].set_title("Loss Curve")
#     axes[0].grid(alpha=0.3)
#
#     axes[1].plot(df["epoch"], df["tr_snr_db"], label="train SNR")
#     axes[1].plot(df["epoch"], df["va_snr_db"], label="val SNR")
#     axes[1].axvline(CONFIG["warmup_epochs"], color="gray",
#                     linestyle="--", alpha=0.6)
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("SNR (dB)")
#     axes[1].legend()
#     axes[1].set_title("SNR Curve")
#     axes[1].grid(alpha=0.3)
#
#     fig.tight_layout()
#     fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
#     plt.close(fig)
#     print("[INFO] 训练曲线已保存")
#
#
# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
"""
大数据量迁移学习（15000条）- 修复版
主要修复：
1. quality_loss 改为用真实SNR增益构造软标签监督
2. 增加幅度一致性损失，防止信号被过度压制
3. STA/LTA 拾取前对去噪波形做峰值归一化
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
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import resample_poly

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR   = os.path.abspath(os.path.join(THIS_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
for p in [THIS_DIR, V3_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from model_v3 import NoiseAwareDenoiserV3
except ModuleNotFoundError:
    from v3.model_v3 import NoiseAwareDenoiserV3

CONFIG = {
    "pretrain_ckpt": r"D:/X/denoise/part1/v3/checkpoints_v3/best_model_v3.pth",
    "finetune_h5":  r"D:/X/p_wave/data/non_naturaldata.hdf5",
    "finetune_csv": r"D:/X/p_wave/data/non_naturaldata.csv",
    "noise_h5":     r"D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv":    r"D:/X/p_wave/data/chunk1.csv",
    "max_samples": 15000,
    "val_ratio":   0.1,
    "snr_db_range": (-15.0, 10.0),
    "noise_boost":  1.0,
    "aug_flip_prob":       0.3,
    "aug_scale_prob":      0.3,
    "aug_scale_range":     (0.7, 1.3),
    "aug_noise_shift_prob":0.5,
    "z_dim":      128,
    "num_heads":  8,
    "cond_len":   400,
    "signal_len": 6000,
    "warmup_epochs": 10,
    "epochs":        60,
    "lr_film":    2e-4,
    "lr_dec":     1e-4,
    "lr_enc":     5e-5,
    "lr_min":     1e-6,
    "lambda_wave":     1.0,
    "lambda_freq":     0.5,
    "lambda_envelope": 0.3,
    "lambda_quality":  0.2,   # 修复：提高quality loss权重
    "lambda_amp":      0.2,   # 新增：幅度一致性损失
    "batch_size":   32,
    "num_workers":  4,
    "weight_decay": 1e-5,
    "grad_clip":    1.0,
    "seed":         42,
    "out_dir":      r"D:/X/denoise/part1/v3/checkpoints_transfer_15k",
    "fig_dir":      r"D:/X/denoise/part1/v3/checkpoints_transfer_15k/val_figs",
    "log_json":     r"D:/X/denoise/part1/v3/checkpoints_transfer_15k/train_log.json",
    "save_fig_num": 8,
    "save_fig_every": 5,
    "pre_p_samples": 500,  # P波前保留的点数
    "post_p_samples": 5500,  # P波后保留的点数
}

class FinetuneDataset(Dataset):
    def __init__(
        self,
        event_h5_path,
        event_csv_path,
        noise_h5_path,
        noise_csv_path,
        signal_len=6000,
        cond_len=400,
        snr_db_range=(-15.0, 10.0),
        noise_boost=1.0,
        max_samples=None,
        seed=42,
        augment=True,
        aug_flip_prob=0.3,
        aug_scale_prob=0.3,
        aug_scale_range=(0.7, 1.3),
        aug_noise_shift_prob=0.5,
    ):
        self.event_h5_path = event_h5_path
        self.noise_h5_path = noise_h5_path
        self.signal_len    = signal_len
        self.cond_len      = cond_len
        self.snr_db_range  = snr_db_range
        self.noise_boost   = float(noise_boost)
        self.seed          = int(seed)
        self.augment       = augment
        self.aug_flip_prob        = aug_flip_prob
        self.aug_scale_prob       = aug_scale_prob
        self.aug_scale_range      = aug_scale_range
        self.aug_noise_shift_prob = aug_noise_shift_prob

        df = pd.read_csv(event_csv_path, low_memory=False)
        if max_samples is not None:
            df = df.iloc[:int(max_samples)].reset_index(drop=True)
        self.event_df = df
        self.noise_df = pd.read_csv(noise_csv_path, low_memory=False)
        self._ev_h5 = None
        self._no_h5 = None
        print(f"[Dataset] events={len(self.event_df)}, noises={len(self.noise_df)}, augment={augment}")

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
        T = x.shape[1]
        start, end = 4000, 10000
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

    def _get_p_onset(self, row):
        for col in ["p_arrival_sample", "p_onset", "itp", "Pg"]:
            if col in row.index and not pd.isna(row[col]):
                try:
                    v = int(row[col]) * 2
                    if 0 <= v < self.signal_len:
                        return v
                except Exception:
                    pass
        return self.signal_len // 10

    def _augment(self, clean, noise, rng):
        if self.augment and rng.random() < self.aug_flip_prob:
            clean = clean[:, ::-1].copy()
            noise = noise[:, ::-1].copy()
        if self.augment and rng.random() < self.aug_scale_prob:
            s = rng.uniform(*self.aug_scale_range)
            clean = clean * s
        if self.augment and rng.random() < self.aug_noise_shift_prob:
            shift = int(rng.integers(0, max(1, self.signal_len // 4)))
            noise = np.roll(noise, shift, axis=1)
        return clean, noise

    def __getitem__(self, idx):
        row        = self.event_df.iloc[idx]
        trace_name = str(row["trace_name"])
        rng        = np.random.default_rng(self.seed + idx)

        clean = self._norm_peak(self._load_event(self.ev_h5, trace_name))
        noise_idx  = int(rng.integers(0, len(self.noise_df)))
        noise_name = str(self.noise_df.iloc[noise_idx]["trace_name"])
        noise      = self._norm_peak(self._load_noise(self.no_h5, noise_name))

        clean, noise = self._augment(clean, noise, rng)

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

def hilbert_envelope(x):
    N  = x.shape[-1]
    Xf = torch.fft.rfft(x, dim=-1)
    h  = torch.zeros(Xf.shape[-1], device=x.device, dtype=x.dtype)
    h[0] = 1.0
    if N % 2 == 0:
        h[1:N // 2] = 2.0
        h[N // 2]   = 1.0
    else:
        h[1:(N + 1) // 2] = 2.0
    analytic = torch.fft.irfft(Xf * h, n=N, dim=-1)
    return torch.sqrt(x ** 2 + analytic ** 2 + 1e-8)

def compute_snr_gain_tensor(clean, noisy, pred):
    """
    计算每个样本的 SNR 增益（dB），返回 [B] tensor
    用于构造 quality 软标签
    """
    eps = 1e-10
    # SNR_in: noisy vs clean
    sig_pow  = (clean ** 2).mean(dim=[1, 2])
    noise_in = ((noisy - clean) ** 2).mean(dim=[1, 2])
    snr_in   = 10.0 * torch.log10((sig_pow + eps) / (noise_in + eps))

    # SNR_out: pred vs clean
    noise_out = ((pred - clean) ** 2).mean(dim=[1, 2])
    snr_out   = 10.0 * torch.log10((sig_pow + eps) / (noise_out + eps))

    return snr_out - snr_in   # [B]，正值表示有改善

class TransferLoss(nn.Module):
    def __init__(self, lw=1.0, lf=0.5, le=0.3, lq=0.2, la=0.2):
        super().__init__()
        self.lw = lw
        self.lf = lf
        self.le = le
        self.lq = lq
        self.la = la

    def forward(self, pred, clean, noisy, quality, valid_mask):
        mask   = valid_mask.unsqueeze(1)
        weight = 1.0 + mask

        # 波形损失
        wave_loss = (torch.abs(pred - clean) * weight).mean()

        # 频域损失
        freq_loss = torch.abs(
            torch.abs(torch.fft.rfft(pred, dim=-1)) -
            torch.abs(torch.fft.rfft(clean, dim=-1))
        ).mean()

        # 包络损失
        env_loss = (
            torch.abs(hilbert_envelope(pred) - hilbert_envelope(clean)) * weight
        ).mean()

        # ── 修复：quality loss 用真实SNR增益构造软标签 ──────────
        with torch.no_grad():
            snr_gain      = compute_snr_gain_tensor(clean, noisy, pred.detach())
            # sigmoid 映射到 [0,1]，增益 0dB → 0.5，+10dB → ~0.99，-10dB → ~0.01
            quality_target = torch.sigmoid(snr_gain / 5.0)

        quality_loss = F.mse_loss(quality, quality_target)

        # ── 新增：幅度一致性损失，防止信号被过度压制 ────────────
        pred_amp  = pred.abs().mean(dim=-1)    # [B, 3]
        clean_amp = clean.abs().mean(dim=-1)   # [B, 3]
        amp_loss  = F.mse_loss(pred_amp, clean_amp)

        total = (
            self.lw * wave_loss
            + self.lf * freq_loss
            + self.le * env_loss
            + self.lq * quality_loss
            + self.la * amp_loss
        )
        return total, {
            "wave":     wave_loss.item(),
            "freq":     freq_loss.item(),
            "envelope": env_loss.item(),
            "quality":  quality_loss.item(),
            "amp":      amp_loss.item(),
        }

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_pretrain(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device)
    sd  = (obj.get("state_dict") or obj.get("model_state_dict") or obj) \
          if isinstance(obj, dict) else obj
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[Pretrain] missing={len(missing)}, unexpected={len(unexpected)}")

def compute_snr(clean, pred, valid_mask):
    m   = valid_mask.unsqueeze(1)
    n   = m.sum(dim=[1, 2]) * clean.shape[1] + 1e-10
    sig = ((clean ** 2) * m).sum(dim=[1, 2]) / n
    noi = (((pred - clean) ** 2) * m).sum(dim=[1, 2]) / n + 1e-10
    return torch.clamp(10.0 * torch.log10(sig / noi), -50, 50).mean().item()

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def setup_phase1(model, cfg):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.noise_encoder.parameters():
        p.requires_grad = False

    film_params, dec_params, enc_params = [], [], []
    film_prefixes = ("film",)
    dec_prefixes  = ("dec", "out_conv", "out_act", "quality_head")
    enc_prefixes  = ("enc", "ref", "bn")

    for name, param in model.denoiser.named_parameters():
        prefix = name.split(".")[0]
        if any(prefix.startswith(p) for p in film_prefixes):
            param.requires_grad = True
            film_params.append(param)
        elif any(prefix.startswith(p) for p in dec_prefixes):
            param.requires_grad = True
            dec_params.append(param)
        elif any(prefix.startswith(p) for p in enc_prefixes):
            enc_params.append(param)

    print(f"[Phase1] film={len(film_params)} | dec={len(dec_params)} | enc={len(enc_params)}(冻结)")

    optimizer = torch.optim.AdamW(
        [
            {"params": film_params, "lr": cfg["lr_film"], "name": "film"},
            {"params": dec_params,  "lr": cfg["lr_dec"],  "name": "dec"},
        ],
        weight_decay=cfg["weight_decay"],
    )
    return optimizer, enc_params

def enter_phase2(model, optimizer, enc_params, cfg):
    for p in enc_params:
        p.requires_grad = True
    optimizer.add_param_group({
        "params": enc_params,
        "lr":     cfg["lr_enc"],
        "name":   "enc",
    })
    print("\n" + "=" * 50)
    print(">>> Phase 2：解冻 Denoiser Encoder，整体微调")
    print("=" * 50)

def get_lr_scale(epoch, warmup_epochs, total_epochs, lr_min_ratio=0.01):
    if epoch <= warmup_epochs:
        return epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    cosine   = 0.5 * (1 + np.cos(np.pi * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cosine

def apply_lr_scale(optimizer, scale, cfg):
    base_lrs = {"film": cfg["lr_film"], "dec": cfg["lr_dec"], "enc": cfg["lr_enc"]}
    for pg in optimizer.param_groups:
        name = pg.get("name", "")
        if name in base_lrs:
            pg["lr"] = base_lrs[name] * scale

@torch.no_grad()
def validate(model, loader, criterion, device, epoch, fig_dir, save_n=0):
    model.eval()
    total_loss = 0.0
    total_snr  = 0.0
    n_batch    = 0
    fig_saved  = 0

    for batch in loader:
        clean  = batch["clean"].to(device)
        noisy  = batch["noisy"].to(device)
        z_cond = batch["z_cond"].to(device)
        vmask  = batch["valid_mask"].to(device)

        pred, quality, _ = model(noisy, z_cond)
        loss, _          = criterion(pred, clean, noisy, quality, vmask)

        total_loss += loss.item()
        total_snr  += compute_snr(clean, pred, vmask)
        n_batch    += 1

        if fig_saved < save_n:
            c_np  = clean.cpu().numpy()
            n_np  = noisy.cpu().numpy()
            d_np  = pred.cpu().numpy()
            q_np  = quality.cpu().numpy()
            names = batch["trace_name"]
            snrs  = batch["snr_db"]
            for i in range(min(c_np.shape[0], save_n - fig_saved)):
                _save_wave_fig(
                    c_np[i], n_np[i], d_np[i],
                    name=str(names[i]),
                    snr_db=float(snrs[i]),
                    quality=float(q_np[i]),
                    epoch=epoch,
                    out_dir=fig_dir,
                )
                fig_saved += 1

    return total_loss / max(n_batch, 1), total_snr / max(n_batch, 1)

def _save_wave_fig(clean, noisy, deno, name, snr_db, quality, epoch, out_dir):
    ensure_dir(out_dir)
    safe = name.replace("/", "_").replace("\\", "_")
    T    = clean.shape[-1]
    t    = np.arange(T)
    ch   = ["E", "N", "Z"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 7), sharex=True)
    for j, (data, title) in enumerate(
        zip([clean, noisy, deno], ["Clean", "Noisy", "Denoised"])
    ):
        axes[0, j].set_title(title, fontsize=10)
        for i in range(3):
            ymax = max(np.abs(clean[i]).max(), np.abs(noisy[i]).max(), 1e-6)
            axes[i, j].plot(t, data[i], lw=0.6, color="#1f77b4")
            axes[i, j].set_ylim(-ymax, ymax)
            axes[i, j].grid(alpha=0.2, linestyle="--")
            if j == 0:
                axes[i, j].set_ylabel(ch[i])

    fig.suptitle(f"Ep{epoch} | {safe} | SNR={snr_db:.1f}dB | Q={quality:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"ep{epoch:03d}_{safe}.png"), dpi=130)
    plt.close(fig)

def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    ensure_dir(CONFIG["out_dir"])
    ensure_dir(CONFIG["fig_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

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
        augment=True,
        aug_flip_prob=CONFIG["aug_flip_prob"],
        aug_scale_prob=CONFIG["aug_scale_prob"],
        aug_scale_range=CONFIG["aug_scale_range"],
        aug_noise_shift_prob=CONFIG["aug_noise_shift_prob"],
    )

    n_total = len(full_ds)
    n_val   = max(1, int(n_total * CONFIG["val_ratio"]))
    n_train = n_total - n_val
    print(f"[INFO] total={n_total} | train={n_train} | val={n_val}")

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=CONFIG["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=CONFIG["num_workers"] > 0,
    )

    model = NoiseAwareDenoiserV3(
        z_dim=CONFIG["z_dim"],
        cond_len=CONFIG["cond_len"],
        num_heads=CONFIG["num_heads"],
    ).to(device)
    load_pretrain(model, CONFIG["pretrain_ckpt"], device)

    optimizer, enc_params = setup_phase1(model, CONFIG)

    criterion = TransferLoss(
        lw=CONFIG["lambda_wave"],
        lf=CONFIG["lambda_freq"],
        le=CONFIG["lambda_envelope"],
        lq=CONFIG["lambda_quality"],
        la=CONFIG["lambda_amp"],
    )

    best_val_loss  = float("inf")
    log            = []
    phase2_entered = False

    for epoch in range(1, CONFIG["epochs"] + 1):

        if epoch > CONFIG["warmup_epochs"] and not phase2_entered:
            enter_phase2(model, optimizer, enc_params, CONFIG)
            phase2_entered = True

        phase = 1 if epoch <= CONFIG["warmup_epochs"] else 2

        lr_scale = get_lr_scale(
            epoch,
            warmup_epochs=CONFIG["warmup_epochs"],
            total_epochs=CONFIG["epochs"],
            lr_min_ratio=CONFIG["lr_min"] / CONFIG["lr_film"],
        )
        apply_lr_scale(optimizer, lr_scale, CONFIG)

        model.train()
        model.noise_encoder.eval()

        tr_loss = 0.0
        tr_snr  = 0.0
        sub_totals = {"wave": 0.0, "freq": 0.0, "envelope": 0.0, "quality": 0.0, "amp": 0.0}
        n_b = 0

        for batch in train_loader:
            clean  = batch["clean"].to(device)
            noisy  = batch["noisy"].to(device)
            z_cond = batch["z_cond"].to(device)
            vmask  = batch["valid_mask"].to(device)

            pred, quality, _ = model(noisy, z_cond)
            # 修复：传入 noisy 用于构造 quality 软标签
            loss, sub = criterion(pred, clean, noisy, quality, vmask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=CONFIG["grad_clip"],
            )
            optimizer.step()

            tr_loss += loss.item()
            tr_snr  += compute_snr(clean, pred, vmask)
            for k in sub_totals:
                sub_totals[k] += sub[k]
            n_b += 1

        tr_loss /= max(n_b, 1)
        tr_snr  /= max(n_b, 1)
        for k in sub_totals:
            sub_totals[k] /= max(n_b, 1)

        save_n = CONFIG["save_fig_num"] if epoch % CONFIG["save_fig_every"] == 0 else 0
        va_loss, va_snr = validate(
            model, val_loader, criterion, device,
            epoch=epoch, fig_dir=CONFIG["fig_dir"], save_n=save_n,
        )

        n_trainable = count_trainable(model)
        cur_lr      = get_lr(optimizer)

        row = {
            "epoch":            epoch,
            "phase":            phase,
            "tr_loss":          round(tr_loss, 6),
            "tr_snr_db":        round(tr_snr,  4),
            "va_loss":          round(va_loss, 6),
            "va_snr_db":        round(va_snr,  4),
            "trainable_params": n_trainable,
            "lr":               round(cur_lr, 8),
            "sub_wave":         round(sub_totals["wave"],     6),
            "sub_freq":         round(sub_totals["freq"],     6),
            "sub_envelope":     round(sub_totals["envelope"], 6),
            "sub_quality":      round(sub_totals["quality"],  6),
            "sub_amp":          round(sub_totals["amp"],      6),
        }
        log.append(row)

        print(
            f"Ep{epoch:03d} [Ph{phase}] "
            f"lr={cur_lr:.2e} | "
            f"trainable={n_trainable / 1e6:.2f}M | "
            f"tr={tr_loss:.5f}/{tr_snr:.2f}dB | "
            f"va={va_loss:.5f}/{va_snr:.2f}dB | "
            f"wave={sub_totals['wave']:.4f} "
            f"freq={sub_totals['freq']:.4f} "
            f"env={sub_totals['envelope']:.4f} "
            f"qual={sub_totals['quality']:.4f} "
            f"amp={sub_totals['amp']:.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(
                {
                    "epoch":            epoch,
                    "phase":            phase,
                    "model_state_dict": model.state_dict(),
                    "optimizer":        optimizer.state_dict(),
                    "val_loss":         va_loss,
                    "val_snr_db":       va_snr,
                    "config":           CONFIG,
                },
                os.path.join(CONFIG["out_dir"], "best_transfer_15k.pth"),
            )
            print(f"  ✅ Best saved (va_loss={va_loss:.5f}, SNR={va_snr:.2f}dB)")

        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer":        optimizer.state_dict(),
                },
                os.path.join(CONFIG["out_dir"], f"ckpt_ep{epoch:03d}.pth"),
            )

    with open(CONFIG["log_json"], "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    _plot_log(log, CONFIG["out_dir"])

    print(f"\n========== 迁移微调完成 ==========")
    print(f"最优val_loss = {best_val_loss:.5f}")
    print(f"输出目录: {CONFIG['out_dir']}")

def _plot_log(log, out_dir):
    df = pd.DataFrame(log)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # Loss 曲线
    axes[0, 0].plot(df["epoch"], df["tr_loss"], label="train loss")
    axes[0, 0].plot(df["epoch"], df["va_loss"], label="val loss")
    axes[0, 0].axvline(CONFIG["warmup_epochs"], color="gray",
                       linestyle="--", alpha=0.6, label="Ph1→Ph2")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].grid(alpha=0.3)

    # SNR 曲线
    axes[0, 1].plot(df["epoch"], df["tr_snr_db"], label="train SNR")
    axes[0, 1].plot(df["epoch"], df["va_snr_db"], label="val SNR")
    axes[0, 1].axvline(CONFIG["warmup_epochs"], color="gray",
                       linestyle="--", alpha=0.6)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("SNR (dB)")
    axes[0, 1].legend()
    axes[0, 1].set_title("SNR Curve")
    axes[0, 1].grid(alpha=0.3)

    # 子损失：wave + freq
    axes[0, 2].plot(df["epoch"], df["sub_wave"],  label="wave")
    axes[0, 2].plot(df["epoch"], df["sub_freq"],  label="freq")
    axes[0, 2].plot(df["epoch"], df["sub_envelope"], label="envelope")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].legend()
    axes[0, 2].set_title("Wave / Freq / Envelope Loss")
    axes[0, 2].grid(alpha=0.3)

    # Quality loss 曲线（关键：应该从高到低收敛，不能趋近于0）
    axes[1, 0].plot(df["epoch"], df["sub_quality"], label="quality loss", color="purple")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].set_title("Quality Loss (MSE vs SNR-gain target)")
    axes[1, 0].grid(alpha=0.3)

    # Amp loss 曲线
    axes[1, 1].plot(df["epoch"], df["sub_amp"], label="amp loss", color="orange")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].set_title("Amplitude Consistency Loss")
    axes[1, 1].grid(alpha=0.3)

    # LR 曲线
    axes[1, 2].plot(df["epoch"], df["lr"], label="lr", color="green")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Learning Rate")
    axes[1, 2].legend()
    axes[1, 2].set_title("Learning Rate Schedule")
    axes[1, 2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
    plt.close(fig)
    print("[INFO] 训练曲线已保存")

if __name__ == "__main__":
    main()