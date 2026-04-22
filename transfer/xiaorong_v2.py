# # -*- coding: utf-8 -*-
# """
# 实验2：冻结策略消融实验
# 比较4种冻结策略在 non_naturaldata 上的迁移效果
# """
#
# import os, sys, h5py, json
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# for p in [THIS_DIR, os.path.abspath(os.path.join(THIS_DIR, ".."))]:
#     if p not in sys.path:
#         sys.path.insert(0, p)
#
# from model_v3 import NoiseAwareDenoiserV3
#
# # ============================================================
# # CONFIG
# # ============================================================
# CONFIG = {
#     "pretrain_ckpt":  r"D:/X/denoise/part1/v3/checkpoints_v3/best_model_v3.pth",
#     "finetune_h5":    r"D:/X/p_wave/data/non_naturaldata.hdf5",
#     "finetune_csv":   r"D:/X/p_wave/data/non_naturaldata.csv",
#     "noise_h5":       r"D:/X/p_wave/data/chunk1.hdf5",
#     "noise_csv":      r"D:/X/p_wave/data/chunk1.csv",
#     "output_dir":     r"D:/X/denoise/part1/v3/exp2_freeze_fixed",
#     "pick_ckpt":      r"D:/X/p_wave/output/Stanford/12.7rightdams/model_unet_mpt.pt",
#     "pick_threshold": 0.5,
#     "max_samples":    5000,
#     "val_ratio":      0.15,
#     "signal_len":     6000,
#     "cond_len":       400,
#     "z_dim":          128,
#     "num_heads":      8,
#     "snr_db_range":   (-15.0, 10.0),
#     "noise_boost":    1.0,
#     "batch_size":     16,
#     "num_workers":    0,
#     "epochs":         50,
#     "lr":             1e-4,
#     "seed":           42,
#     "sta_len":        0.5,
#     "lta_len":        4.0,
#     "stalta_thr":     2.5,
#     "pick_tol":       50,
#     "fs":             100,
# }
#
# EPS = 1e-10
#
# # ============================================================
# # 冻结策略定义（修正版）
# # ============================================================
# FREEZE_STRATEGIES = [
#     {
#         "name":    "full_freeze",
#         "display": "全冻结（只训练输出层）",
#         "unfreeze": ["quality_head", "out_conv"],
#     },
#     {
#         "name":    "freeze_encoder",
#         "display": "冻结Encoder（训练FiLM+Decoder）",
#         "unfreeze": ["film", "dec", "quality_head", "out_conv"],
#     },
#     {
#         "name":    "freeze_noise_encoder",
#         "display": "冻结NoiseEncoder（训练主干）",
#         "unfreeze": ["denoiser"],
#     },
#     {
#         "name":    "full_unfreeze",
#         "display": "全解冻（整体微调）",
#         "unfreeze": ["all"],
#     },
# ]
#
# # ============================================================
# # Dataset
# # ============================================================
# class FinetuneDataset(Dataset):
#     def __init__(self, event_h5_path, event_csv_path,
#                  noise_h5_path, noise_csv_path,
#                  signal_len=6000, cond_len=400,
#                  snr_db_range=(-15.0, 10.0),
#                  noise_boost=1.0, max_samples=None, seed=42):
#         self.event_h5_path = event_h5_path
#         self.noise_h5_path = noise_h5_path
#         self.signal_len    = signal_len
#         self.cond_len      = cond_len
#         self.snr_db_range  = snr_db_range
#         self.noise_boost   = float(noise_boost)
#         self.seed          = int(seed)
#         self._ev_h5        = None
#         self._no_h5        = None
#
#         df = pd.read_csv(event_csv_path, low_memory=False)
#         if max_samples is not None:
#             df = df.iloc[:int(max_samples)].reset_index(drop=True)
#         self.event_df = df
#         self.noise_df = pd.read_csv(noise_csv_path, low_memory=False)
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
#     def _get_p_sample(self, row):
#         for col in ["trace_P_arrival_sample", "Pg"]:
#             if col in row.index and not pd.isna(row[col]):
#                 try:
#                     return int(float(row[col]))
#                 except Exception:
#                     pass
#         return 2250
#
#     def _load_event(self, h5f, name, p_sample, pre_p=500):
#         x = h5f["data"][name][:]
#         x = x.T.astype(np.float32)
#         x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
#         T = x.shape[1]
#         start = max(0, p_sample - pre_p)
#         end   = start + self.signal_len
#         if end > T:
#             end   = T
#             start = max(0, end - self.signal_len)
#         seg  = x[:, start:end]
#         if seg.shape[1] < self.signal_len:
#             out = np.zeros((3, self.signal_len), dtype=np.float32)
#             out[:, :seg.shape[1]] = seg
#             seg = out
#         p_rel = int(np.clip(p_sample - start, 0, self.signal_len - 1))
#         return seg, p_rel
#
#     def _load_noise(self, h5f, name):
#         x = h5f["data"][name][:]
#         x = x.T.astype(np.float32)
#         x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
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
#     def __getitem__(self, idx):
#         row        = self.event_df.iloc[idx]
#         trace_name = str(row["trace_name"])
#         rng        = np.random.default_rng(self.seed + idx)
#         p_sample   = self._get_p_sample(row)
#
#         clean, p_rel = self._load_event(self.ev_h5, trace_name, p_sample)
#         clean = self._norm_peak(clean)
#
#         ni         = int(rng.integers(0, len(self.noise_df)))
#         noise_name = str(self.noise_df.iloc[ni]["trace_name"])
#         noise      = self._norm_peak(self._load_noise(self.no_h5, noise_name))
#
#         snr_db_val  = float(rng.uniform(self.snr_db_range[0], self.snr_db_range[1]))
#         noisy_base  = self._mix_snr(clean, noise, snr_db_val)
#         noisy       = clean + self.noise_boost * (noisy_base - clean)
#
#         z_cond = noise[:, :self.cond_len].copy()
#         m = np.abs(z_cond).max()
#         if m > 1e-10:
#             z_cond = z_cond / m
#
#         return {
#             "clean":   torch.from_numpy(np.clip(clean, -10, 10).astype(np.float32)),
#             "noisy":   torch.from_numpy(np.clip(noisy, -10, 10).astype(np.float32)),
#             "z_cond":  torch.from_numpy(np.clip(z_cond, -10, 10).astype(np.float32)),
#             "p_onset": p_rel,
#             "snr_db":  snr_db_val,
#         }
#
# # ============================================================
# # 指标函数
# # ============================================================
# def snr_db_fn(clean, test):
#     sig = np.sum(clean ** 2)
#     noi = np.sum((test - clean) ** 2)
#     return float(10.0 * np.log10((sig + EPS) / (noi + EPS)))
#
# def cc_fn(clean, test):
#     c = clean - clean.mean()
#     t = test  - test.mean()
#     d = np.sqrt(np.sum(c**2) * np.sum(t**2))
#     return float(np.sum(c * t) / d) if d > EPS else 0.0
#
# def rmse_fn(clean, test):
#     return float(np.sqrt(np.mean((clean - test) ** 2)))
#
# def prd_fn(clean, test):
#     return float(np.sqrt(
#         np.sum((clean - test) ** 2) / (np.sum(clean ** 2) + EPS)
#     ))
#
# def st_mae_mean(clean, test, fs=100, win_ms=100, overlap=0.5):
#     n   = min(len(clean), len(test))
#     win = min(int(fs * win_ms / 1000), n)
#     if win <= 0:
#         return float('nan')
#     hop  = max(1, int(win * (1.0 - overlap)))
#     vals = [np.abs(clean[s:s + win] - test[s:s + win]).mean()
#             for s in range(0, n - win + 1, hop)]
#     return float(np.mean(vals)) if vals else float('nan')
#
# def stalta_pick(wave, fs, sta_len, lta_len, threshold):
#     x    = wave[2].astype(np.float64) if wave.ndim == 2 else wave.astype(np.float64)
#     nsta = int(sta_len * fs)
#     nlta = int(lta_len * fs)
#     T    = len(x)
#     if T < nlta + nsta:
#         return -1
#     cf    = x ** 2
#     cs    = np.cumsum(np.concatenate([[0.0], cf]))
#     i0, i1 = nlta, T - nsta
#     if i0 >= i1:
#         return -1
#     idx   = np.arange(i0, i1)
#     lta   = (cs[idx] - cs[idx - nlta]) / nlta
#     sta   = (cs[idx + nsta] - cs[idx]) / nsta
#     valid = lta > EPS
#     ratio = np.zeros(len(idx))
#     ratio[valid] = sta[valid] / lta[valid]
#     trig  = np.where(ratio > threshold)[0]
#     return int(trig[0] + i0) if len(trig) > 0 else -1
#
# # ============================================================
# # 神经网络 P波拾取封装
# # ============================================================
# class NNPicker:
#     def __init__(self, ckpt_path: str, device, threshold: float = 0.5):
#         from UNetAS import UNet_mpt
#         self.model = UNet_mpt().to(device)
#         ckpt  = torch.load(ckpt_path, map_location=device)
#         state = ckpt.get("model_state_dict", ckpt)
#         self.model.load_state_dict(state)
#         self.model.eval()
#         self.device    = device
#         self.threshold = threshold
#
#     @torch.no_grad()
#     def pick(self, wave_np: np.ndarray) -> int:
#         x = torch.from_numpy(
#             wave_np.astype(np.float32)
#         ).unsqueeze(0).to(self.device)
#         out  = self.model(x)
#         prob = torch.sigmoid(out).squeeze().cpu().numpy()
#         peak = int(np.argmax(prob))
#         return peak if prob[peak] >= self.threshold else -1
#
# # ============================================================
# # 冻结工具（带调试输出）
# # ============================================================
# def apply_freeze_strategy(model, strategy):
#     # 第一步：全部冻结
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # 第二步：按关键词解冻
#     matched_modules = []
#     for module_key in strategy["unfreeze"]:
#         if module_key.lower() == "all":
#             for param in model.parameters():
#                 param.requires_grad = True
#             matched_modules.append("all")
#             break
#         for name, module in model.named_modules():
#             if module_key.lower() in name.lower():
#                 matched_modules.append(name)
#                 for param in module.parameters():
#                     param.requires_grad = True
#
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total     = sum(p.numel() for p in model.parameters())
#     print(f"    匹配的模块: {set([m.split('.')[0] + '.*' if '.' in m else m for m in matched_modules[:10]])}")
#     print(f"    可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
#     return model
#
# # ============================================================
# # 训练一个 epoch
# # ============================================================
# def train_one_epoch(model, loader, optimizer, device):
#     model.train()
#     total_loss = 0.0
#     n = 0
#     for batch in loader:
#         noisy  = batch["noisy"].to(device)
#         clean  = batch["clean"].to(device)
#         z_cond = batch["z_cond"].to(device)
#
#         if not all(torch.isfinite(t).all() for t in [noisy, clean, z_cond]):
#             continue
#
#         optimizer.zero_grad()
#         try:
#             pred, quality, _ = model(noisy, z_cond)
#         except Exception:
#             continue
#
#         loss = nn.functional.l1_loss(pred, clean)
#         if not torch.isfinite(loss):
#             continue
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#
#         total_loss += loss.item()
#         n += 1
#
#     return total_loss / n if n > 0 else float("nan")
#
# # ============================================================
# # 验证（完整7指标版）
# # ============================================================
# @torch.no_grad()
# def validate(model, loader, device, nn_picker=None):
#     model.eval()
#     delta_snrs, ccs, rmses, prds, st_maes = [], [], [], [], []
#     qualities, pick_suc = [], []
#     kw = dict(fs=CONFIG["fs"], sta_len=CONFIG["sta_len"],
#               lta_len=CONFIG["lta_len"], threshold=CONFIG["stalta_thr"])
#
#     for batch in loader:
#         noisy  = batch["noisy"].to(device)
#         clean  = batch["clean"].to(device)
#         z_cond = batch["z_cond"].to(device)
#         p_true = batch["p_onset"]
#
#         if not all(torch.isfinite(t).all() for t in [noisy, clean, z_cond]):
#             continue
#         try:
#             pred, quality, _ = model(noisy, z_cond)
#         except Exception:
#             continue
#         if not torch.isfinite(pred).all():
#             continue
#
#         for i in range(noisy.shape[0]):
#             x_np = noisy[i].cpu().numpy()
#             y_np = clean[i].cpu().numpy()
#             p_np = pred[i].cpu().numpy()
#             pt   = int(p_true[i].item())
#             qual = float(quality[i].item())
#
#             y_z = y_np[2].astype(np.float64)
#             p_z = p_np[2].astype(np.float64)
#             x_z = x_np[2].astype(np.float64)
#
#             snr_i = snr_db_fn(y_z, x_z)
#             snr_o = snr_db_fn(y_z, p_z)
#
#             delta_snrs.append(snr_o - snr_i)
#             ccs.append(cc_fn(y_z, p_z))
#             rmses.append(rmse_fn(y_z, p_z))
#             prds.append(prd_fn(y_z, p_z))
#             st_maes.append(st_mae_mean(y_z, p_z, CONFIG["fs"]))
#             qualities.append(qual)
#
#             if nn_picker is not None:
#                 pick = nn_picker.pick(p_np)
#             else:
#                 pick = stalta_pick(p_np, **kw)
#             pick_suc.append(pick >= 0 and abs(pick - pt) <= CONFIG["pick_tol"])
#
#     def safe_mean(lst):
#         vals = [v for v in lst if not np.isnan(float(v))]
#         return float(np.mean(vals)) if vals else float("nan")
#
#     return {
#         "delta_snr":         safe_mean(delta_snrs),
#         "cc":                safe_mean(ccs),
#         "rmse":              safe_mean(rmses),
#         "prd":               safe_mean(prds),
#         "st_mae":            safe_mean(st_maes),
#         "quality":           safe_mean(qualities),
#         "pick_success_rate": float(np.mean(pick_suc)) if pick_suc else 0.0,
#     }
#
# # ============================================================
# # 主函数
# # ============================================================
# def main():
#     os.makedirs(CONFIG["output_dir"], exist_ok=True)
#     torch.manual_seed(CONFIG["seed"])
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")
#
#     nn_picker = None
#     if os.path.exists(CONFIG.get("pick_ckpt", "")):
#         try:
#             nn_picker = NNPicker(
#                 ckpt_path = CONFIG["pick_ckpt"],
#                 device    = device,
#                 threshold = CONFIG["pick_threshold"],
#             )
#             print(f"  P波拾取模型已加载: {CONFIG['pick_ckpt']}")
#         except Exception as e:
#             print(f"  [警告] 拾取模型加载失败，回退STA/LTA: {e}")
#             nn_picker = None
#     else:
#         print("  [提示] 未配置pick_ckpt或文件不存在，使用STA/LTA拾取")
#
#     full_ds = FinetuneDataset(
#         event_h5_path  = CONFIG["finetune_h5"],
#         event_csv_path = CONFIG["finetune_csv"],
#         noise_h5_path  = CONFIG["noise_h5"],
#         noise_csv_path = CONFIG["noise_csv"],
#         signal_len     = CONFIG["signal_len"],
#         cond_len       = CONFIG["cond_len"],
#         snr_db_range   = CONFIG["snr_db_range"],
#         noise_boost    = CONFIG["noise_boost"],
#         max_samples    = CONFIG["max_samples"],
#         seed           = CONFIG["seed"],
#     )
#     n_val   = max(1, int(len(full_ds) * CONFIG["val_ratio"]))
#     n_train = len(full_ds) - n_val
#     train_ds, val_ds = random_split(
#         full_ds, [n_train, n_val],
#         generator=torch.Generator().manual_seed(CONFIG["seed"])
#     )
#     train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
#                               shuffle=True,  num_workers=CONFIG["num_workers"])
#     val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
#                               shuffle=False, num_workers=CONFIG["num_workers"])
#     print(f"训练集: {len(train_ds)}  验证集: {len(val_ds)}")
#
#     all_results = []
#     for strategy in FREEZE_STRATEGIES:
#         print(f"\n{'=' * 60}")
#         print(f"策略: {strategy['display']}")
#
#         model = NoiseAwareDenoiserV3(
#             z_dim=CONFIG["z_dim"],
#             cond_len=CONFIG["cond_len"],
#             num_heads=CONFIG["num_heads"],
#         ).to(device)
#         ckpt = torch.load(CONFIG["pretrain_ckpt"], map_location=device)
#         state = ckpt.get("model_state_dict", ckpt)
#         model.load_state_dict(state)
#
#         model = apply_freeze_strategy(model, strategy)
#
#         trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
#
#         if len(trainable_params) == 0:
#             print("    [提示] 无可训练参数，直接评估预训练权重")
#             val_metrics = validate(model, val_loader, device, nn_picker=nn_picker)
#             best_metrics = val_metrics.copy()
#             history = [{"epoch": 0, "train_loss": float("nan"), **val_metrics}]
#             torch.save(model.state_dict(),
#                        os.path.join(CONFIG["output_dir"],
#                                     f"best_{strategy['name']}.pth"))
#         else:
#             optimizer = torch.optim.Adam(trainable_params, lr=CONFIG["lr"])
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#                 optimizer, T_max=CONFIG["epochs"]
#             )
#
#             history = []
#             best_snr = -999
#             best_metrics = {}
#
#             for epoch in range(1, CONFIG["epochs"] + 1):
#                 train_loss = train_one_epoch(model, train_loader, optimizer, device)
#                 val_metrics = validate(model, val_loader, device, nn_picker=nn_picker)
#                 scheduler.step()
#
#                 history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
#
#                 if val_metrics["delta_snr"] > best_snr:
#                     best_snr = val_metrics["delta_snr"]
#                     best_metrics = val_metrics.copy()
#                     torch.save(model.state_dict(),
#                                os.path.join(CONFIG["output_dir"],
#                                             f"best_{strategy['name']}.pth"))
#
#                 if epoch % 5 == 0 or epoch == 1:
#                     print(f"  Epoch {epoch:>3} | loss={train_loss:.4f} | "
#                           f"ΔSNR={val_metrics['delta_snr']:+.3f} | "
#                           f"CC={val_metrics['cc']:.4f} | "
#                           f"RMSE={val_metrics['rmse']:.4f} | "
#                           f"PRD={val_metrics['prd']:.4f} | "
#                           f"ST-MAE={val_metrics['st_mae']:.4f} | "
#                           f"Quality={val_metrics['quality']:.4f} | "
#                           f"Pick={val_metrics['pick_success_rate']:.3f}")
#
#         pd.DataFrame(history).to_csv(
#             os.path.join(CONFIG["output_dir"], f"history_{strategy['name']}.csv"),
#             index=False
#         )
#         all_results.append({
#             "strategy": strategy["name"],
#             "display": strategy["display"],
#             "delta_snr": best_metrics.get("delta_snr", float("nan")),
#             "cc": best_metrics.get("cc", float("nan")),
#             "rmse": best_metrics.get("rmse", float("nan")),
#             "prd": best_metrics.get("prd", float("nan")),
#             "st_mae": best_metrics.get("st_mae", float("nan")),
#             "quality": best_metrics.get("quality", float("nan")),
#             "pick_success_rate": best_metrics.get("pick_success_rate", 0.0),
#         })
#
#     df_result = pd.DataFrame(all_results)
#     df_result.to_csv(os.path.join(CONFIG["output_dir"], "exp2_summary.csv"),
#                      index=False, float_format="%.4f")
#
#     print(f"\n{'='*110}")
#     print("实验2 冻结策略消融对比")
#     print(f"{'='*110}")
#     print(f"  {'策略':<30} {'ΔSNR':>8} {'CC':>8} {'RMSE':>8} "
#           f"{'PRD':>8} {'ST-MAE':>8} {'Quality':>8} {'Pick':>8}")
#     print(f"  {'-'*105}")
#     for r in all_results:
#         print(f"  {r['display']:<30} "
#               f"{r['delta_snr']:>+8.4f} "
#               f"{r['cc']:>8.4f} "
#               f"{r['rmse']:>8.4f} "
#               f"{r['prd']:>8.4f} "
#               f"{r['st_mae']:>8.4f} "
#               f"{r['quality']:>8.4f} "
#               f"{r['pick_success_rate']:>8.4f}")
#
#     metrics_plot = [
#         ("delta_snr",         "ΔSNR (dB)",    True),
#         ("cc",                "CC",            True),
#         ("rmse",              "RMSE",          False),
#         ("prd",               "PRD",           False),
#         ("st_mae",            "ST-MAE",        False),
#         ("quality",           "Quality Score", True),
#         ("pick_success_rate", "P波拾取成功率", True),
#     ]
#     colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
#
#     fig, axes = plt.subplots(1, len(metrics_plot), figsize=(28, 5))
#     fig.suptitle("实验2：冻结策略消融对比", fontsize=13, fontweight="bold")
#
#     for ax, (key, label, higher) in zip(axes, metrics_plot):
#         vals = [r.get(key, float("nan")) for r in all_results]
#         bars = ax.bar(range(len(vals)), vals,
#                       color=colors[:len(vals)], alpha=0.85, width=0.5)
#         ax.set_xticks(range(len(vals)))
#         ax.set_xticklabels(
#             [s["name"].replace("_", "\n") for s in FREEZE_STRATEGIES],
#             fontsize=7
#         )
#         ax.set_title(label, fontsize=9)
#         ax.set_ylabel(label, fontsize=8)
#         ax.grid(axis="y", alpha=0.3)
#         for bar, v in zip(bars, vals):
#             if not np.isnan(v):
#                 ax.text(bar.get_x() + bar.get_width() / 2,
#                         v + abs(v) * 0.01,
#                         f"{v:.3f}", ha="center", fontsize=7)
#         note = "↑ better" if higher else "↓ better"
#         ax.text(0.98, 0.98, note, transform=ax.transAxes,
#                 fontsize=7, ha="right", va="top", color="gray")
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(CONFIG["output_dir"], "exp2_compare.png"),
#                 dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     print(f"\n[✅] 结果目录: {CONFIG['output_dir']}")
#
# if __name__ == "__main__":
#     main()






# -*- coding: utf-8 -*-
"""
实验2：冻结策略消融实验 + 可视化
比较4种冻结策略在 non_naturaldata 上的迁移效果
"""

import os, sys, h5py, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import platform

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [THIS_DIR, os.path.abspath(os.path.join(THIS_DIR, ".."))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_v3 import NoiseAwareDenoiserV3

# ============================================================
# 中文字体配置
# ============================================================
def _set_font():
    system = platform.system()
    if system == "Windows":
        candidates = ["Microsoft YaHei", "SimHei", "SimSun"]
    elif system == "Darwin":
        candidates = ["PingFang SC", "Heiti TC", "STHeiti"]
    else:
        candidates = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字体] 使用: {font}")
            return
    print("[字体] 未找到中文字体，使用默认字体")

_set_font()

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "pretrain_ckpt":  r"D:/X/denoise/part1/v3/checkpoints_v3/best_model_v3.pth",
    "finetune_h5":    r"D:/X/p_wave/data/non_naturaldata.hdf5",
    "finetune_csv":   r"D:/X/p_wave/data/non_naturaldata.csv",
    "noise_h5":       r"D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv":      r"D:/X/p_wave/data/chunk1.csv",
    "output_dir":     r"D:/X/denoise/part1/v3/exp2_freeze_fixed",
    "pick_ckpt":      r"D:/X/p_wave/output/Stanford/12.7rightdams/model_unet_mpt.pt",
    "pick_threshold": 0.5,
    "max_samples":    5000,
    "val_ratio":      0.15,
    "signal_len":     6000,
    "cond_len":       400,
    "z_dim":          128,
    "num_heads":      8,
    "snr_db_range":   (-15.0, 10.0),
    "noise_boost":    1.0,
    "batch_size":     16,
    "num_workers":    0,
    "epochs":         50,
    "lr":             1e-4,
    "seed":           42,
    "sta_len":        0.5,
    "lta_len":        4.0,
    "stalta_thr":     2.5,
    "pick_tol":       50,
    "fs":             100,
    "num_plots":      50,
}

EPS = 1e-10

# ============================================================
# 冻结策略定义（修正版）
# ============================================================
FREEZE_STRATEGIES = [
    {
        "name":    "full_freeze",
        "display": "全冻结（只训练输出层）",
        "unfreeze": ["quality_head", "out_conv"],
    },
    {
        "name":    "freeze_encoder",
        "display": "冻结Encoder（训练FiLM+Decoder）",
        "unfreeze": ["film", "dec", "quality_head", "out_conv"],
    },
    {
        "name":    "freeze_noise_encoder",
        "display": "冻结NoiseEncoder（训练主干）",
        "unfreeze": ["denoiser"],
    },
    {
        "name":    "full_unfreeze",
        "display": "全解冻（整体微调）",
        "unfreeze": ["all"],
    },
]

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
        seg  = x[:, start:end]
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

        snr_db_val  = float(rng.uniform(self.snr_db_range[0], self.snr_db_range[1]))
        noisy_base  = self._mix_snr(clean, noise, snr_db_val)
        noisy       = clean + self.noise_boost * (noisy_base - clean)

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
# 指标函数
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

def prd_fn(clean, test):
    return float(np.sqrt(
        np.sum((clean - test) ** 2) / (np.sum(clean ** 2) + EPS)
    ))

def st_mae_mean(clean, test, fs=100, win_ms=100, overlap=0.5):
    n   = min(len(clean), len(test))
    win = min(int(fs * win_ms / 1000), n)
    if win <= 0:
        return float('nan')
    hop  = max(1, int(win * (1.0 - overlap)))
    vals = [np.abs(clean[s:s + win] - test[s:s + win]).mean()
            for s in range(0, n - win + 1, hop)]
    return float(np.mean(vals)) if vals else float('nan')

def compute_st_mae(clean, test, fs=100, win_ms=100.0, overlap=0.5):
    win_len = int(fs * win_ms / 1000)
    hop_len = max(1, int(win_len * (1 - overlap)))
    T = min(len(clean), len(test))
    times, st_mae = [], []
    start = 0
    while start + win_len <= T:
        end = start + win_len
        st_mae.append(np.abs(clean[start:end] - test[start:end]).mean())
        times.append((start + end) / 2 / fs)
        start += hop_len
    return np.array(times), np.array(st_mae)

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
# 神经网络 P波拾取封装
# ============================================================
class NNPicker:
    def __init__(self, ckpt_path: str, device, threshold: float = 0.5):
        from UNetAS import UNet_mpt
        self.model = UNet_mpt().to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()
        self.device    = device
        self.threshold = threshold

    @torch.no_grad()
    def pick(self, wave_np: np.ndarray) -> int:
        x = torch.from_numpy(
            wave_np.astype(np.float32)
        ).unsqueeze(0).to(self.device)
        out  = self.model(x)
        prob = torch.sigmoid(out).squeeze().cpu().numpy()
        peak = int(np.argmax(prob))
        return peak if prob[peak] >= self.threshold else -1

# ============================================================
# 冻结工具（带调试输出）
# ============================================================
def apply_freeze_strategy(model, strategy):
    for param in model.parameters():
        param.requires_grad = False

    matched_modules = []
    for module_key in strategy["unfreeze"]:
        if module_key.lower() == "all":
            for param in model.parameters():
                param.requires_grad = True
            matched_modules.append("all")
            break
        for name, module in model.named_modules():
            if module_key.lower() in name.lower():
                matched_modules.append(name)
                for param in module.parameters():
                    param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"    匹配的模块: {set([m.split('.')[0] + '.*' if '.' in m else m for m in matched_modules[:10]])}")
    print(f"    可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model

# ============================================================
# 训练一个 epoch
# ============================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        noisy  = batch["noisy"].to(device)
        clean  = batch["clean"].to(device)
        z_cond = batch["z_cond"].to(device)

        if not all(torch.isfinite(t).all() for t in [noisy, clean, z_cond]):
            continue

        optimizer.zero_grad()
        try:
            pred, quality, _ = model(noisy, z_cond)
        except Exception:
            continue

        loss = nn.functional.l1_loss(pred, clean)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n if n > 0 else float("nan")

# ============================================================
# 验证（完整7指标版）
# ============================================================
@torch.no_grad()
def validate(model, loader, device, nn_picker=None):
    model.eval()
    delta_snrs, ccs, rmses, prds, st_maes = [], [], [], [], []
    qualities, pick_suc = [], []
    kw = dict(fs=CONFIG["fs"], sta_len=CONFIG["sta_len"],
              lta_len=CONFIG["lta_len"], threshold=CONFIG["stalta_thr"])

    for batch in loader:
        noisy  = batch["noisy"].to(device)
        clean  = batch["clean"].to(device)
        z_cond = batch["z_cond"].to(device)
        p_true = batch["p_onset"]

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
            qual = float(quality[i].item())

            y_z = y_np[2].astype(np.float64)
            p_z = p_np[2].astype(np.float64)
            x_z = x_np[2].astype(np.float64)

            snr_i = snr_db_fn(y_z, x_z)
            snr_o = snr_db_fn(y_z, p_z)

            delta_snrs.append(snr_o - snr_i)
            ccs.append(cc_fn(y_z, p_z))
            rmses.append(rmse_fn(y_z, p_z))
            prds.append(prd_fn(y_z, p_z))
            st_maes.append(st_mae_mean(y_z, p_z, CONFIG["fs"]))
            qualities.append(qual)

            if nn_picker is not None:
                pick = nn_picker.pick(p_np)
            else:
                pick = stalta_pick(p_np, **kw)
            pick_suc.append(pick >= 0 and abs(pick - pt) <= CONFIG["pick_tol"])

    def safe_mean(lst):
        vals = [v for v in lst if not np.isnan(float(v))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "delta_snr":         safe_mean(delta_snrs),
        "cc":                safe_mean(ccs),
        "rmse":              safe_mean(rmses),
        "prd":               safe_mean(prds),
        "st_mae":            safe_mean(st_maes),
        "quality":           safe_mean(qualities),
        "pick_success_rate": float(np.mean(pick_suc)) if pick_suc else 0.0,
    }

# ============================================================
# 可视化函数
# ============================================================
def _style_ax(ax, ylabel="", title="", label_color="#333333",
              xlim=None, ylim=None, show_xlabel=False):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=label_color,
                      labelpad=4, rotation=90, va="center")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#aaaaaa")
        ax.spines[sp].set_linewidth(0.7)

def plot_one_sample(x_noisy, y_clean, y_pred, p_onset, fs, save_path,
                    sample_idx, snr_in, snr_out, strategy_display):
    T = x_noisy.shape[1]
    t_axis = np.arange(T) / fs
    xlim = (0, T / fs)
    p_t = p_onset / fs
    CH_NAMES = ["Z", "N", "E"]
    COLORS = {
        "noisy": "#2ca02c",
        "clean": "#d62728",
        "denoised": "#1a1a1a",
        "error": "#1f77b4",
        "stmae_d": "#ff7f0e",
        "stmae_n": "#2ca02c",
    }

    ROW_LABELS = [
        "Noisy signal",
        "Clean signal",
        "Denoised signal",
        "Error (Denoised - Clean)",
        "ST-MAE",
    ]

    ROW_LABEL_COLORS = [COLORS["noisy"], COLORS["clean"],
                        COLORS["denoised"], COLORS["error"], COLORS["stmae_d"]]

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        f"Qualitative Denoising Performance — Sample #{sample_idx}"
        f"  [{strategy_display}]\n"
        f"Input SNR = {snr_in:.2f} dB   →   "
        f"Output SNR = {snr_out:.2f} dB   "
        f"(Gain = {snr_out - snr_in:+.2f} dB)",
        fontsize=12, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.42, wspace=0.22,
                           top=0.93, bottom=0.05, left=0.08, right=0.98)

    for col, ch in enumerate(CH_NAMES):
        noisy_ch = x_noisy[col].astype(np.float64)
        clean_ch = y_clean[col].astype(np.float64)
        denoised_ch = y_pred[col].astype(np.float64)
        error_ch = denoised_ch - clean_ch

        ax0 = fig.add_subplot(gs[0, col])
        ax0.plot(t_axis, noisy_ch, color=COLORS["noisy"], linewidth=0.5, alpha=0.9)
        ax0.axvline(p_t, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
        _style_ax(ax0, ylabel=ROW_LABELS[0] if col == 0 else "",
                  title=f"Channel {ch}", label_color=ROW_LABEL_COLORS[0],
                  xlim=xlim, ylim=(-1.1, 1.1))

        ax1 = fig.add_subplot(gs[1, col])
        ax1.plot(t_axis, clean_ch, color=COLORS["clean"], linewidth=0.5, alpha=0.9)
        ax1.axvline(p_t, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
        _style_ax(ax1, ylabel=ROW_LABELS[1] if col == 0 else "",
                  label_color=ROW_LABEL_COLORS[1], xlim=xlim, ylim=(-1.1, 1.1))

        ax2 = fig.add_subplot(gs[2, col])
        ax2.plot(t_axis, denoised_ch, color=COLORS["denoised"], linewidth=0.5, alpha=0.9)
        ax2.axvline(p_t, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
        snr_ch = snr_db_fn(clean_ch, denoised_ch)
        rmse_ch = rmse_fn(clean_ch, denoised_ch)
        prd_ch = prd_fn(clean_ch, denoised_ch)
        ax2.set_title(f"SNR={snr_ch:.1f}dB  RMSE={rmse_ch:.4f}  PRD={prd_ch:.3f}",
                      fontsize=7.5, color="#333333", pad=3)
        _style_ax(ax2, ylabel=ROW_LABELS[2] if col == 0 else "",
                  label_color=ROW_LABEL_COLORS[2], xlim=xlim, ylim=(-1.1, 1.1))

        ax3 = fig.add_subplot(gs[3, col])
        ax3.plot(t_axis, error_ch, color=COLORS["error"], linewidth=0.45, alpha=0.85)
        ax3.axhline(0, color="#aaaaaa", linewidth=0.6)
        ax3.axvline(p_t, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
        _style_ax(ax3, ylabel=ROW_LABELS[3] if col == 0 else "",
                  label_color=ROW_LABEL_COLORS[3], xlim=xlim, ylim=(-0.25, 0.25))

        ax4 = fig.add_subplot(gs[4, col])
        t_n, stmae_n = compute_st_mae(clean_ch, noisy_ch, fs)
        t_d, stmae_d = compute_st_mae(clean_ch, denoised_ch, fs)
        ax4.plot(t_n, stmae_n, color=COLORS["stmae_n"], linewidth=0.6,
                 alpha=0.75, label="Noisy")
        ax4.plot(t_d, stmae_d, color=COLORS["stmae_d"], linewidth=0.8,
                 alpha=0.95, label="Denoised")
        ax4.axvline(p_t, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
        ymax = max(stmae_n.max() if len(stmae_n) > 0 else 0.06,
                   stmae_d.max() if len(stmae_d) > 0 else 0.06) * 1.15
        ymax = max(ymax, 0.06)
        if col == 0:
            ax4.legend(fontsize=7, loc="upper right", framealpha=0.5, handlelength=1.5)
        _style_ax(ax4, ylabel=ROW_LABELS[4] if col == 0 else "",
                  label_color=ROW_LABEL_COLORS[4], xlim=xlim, ylim=(0, ymax),
                  show_xlabel=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {save_path}")

# ============================================================
# 主函数
# ============================================================
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    nn_picker = None
    if os.path.exists(CONFIG.get("pick_ckpt", "")):
        try:
            nn_picker = NNPicker(
                ckpt_path = CONFIG["pick_ckpt"],
                device    = device,
                threshold = CONFIG["pick_threshold"],
            )
            print(f"  P波拾取模型已加载: {CONFIG['pick_ckpt']}")
        except Exception as e:
            print(f"  [警告] 拾取模型加载失败，回退STA/LTA: {e}")
            nn_picker = None
    else:
        print("  [提示] 未配置pick_ckpt或文件不存在，使用STA/LTA拾取")

    full_ds = FinetuneDataset(
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
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(CONFIG["seed"])
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                                  shuffle=True,  num_workers=CONFIG["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                                  shuffle=False, num_workers=CONFIG["num_workers"])
    print(f"训练集: {len(train_ds)}  验证集: {len(val_ds)}")

    # 预先缓存固定波形用于可视化
    num_plots = CONFIG["num_plots"]
    fixed_samples = []
    for idx in range(min(num_plots, len(val_ds))):
        batch = val_ds[idx]
        fixed_samples.append({
            "noisy":   batch["noisy"].unsqueeze(0),
            "clean":   batch["clean"].unsqueeze(0),
            "z_cond":  batch["z_cond"].unsqueeze(0),
            "p_onset": int(batch["p_onset"]),
        })
    print(f"已缓存 {len(fixed_samples)} 条固定波形用于可视化")

    all_results = []
    for strategy in FREEZE_STRATEGIES:
        print(f"\n{'=' * 60}")
        print(f"策略: {strategy['display']}")

        model = NoiseAwareDenoiserV3(
            z_dim=CONFIG["z_dim"],
            cond_len=CONFIG["cond_len"],
            num_heads=CONFIG["num_heads"],
        ).to(device)
        ckpt = torch.load(CONFIG["pretrain_ckpt"], map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)

        model = apply_freeze_strategy(model, strategy)

        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

        if len(trainable_params) == 0:
            print("    [提示] 无可训练参数，直接评估预训练权重")
            val_metrics = validate(model, val_loader, device, nn_picker=nn_picker)
            best_metrics = val_metrics.copy()
            history = [{"epoch": 0, "train_loss": float("nan"), **val_metrics}]
            torch.save(model.state_dict(),
                       os.path.join(CONFIG["output_dir"],
                                    f"best_{strategy['name']}.pth"))
        else:
            optimizer = torch.optim.Adam(trainable_params, lr=CONFIG["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CONFIG["epochs"]
            )

            history = []
            best_snr = -999
            best_metrics = {}

            for epoch in range(1, CONFIG["epochs"] + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, device)
                val_metrics = validate(model, val_loader, device, nn_picker=nn_picker)
                scheduler.step()

                history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

                if val_metrics["delta_snr"] > best_snr:
                    best_snr = val_metrics["delta_snr"]
                    best_metrics = val_metrics.copy()
                    torch.save(model.state_dict(),
                               os.path.join(CONFIG["output_dir"],
                                            f"best_{strategy['name']}.pth"))

                if epoch % 5 == 0 or epoch == 1:
                    print(f"  Epoch {epoch:>3} | loss={train_loss:.4f} | "
                          f"ΔSNR={val_metrics['delta_snr']:+.3f} | "
                          f"CC={val_metrics['cc']:.4f} | "
                          f"RMSE={val_metrics['rmse']:.4f} | "
                          f"PRD={val_metrics['prd']:.4f} | "
                          f"ST-MAE={val_metrics['st_mae']:.4f} | "
                          f"Quality={val_metrics['quality']:.4f} | "
                          f"Pick={val_metrics['pick_success_rate']:.3f}")

        pd.DataFrame(history).to_csv(
            os.path.join(CONFIG["output_dir"], f"history_{strategy['name']}.csv"),
            index=False
        )
        all_results.append({
            "strategy": strategy["name"],
            "display": strategy["display"],
            "delta_snr": best_metrics.get("delta_snr", float("nan")),
            "cc": best_metrics.get("cc", float("nan")),
            "rmse": best_metrics.get("rmse", float("nan")),
            "prd": best_metrics.get("prd", float("nan")),
            "st_mae": best_metrics.get("st_mae", float("nan")),
            "quality": best_metrics.get("quality", float("nan")),
            "pick_success_rate": best_metrics.get("pick_success_rate", 0.0),
        })

        # ============================================================
        # 可视化：为当前策略生成波形图
        # ============================================================
        print(f"\n  [可视化] 生成 {strategy['display']} 的波形图...")
        ckpt_path = os.path.join(CONFIG["output_dir"], f"best_{strategy['name']}.pth")

        if os.path.exists(ckpt_path):
            model_vis = NoiseAwareDenoiserV3(
                z_dim=CONFIG["z_dim"],
                cond_len=CONFIG["cond_len"],
                num_heads=CONFIG["num_heads"],
            ).to(device)
            state_vis = torch.load(ckpt_path, map_location=device)
            if isinstance(state_vis, dict) and "model_state_dict" in state_vis:
                state_vis = state_vis["model_state_dict"]
            model_vis.load_state_dict(state_vis)
            model_vis.eval()

            out_dir = os.path.join(CONFIG["output_dir"], "val_plots", strategy["name"])
            os.makedirs(out_dir, exist_ok=True)

            snr_in_list, snr_out_list = [], []

            with torch.no_grad():
                for i, sample in enumerate(fixed_samples):
                    noisy   = sample["noisy"].to(device)
                    clean   = sample["clean"].to(device)
                    z_cond  = sample["z_cond"].to(device)
                    p_onset = sample["p_onset"]

                    if not all(torch.isfinite(t).all() for t in [noisy, clean, z_cond]):
                        continue

                    try:
                        pred, _, _ = model_vis(noisy, z_cond)
                    except Exception:
                        continue

                    if not torch.isfinite(pred).all():
                        continue

                    x_np    = noisy[0].cpu().numpy()
                    y_np    = clean[0].cpu().numpy()
                    pred_np = pred[0].cpu().numpy()

                    snr_in  = snr_db_fn(y_np[2], x_np[2])
                    snr_out = snr_db_fn(y_np[2], pred_np[2])
                    snr_in_list.append(snr_in)
                    snr_out_list.append(snr_out)

                    save_path = os.path.join(
                        out_dir,
                        f"sample_{i + 1:03d}_snrin{snr_in:.1f}dB.png",
                    )
                    plot_one_sample(
                        x_noisy          = x_np,
                        y_clean          = y_np,
                        y_pred           = pred_np,
                        p_onset          = p_onset,
                        fs               = CONFIG["fs"],
                        save_path        = save_path,
                        sample_idx       = i + 1,
                        snr_in           = snr_in,
                        snr_out          = snr_out,
                        strategy_display = strategy["display"],
                    )

            if snr_in_list:
                gains = np.array(snr_out_list) - np.array(snr_in_list)
                print(f"    生成图数: {len(snr_in_list)}")
                print(f"    Input SNR:  {np.mean(snr_in_list):.2f} dB")
                print(f"    Output SNR: {np.mean(snr_out_list):.2f} dB")
                print(f"    SNR Gain:   {gains.mean():+.2f} dB")

    # ============================================================
    # 汇总结果
    # ============================================================
    df_result = pd.DataFrame(all_results)
    df_result.to_csv(os.path.join(CONFIG["output_dir"], "exp2_summary.csv"),
                     index=False, float_format="%.4f")

    print(f"\n{'='*110}")
    print("实验2 冻结策略消融对比")
    print(f"{'='*110}")
    print(f"  {'策略':<30} {'ΔSNR':>8} {'CC':>8} {'RMSE':>8} "
          f"{'PRD':>8} {'ST-MAE':>8} {'Quality':>8} {'Pick':>8}")
    print(f"  {'-'*105}")
    for r in all_results:
        print(f"  {r['display']:<30} "
              f"{r['delta_snr']:>+8.4f} "
              f"{r['cc']:>8.4f} "
              f"{r['rmse']:>8.4f} "
              f"{r['prd']:>8.4f} "
              f"{r['st_mae']:>8.4f} "
              f"{r['quality']:>8.4f} "
              f"{r['pick_success_rate']:>8.4f}")

    metrics_plot = [
        ("delta_snr",         "ΔSNR (dB)",    True),
        ("cc",                "CC",            True),
        ("rmse",              "RMSE",          False),
        ("prd",               "PRD",           False),
        ("st_mae",            "ST-MAE",        False),
        ("quality",           "Quality Score", True),
        ("pick_success_rate", "P波拾取成功率", True),
    ]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    fig, axes = plt.subplots(1, len(metrics_plot), figsize=(28, 5))
    fig.suptitle("实验2：冻结策略消融对比", fontsize=13, fontweight="bold")

    for ax, (key, label, higher) in zip(axes, metrics_plot):
        vals = [r.get(key, float("nan")) for r in all_results]
        bars = ax.bar(range(len(vals)), vals,
                      color=colors[:len(vals)], alpha=0.85, width=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(
            [s["name"].replace("_", "\n") for s in FREEZE_STRATEGIES],
            fontsize=7
        )
        ax.set_title(label, fontsize=9)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + abs(v) * 0.01,
                        f"{v:.3f}", ha="center", fontsize=7)
        note = "↑ better" if higher else "↓ better"
        ax.text(0.98, 0.98, note, transform=ax.transAxes,
                fontsize=7, ha="right", va="top", color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "exp2_compare.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[✅] 结果目录: {CONFIG['output_dir']}")
    print(f"[✅] 可视化目录: {os.path.join(CONFIG['output_dir'], 'val_plots')}")

if __name__ == "__main__":
    main()