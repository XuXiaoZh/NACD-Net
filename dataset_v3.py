# # v3/dataset_v3.py
# """
# 数据集 V3：两部分数据
#   Part A：干净地震波 + 叠加噪声 → 有监督去噪对
#   Part B：原始波形（可能已含噪）→ 无监督/自监督去噪
# """
#
# import h5py
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
#
# class STEADDatasetV3(Dataset):
#     """
#     Part A (has_target=True):
#         x_noisy  = clean_event + scaled_noise   [3, T]
#         y_clean  = clean_event                  [3, T]
#         z_cond   = pure noise segment           [3, C]  → 送入噪声编码器
#
#     Part B (has_target=False):
#         x_noisy  = raw waveform (may contain noise)  [3, T]
#         y_clean  = x_noisy (自监督，目标=自身)
#         z_cond   = 波形前段（P波前）作为噪声条件     [3, C]
#     """
#
#     def __init__(
#         self,
#         # Part A：事件 + 噪声
#         event_h5_path:  str,
#         event_csv_path: str,
#         noise_h5_path:  str,
#         noise_csv_path: str,
#         # Part B：原始波形（可选）
#         raw_h5_path:    str  = None,
#         raw_csv_path:   str  = None,
#         # 参数
#         signal_len:     int   = 6000,
#         cond_len:       int   = 400,
#         snr_range:      tuple = (0.1, 20.0),   # 功率比
#         clean_prob:     float = 0.10,
#         part_b_ratio:   float = 0.3,           # Part B 占比
#         normalize:      bool  = True,
#         seed:           int   = 42,
#         debug:          bool  = False,
#     ):
#         super().__init__()
#         self.event_h5_path = event_h5_path
#         self.noise_h5_path = noise_h5_path
#         self.raw_h5_path   = raw_h5_path
#         self.signal_len    = signal_len
#         self.cond_len      = cond_len
#         self.snr_range     = snr_range
#         self.clean_prob    = clean_prob
#         self.normalize     = normalize
#         self.debug         = debug
#         self.rng           = np.random.default_rng(seed)
#
#         # ── 读取 CSV ──────────────────────────────────────
#         self.event_df = pd.read_csv(event_csv_path, low_memory=False)
#         self.noise_df = pd.read_csv(noise_csv_path, low_memory=False)
#
#         self.has_part_b = (raw_h5_path is not None and
#                            raw_csv_path is not None)
#         if self.has_part_b:
#             self.raw_df = pd.read_csv(raw_csv_path, low_memory=False)
#             n_b = int(len(self.event_df) * part_b_ratio)
#             self.n_part_b = min(n_b, len(self.raw_df))
#         else:
#             self.raw_df   = None
#             self.n_part_b = 0
#
#         self.n_part_a = len(self.event_df)
#         self.total    = self.n_part_a + self.n_part_b
#
#         # lazy h5
#         self._event_h5 = None
#         self._noise_h5 = None
#         self._raw_h5   = None
#
#         print(f"[Dataset V3] Part A (supervised)   : {self.n_part_a}")
#         print(f"[Dataset V3] Part B (unsupervised) : {self.n_part_b}")
#         print(f"[Dataset V3] Total                 : {self.total}")
#
#     # ── lazy open ─────────────────────────────────────────
#     @property
#     def event_h5(self):
#         if self._event_h5 is None:
#             self._event_h5 = h5py.File(self.event_h5_path, 'r')
#         return self._event_h5
#
#     @property
#     def noise_h5(self):
#         if self._noise_h5 is None:
#             self._noise_h5 = h5py.File(self.noise_h5_path, 'r')
#         return self._noise_h5
#
#     @property
#     def raw_h5(self):
#         if self._raw_h5 is None and self.raw_h5_path:
#             self._raw_h5 = h5py.File(self.raw_h5_path, 'r')
#         return self._raw_h5
#
#     # ── 工具 ──────────────────────────────────────────────
#     def _load(self, h5file, name: str) -> np.ndarray:
#         data = h5file['data'][name][:]
#         data = data.T.astype(np.float32)          # [3, T]
#         return np.nan_to_num(data, nan=0., posinf=0., neginf=0.)
#
#     def _pad_or_crop(self, wave: np.ndarray) -> np.ndarray:
#         T = wave.shape[1]
#         if T >= self.signal_len:
#             return wave[:, :self.signal_len]
#         pad = np.zeros((3, self.signal_len), dtype=np.float32)
#         pad[:, :T] = wave
#         return pad
#
#     def _normalize(self, wave: np.ndarray) -> np.ndarray:
#         m = np.abs(wave).max()
#         return wave / m if m > 1e-10 else wave
#
#     def _mix_snr(self, signal, noise, snr_linear):
#         sp = np.mean(signal ** 2)
#         np_ = np.mean(noise  ** 2)
#         if sp < 1e-10 or np_ < 1e-10:
#             return signal.copy()
#         scale = float(np.clip(np.sqrt(sp / (snr_linear * np_)), 0, 10))
#         return signal + scale * noise
#
#     def _get_p_onset(self, row) -> int:
#         for col in ['p_arrival_sample', 'p_onset', 'itp']:
#             if col in row.index and not pd.isna(row[col]):
#                 try:
#                     v = int(row[col])
#                     if 0 <= v < self.signal_len:
#                         return v
#                 except Exception:
#                     pass
#         return self.signal_len // 10
#
#     # ── Part A：有监督 ────────────────────────────────────
#     def _get_part_a(self, idx):
#         row        = self.event_df.iloc[idx]
#         wave_clean = self._pad_or_crop(
#             self._load(self.event_h5, row['trace_name'])
#         )
#
#         # 噪声采样
#         ni         = self.rng.integers(0, len(self.noise_df))
#         noise_row  = self.noise_df.iloc[ni]
#         noise_wave = self._pad_or_crop(
#             self._load(self.noise_h5, noise_row['trace_name'])
#         )
#
#         # 纯噪声段作为条件（z_cond）
#         noise_cond = noise_wave[:, :self.cond_len].copy()
#
#         # SNR 混合
#         if self.rng.random() < self.clean_prob:
#             x_noisy = wave_clean.copy()
#         else:
#             snr = float(self.rng.uniform(*self.snr_range))
#             x_noisy = self._mix_snr(wave_clean, noise_wave, snr)
#
#         y_clean = wave_clean.copy()
#
#         # 统一归一化
#         if self.normalize:
#             scale = np.abs(x_noisy).max()
#             if scale > 1e-10:
#                 x_noisy    = x_noisy    / scale
#                 y_clean    = y_clean    / scale
#                 noise_cond = noise_cond / max(np.abs(noise_cond).max(), 1e-10)
#
#         p_onset = self._get_p_onset(row)
#
#         # valid_mask
#         valid_mask = np.zeros(self.signal_len, dtype=np.float32)
#         valid_mask[p_onset:min(p_onset + 4000, self.signal_len)] = 1.0
#
#         return {
#             "x":          torch.from_numpy(np.clip(x_noisy,    -10, 10)),
#             "y_clean":    torch.from_numpy(np.clip(y_clean,    -10, 10)),
#             "z_cond":     torch.from_numpy(np.clip(noise_cond, -10, 10)),
#             "valid_mask": torch.from_numpy(valid_mask),
#             "p_onset":    torch.tensor(p_onset, dtype=torch.long),
#             "has_target": torch.tensor(1, dtype=torch.float32),
#         }
#
#     # ── Part B：无监督 ────────────────────────────────────
#     def _get_part_b(self, idx):
#         row  = self.raw_df.iloc[idx]
#         wave = self._pad_or_crop(
#             self._load(self.raw_h5, row['trace_name'])
#         )
#
#         p_onset    = self._get_p_onset(row)
#         cond_start = max(0, p_onset - self.cond_len)
#         noise_cond = wave[:, cond_start:p_onset]
#
#         # 填充 cond
#         pad_cond = np.zeros((3, self.cond_len), dtype=np.float32)
#         seg_len  = noise_cond.shape[1]
#         if seg_len > 0:
#             pad_cond[:, self.cond_len - seg_len:] = noise_cond
#
#         if self.normalize:
#             scale = np.abs(wave).max()
#             if scale > 1e-10:
#                 wave     = wave     / scale
#                 pad_cond = pad_cond / max(np.abs(pad_cond).max(), 1e-10)
#
#         valid_mask = np.zeros(self.signal_len, dtype=np.float32)
#         valid_mask[p_onset:min(p_onset + 4000, self.signal_len)] = 1.0
#
#         wave     = np.clip(wave,     -10, 10).astype(np.float32)
#         pad_cond = np.clip(pad_cond, -10, 10).astype(np.float32)
#
#         return {
#             "x":          torch.from_numpy(wave),
#             "y_clean":    torch.from_numpy(wave),      # 自监督：目标=自身
#             "z_cond":     torch.from_numpy(pad_cond),
#             "valid_mask": torch.from_numpy(valid_mask),
#             "p_onset":    torch.tensor(p_onset, dtype=torch.long),
#             "has_target": torch.tensor(0, dtype=torch.float32),  # 无监督标记
#         }
#
#     def __len__(self):
#         return self.total
#
#     def __getitem__(self, idx):
#         try:
#             if idx < self.n_part_a:
#                 return self._get_part_a(idx)
#             else:
#                 return self._get_part_b(idx - self.n_part_a)
#         except Exception as e:
#             if self.debug:
#                 print(f"[Dataset V3] ⚠ idx={idx} 异常: {e}")
#             return self._zero_sample()
#
#     def _zero_sample(self):
#         return {
#             "x":          torch.zeros(3, self.signal_len),
#             "y_clean":    torch.zeros(3, self.signal_len),
#             "z_cond":     torch.zeros(3, self.cond_len),
#             "valid_mask": torch.zeros(self.signal_len),
#             "p_onset":    torch.tensor(0, dtype=torch.long),
#             "has_target": torch.tensor(0, dtype=torch.float32),
#         }









# v3/dataset_v3.py
"""
数据集 V3：两部分数据
  Part A：干净地震波 + 叠加噪声 → 有监督去噪对
  Part B：原始波形（可能已含噪）→ 无监督/自监督去噪

[PATCH v3.1] 修复归一化顺序问题：
  - 修复前：用 x_noisy 的最大值归一化 → clean 和 noisy 差值极小
  - 修复后：先归一化 clean（scale=1.0），再在归一化域内叠加噪声
  - 效果：y_clean 峰值恒为 1.0，diff(noisy, clean) 清晰可见，SNR 有意义
"""
"""先归一化再叠加噪声"""
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class STEADDatasetV3(Dataset):
    """
    Part A (has_target=True):
        x_noisy  = norm(clean_event) + scaled_noise   [3, T]
        y_clean  = norm(clean_event)                  [3, T]  ← 峰值恒为 1.0
        z_cond   = pure noise segment                 [3, C]  → 送入噪声编码器

    Part B (has_target=False):
        x_noisy  = raw waveform (may contain noise)   [3, T]
        y_clean  = x_noisy (自监督，目标=自身)
        z_cond   = 波形前段（P波前）作为噪声条件      [3, C]
    """

    def __init__(
        self,
        # Part A：事件 + 噪声
        event_h5_path:  str,
        event_csv_path: str,
        noise_h5_path:  str,
        noise_csv_path: str,
        # Part B：原始波形（可选）
        raw_h5_path:    str   = None,
        raw_csv_path:   str   = None,
        # 参数
        signal_len:     int   = 6000,
        cond_len:       int   = 400,
        snr_range:      tuple = (0.1, 20.0),   # 功率比
        clean_prob:     float = 0.10,
        part_b_ratio:   float = 0.3,           # Part B 占比
        normalize:      bool  = True,
        seed:           int   = 42,
        debug:          bool  = False,
    ):
        super().__init__()
        self.event_h5_path = event_h5_path
        self.noise_h5_path = noise_h5_path
        self.raw_h5_path   = raw_h5_path
        self.signal_len    = signal_len
        self.cond_len      = cond_len
        self.snr_range     = snr_range
        self.clean_prob    = clean_prob
        self.normalize     = normalize
        self.debug         = debug
        self.rng           = np.random.default_rng(seed)

        # ── 读取 CSV ──────────────────────────────────────
        self.event_df = pd.read_csv(event_csv_path, low_memory=False)
        self.noise_df = pd.read_csv(noise_csv_path, low_memory=False)

        self.has_part_b = (raw_h5_path is not None and
                           raw_csv_path is not None)
        if self.has_part_b:
            self.raw_df   = pd.read_csv(raw_csv_path, low_memory=False)
            n_b           = int(len(self.event_df) * part_b_ratio)
            self.n_part_b = min(n_b, len(self.raw_df))
        else:
            self.raw_df   = None
            self.n_part_b = 0

        self.n_part_a = len(self.event_df)
        self.total    = self.n_part_a + self.n_part_b

        # lazy h5
        self._event_h5 = None
        self._noise_h5 = None
        self._raw_h5   = None

        print(f"[Dataset V3] Part A (supervised)   : {self.n_part_a}")
        print(f"[Dataset V3] Part B (unsupervised) : {self.n_part_b}")
        print(f"[Dataset V3] Total                 : {self.total}")

    # ── lazy open ─────────────────────────────────────────
    @property
    def event_h5(self):
        if self._event_h5 is None:
            self._event_h5 = h5py.File(self.event_h5_path, 'r')
        return self._event_h5

    @property
    def noise_h5(self):
        if self._noise_h5 is None:
            self._noise_h5 = h5py.File(self.noise_h5_path, 'r')
        return self._noise_h5

    @property
    def raw_h5(self):
        if self._raw_h5 is None and self.raw_h5_path:
            self._raw_h5 = h5py.File(self.raw_h5_path, 'r')
        return self._raw_h5

    # ── 工具 ──────────────────────────────────────────────
    def _load(self, h5file, name: str) -> np.ndarray:
        data = h5file['data'][name][:]
        data = data.T.astype(np.float32)          # [3, T]
        return np.nan_to_num(data, nan=0., posinf=0., neginf=0.)

    def _pad_or_crop(self, wave: np.ndarray) -> np.ndarray:
        T = wave.shape[1]
        if T >= self.signal_len:
            return wave[:, :self.signal_len]
        pad = np.zeros((3, self.signal_len), dtype=np.float32)
        pad[:, :T] = wave
        return pad

    def _normalize(self, wave: np.ndarray) -> np.ndarray:
        """将波形归一化到峰值 = 1.0"""
        m = np.abs(wave).max()
        return wave / m if m > 1e-10 else wave

    # ── [PATCH] 修复后的 _mix_snr ─────────────────────────
    def _mix_snr(self, signal: np.ndarray,
                 noise: np.ndarray,
                 snr_linear: float) -> np.ndarray:
        """
        在归一化域内按线性功率 SNR 混合信号和噪声。
        signal 和 noise 均应已归一化（峰值≈1.0）。

        snr_linear = signal_power / noise_power （线性功率比，非dB）
        scale      = sqrt(signal_power / (snr_linear * noise_power))
        mixed      = signal + scale * noise
        """
        sp  = np.mean(signal ** 2)
        np_ = np.mean(noise  ** 2)
        if sp < 1e-10 or np_ < 1e-10:
            return signal.copy()
        scale = float(np.clip(np.sqrt(sp / (snr_linear * np_)), 0, 10))
        return signal + scale * noise

    def _get_p_onset(self, row) -> int:
        for col in ['p_arrival_sample', 'p_onset', 'itp']:
            if col in row.index and not pd.isna(row[col]):
                try:
                    v = int(row[col])
                    if 0 <= v < self.signal_len:
                        return v
                except Exception:
                    pass
        return self.signal_len // 10

    # ── Part A：有监督 ────────────────────────────────────
    def _get_part_a(self, idx):
        row        = self.event_df.iloc[idx]
        wave_clean = self._pad_or_crop(
            self._load(self.event_h5, row['trace_name'])
        )

        # 噪声采样
        ni         = self.rng.integers(0, len(self.noise_df))
        noise_row  = self.noise_df.iloc[ni]
        noise_wave = self._pad_or_crop(
            self._load(self.noise_h5, noise_row['trace_name'])
        )

        # ── [PATCH] 关键修复：先归一化，再叠加噪声 ──────────
        if self.normalize:
            # Step 1: 先把 clean 和 noise 各自归一化到峰值 = 1.0
            wave_clean = self._normalize(wave_clean)   # y_clean 峰值 = 1.0
            noise_wave = self._normalize(noise_wave)   # 噪声也归一化

        # 纯噪声段作为条件（z_cond），取归一化后的 noise_wave 前段
        noise_cond = noise_wave[:, :self.cond_len].copy()

        # Step 2: 在归一化域内叠加噪声
        if self.rng.random() < self.clean_prob:
            # clean case：不加噪声
            x_noisy = wave_clean.copy()
        else:
            snr     = float(self.rng.uniform(*self.snr_range))
            x_noisy = self._mix_snr(wave_clean, noise_wave, snr)
            # 此时：
            #   y_clean 峰值 ≈ 1.0
            #   x_noisy 峰值 ≈ 1.0 ~ 1.5（取决于 SNR）
            #   diff(noisy, clean) 清晰可见！

        y_clean = wave_clean.copy()

        # ── [PATCH] 不再需要第二次归一化，noise_cond 已归一化 ──
        # （移除了原来错误的"用 x_noisy.max() 统一归一化"的代码）

        # noise_cond 单独归一化（已经在 noise_wave 归一化后取段，可直接用）
        nc_scale = np.abs(noise_cond).max()
        if nc_scale > 1e-10:
            noise_cond = noise_cond / nc_scale

        # debug 日志
        if self.debug:
            diff_max = np.max(np.abs(x_noisy - y_clean))
            print(
                f"[PartA idx={idx}] "
                f"clean_max={np.abs(y_clean).max():.4f}  "
                f"noisy_max={np.abs(x_noisy).max():.4f}  "
                f"diff_max={diff_max:.4f}  "
                f"snr={snr if not self.rng.random() < self.clean_prob else 'clean'}"
            )

        p_onset = self._get_p_onset(row)

        # valid_mask
        valid_mask = np.zeros(self.signal_len, dtype=np.float32)
        valid_mask[p_onset:min(p_onset + 4000, self.signal_len)] = 1.0

        return {
            "x":          torch.from_numpy(np.clip(x_noisy,    -10, 10)),
            "y_clean":    torch.from_numpy(np.clip(y_clean,    -10, 10)),
            "z_cond":     torch.from_numpy(np.clip(noise_cond, -10, 10)),
            "valid_mask": torch.from_numpy(valid_mask),
            "p_onset":    torch.tensor(p_onset, dtype=torch.long),
            "has_target": torch.tensor(1, dtype=torch.float32),
        }

    # ── Part B：无监督（无需修改，逻辑正确）────────────────
    def _get_part_b(self, idx):
        row  = self.raw_df.iloc[idx]
        wave = self._pad_or_crop(
            self._load(self.raw_h5, row['trace_name'])
        )

        p_onset    = self._get_p_onset(row)
        cond_start = max(0, p_onset - self.cond_len)
        noise_cond = wave[:, cond_start:p_onset]

        # 填充 cond
        pad_cond = np.zeros((3, self.cond_len), dtype=np.float32)
        seg_len  = noise_cond.shape[1]
        if seg_len > 0:
            pad_cond[:, self.cond_len - seg_len:] = noise_cond

        if self.normalize:
            scale = np.abs(wave).max()
            if scale > 1e-10:
                wave     = wave     / scale
                pad_cond = pad_cond / max(np.abs(pad_cond).max(), 1e-10)

        valid_mask = np.zeros(self.signal_len, dtype=np.float32)
        valid_mask[p_onset:min(p_onset + 4000, self.signal_len)] = 1.0

        wave     = np.clip(wave,     -10, 10).astype(np.float32)
        pad_cond = np.clip(pad_cond, -10, 10).astype(np.float32)

        return {
            "x":          torch.from_numpy(wave),
            "y_clean":    torch.from_numpy(wave),      # 自监督：目标=自身
            "z_cond":     torch.from_numpy(pad_cond),
            "valid_mask": torch.from_numpy(valid_mask),
            "p_onset":    torch.tensor(p_onset, dtype=torch.long),
            "has_target": torch.tensor(0, dtype=torch.float32),
        }

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        try:
            if idx < self.n_part_a:
                return self._get_part_a(idx)
            else:
                return self._get_part_b(idx - self.n_part_a)
        except Exception as e:
            if self.debug:
                print(f"[Dataset V3] ⚠ idx={idx} 异常: {e}")
            return self._zero_sample()

    def _zero_sample(self):
        return {
            "x":          torch.zeros(3, self.signal_len),
            "y_clean":    torch.zeros(3, self.signal_len),
            "z_cond":     torch.zeros(3, self.cond_len),
            "valid_mask": torch.zeros(self.signal_len),
            "p_onset":    torch.tensor(0, dtype=torch.long),
            "has_target": torch.tensor(0, dtype=torch.float32),
        }