# v3/loss_v3.py
"""
损失函数 V3：
  1. recon_loss     : 有监督重建损失（MSE，加权 valid_mask）
  2. freq_loss      : 频域损失
  3. grad_loss      : 梯度平滑损失
  4. quality_loss   : 质量评分监督
  5. identity_loss  : 高SNR样本约束
  6. consistency_loss: Part B 无监督一致性损失
  7. noise_contrast : 噪声特征对比损失（同批次噪声特征应不同）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DenoiserLossV3(nn.Module):

    def __init__(
        self,
        alpha_recon:       float = 1.0,
        alpha_freq:        float = 0.15,
        alpha_grad:        float = 0.15,
        alpha_quality:     float = 0.10,
        alpha_identity:    float = 30.0,
        alpha_consistency: float = 0.20,
        alpha_contrast:    float = 0.05,
        valid_weight:      float = 3.0,
        bg_weight:         float = 0.3,
        identity_snr_thr:  float = 6.0,
    ):
        super().__init__()
        self.alpha_recon       = alpha_recon
        self.alpha_freq        = alpha_freq
        self.alpha_grad        = alpha_grad
        self.alpha_quality     = alpha_quality
        self.alpha_identity    = alpha_identity
        self.alpha_consistency = alpha_consistency
        self.alpha_contrast    = alpha_contrast
        self.valid_weight      = valid_weight
        self.bg_weight         = bg_weight
        self.identity_snr_thr  = identity_snr_thr

    def forward(
        self,
        pred:       torch.Tensor,   # [B, 3, T]
        target:     torch.Tensor,   # [B, 3, T]
        quality:    torch.Tensor,   # [B, 1]
        x_input:    torch.Tensor,   # [B, 3, T]
        z_noise:    torch.Tensor,   # [B, z_dim]
        valid_mask: torch.Tensor,   # [B, T]
        has_target: torch.Tensor,   # [B]  1=有监督, 0=无监督
    ) -> tuple:

        detail = {}
        loss   = torch.tensor(0.0, device=pred.device)

        sup_mask   = has_target.bool()                 # [B]
        unsup_mask = ~sup_mask

        # ── 1. 有监督重建损失 ─────────────────────────────
        if sup_mask.any():
            l = self._recon_loss(
                pred[sup_mask], target[sup_mask], valid_mask[sup_mask]
            )
            loss = loss + self.alpha_recon * l
            detail['recon'] = l.item()

        # ── 2. 频域损失 ───────────────────────────────────
        if sup_mask.any():
            l = self._freq_loss(pred[sup_mask], target[sup_mask])
            loss = loss + self.alpha_freq * l
            detail['freq'] = l.item()

        # ── 3. 梯度平滑损失 ───────────────────────────────
        if sup_mask.any():
            l = self._grad_loss(pred[sup_mask], target[sup_mask])
            loss = loss + self.alpha_grad * l
            detail['grad'] = l.item()

        # ── 4. 质量评分损失 ───────────────────────────────
        if sup_mask.any():
            l = self._quality_loss(
                pred[sup_mask], target[sup_mask],
                quality[sup_mask], valid_mask[sup_mask]
            )
            loss = loss + self.alpha_quality * l
            detail['quality'] = l.item()

        # ── 5. Identity Loss（高SNR不乱动）────────────────
        l = self._identity_loss(x_input, pred, valid_mask)
        loss = loss + self.alpha_identity * l
        detail['identity'] = l.item()

        # ── 6. 无监督一致性损失（Part B）─────────────────
        if unsup_mask.any():
            l = self._consistency_loss(
                pred[unsup_mask], x_input[unsup_mask],
                valid_mask[unsup_mask]
            )
            loss = loss + self.alpha_consistency * l
            detail['consistency'] = l.item()

        # ── 7. 噪声对比损失 ───────────────────────────────
        l = self._noise_contrast_loss(z_noise)
        loss = loss + self.alpha_contrast * l
        detail['contrast'] = l.item()

        detail['total'] = loss.item()
        return loss, detail

    # ── 各子损失 ──────────────────────────────────────────
    def _recon_loss(self, pred, target, valid_mask):
        """加权 MSE：信号段权重高，背景段权重低"""
        vm    = valid_mask.unsqueeze(1)                    # [B,1,T]
        w     = vm * self.valid_weight + (1 - vm) * self.bg_weight
        loss  = (F.mse_loss(pred, target, reduction='none') * w)
        return loss.mean()

    def _freq_loss(self, pred, target):
        """FFT 幅度谱 MSE"""
        p_fft = torch.fft.rfft(pred,   dim=-1)
        t_fft = torch.fft.rfft(target, dim=-1)
        return F.mse_loss(p_fft.abs(), t_fft.abs())

    def _grad_loss(self, pred, target):
        """时间梯度 MSE（保持波形形态）"""
        p_grad = pred[:, :, 1:] - pred[:, :, :-1]
        t_grad = target[:, :, 1:] - target[:, :, :-1]
        return F.mse_loss(p_grad, t_grad)

    def _quality_loss(self, pred, target, quality, valid_mask):
        """
        质量评分监督：
          真实质量 = 1 - normalized_residual_power
        """
        vm        = valid_mask.unsqueeze(1)
        n         = vm.sum(dim=[1, 2]).clamp(min=1.0)
        res_power = ((pred - target) ** 2 * vm).sum(dim=[1, 2]) / n
        sig_power = (target ** 2 * vm).sum(dim=[1, 2]) / n + 1e-10
        true_q    = (1.0 - (res_power / sig_power).clamp(0, 1)).unsqueeze(1)
        return F.mse_loss(quality, true_q)

    def _identity_loss(self, x_input, pred, valid_mask):
        """高SNR样本：输出应接近输入"""
        vm        = valid_mask.unsqueeze(1)
        bg        = 1.0 - vm
        sig_n     = vm.sum(dim=[1, 2]).clamp(min=1.0)
        bg_n      = bg.sum(dim=[1, 2]).clamp(min=1.0)
        sig_pow   = (x_input.pow(2) * vm).sum(dim=[1, 2]) / sig_n
        bg_pow    = (x_input.pow(2) * bg).sum(dim=[1, 2]) / bg_n
        snr_db    = 10.0 * torch.log10(
            sig_pow.clamp(1e-10) / bg_pow.clamp(1e-10)
        )
        mask_id   = (snr_db > self.identity_snr_thr).float()   # [B]
        per_s     = F.mse_loss(pred, x_input, reduction='none').mean(dim=[1, 2])
        return (mask_id * per_s).mean()

    def _consistency_loss(self, pred, x_input, valid_mask):
        """
        Part B 无监督：
          - 背景段（P波前）应被抑制（接近0）
          - 信号段应保持（接近输入）
        """
        vm      = valid_mask.unsqueeze(1)
        bg      = 1.0 - vm

        # 背景段：输出应接近0（去噪）
        bg_loss = (pred.pow(2) * bg).mean()

        # 信号段：输出应接近输入（保持信号）
        sig_loss = F.mse_loss(pred * vm, x_input * vm)

        return bg_loss + sig_loss

    def _noise_contrast_loss(self, z_noise: torch.Tensor) -> torch.Tensor:
        """
        噪声特征对比：
          同批次不同样本的 z_noise 应该有差异
          防止模型学到退化解（所有 z 相同）
        """
        B = z_noise.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z_noise.device)

        z_norm = F.normalize(z_noise, dim=-1)           # [B, z_dim]
        sim    = torch.mm(z_norm, z_norm.t())            # [B, B]
        # 对角线（自相似）= 1，非对角线应该小
        eye    = torch.eye(B, device=z_noise.device)
        off_diag = sim * (1 - eye)
        # 惩罚过高的相似度
        return off_diag.pow(2).mean()