# v3/model_v3.py  ── V3.1 修复版（cuDNN兼容 + skip尺寸对齐）
"""
Stage 1: NoiseEncoder    → z_noise [B, z_dim]
Stage 2: CondDenoiserV3  → clean_wave + quality_score
         (以 z_noise 为 FiLM 条件调制)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
#  工具模块
# ============================================================
class ConvBlock1d(nn.Module):
    """
    Conv1d + BN + ReLU
    ✅ 修复：stride=2 时改用 kernel=3，避免 cuDNN 不支持大核+stride
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class ConvBlockLarge(nn.Module):
    """
    大核卷积块（stride=1 时安全使用大核）
    用于 Decoder 的特征细化
    """
    def __init__(self, in_ch, out_ch, kernel=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel,
                      stride=1, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation
    out = γ(z) * x + β(z)
    """
    def __init__(self, z_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(z_dim, feat_dim)
        self.beta  = nn.Linear(z_dim, feat_dim)
        # 初始化为恒等变换
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]  z: [B, z_dim]
        gamma = self.gamma(z).unsqueeze(-1)   # [B, C, 1]
        beta  = self.beta(z).unsqueeze(-1)    # [B, C, 1]
        return gamma * x + beta

class MultiHeadSelfAttention1d(nn.Module):
    """1D 多头自注意力"""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        # ✅ 确保 d_model 能被 num_heads 整除
        while d_model % num_heads != 0:
            num_heads //= 2
        self.attn  = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        attn_out, _ = self.attn(x, x, x)
        x           = self.norm1(x + attn_out)
        x           = self.norm2(x + self.ff(x))
        return x

# ============================================================
#  Stage 1: 噪声编码器
# ============================================================
class NoiseEncoder(nn.Module):
    """
    输入：背景噪声片段 [B, 3, cond_len]
    输出：噪声特征向量 z_noise [B, z_dim]

    ✅ 修复：所有 stride=2 层改用 kernel=3
    结构：
      Conv(k=3,s=2) × 4  → 下采样
      → Transformer Attention
      → Global Average Pooling
      → FC → z_noise
    """

    def __init__(
        self,
        in_ch:     int = 3,
        z_dim:     int = 128,
        cond_len:  int = 400,
        num_heads: int = 8,
    ):
        super().__init__()
        self.z_dim = z_dim

        # ✅ 全部用 kernel=3, stride=2（cuDNN 安全）
        self.enc = nn.Sequential(
            ConvBlock1d(in_ch, 32,  kernel=3, stride=2),   # 400→200
            ConvBlock1d(32,    64,  kernel=3, stride=2),   # 200→100
            ConvBlock1d(64,    128, kernel=3, stride=2),   # 100→50
            ConvBlock1d(128,   128, kernel=3, stride=2),   # 50→25
            # stride=1 细化
            ConvBlockLarge(128, 128, kernel=15),
        )

        self.attn = MultiHeadSelfAttention1d(128, num_heads=num_heads)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(128, z_dim),
            nn.LayerNorm(z_dim),
            nn.Tanh(),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: [B, 3, cond_len]  →  z_noise: [B, z_dim]"""
        x = self.enc(cond)                     # [B, 128, ~25]
        x = x.permute(0, 2, 1)                # [B, ~25, 128]
        x = self.attn(x)                       # [B, ~25, 128]
        x = x.permute(0, 2, 1)                # [B, 128, ~25]
        x = self.pool(x).squeeze(-1)           # [B, 128]
        return self.proj(x)                    # [B, z_dim]

# ============================================================
#  Stage 2: 条件去噪网络
# ============================================================
class CondDenoiserV3(nn.Module):
    """
    U-Net + FiLM 条件调制 + Transformer Bottleneck

    ✅ 修复要点：
      1. 所有 stride=2 用 kernel=3
      2. Decoder 每层用 F.interpolate 精确对齐 skip 尺寸
      3. Bottleneck 不再二次下采样，直接在 enc5 尺寸上做 Attention
    """

    def __init__(
        self,
        in_ch:     int = 3,
        z_dim:     int = 128,
        num_heads: int = 8,
    ):
        super().__init__()
        self.in_ch = in_ch

        # ── Encoder（5 层，stride=2 下采样）─────────────
        # ✅ stride=2 全用 kernel=3
        self.enc1 = ConvBlock1d(in_ch, 16,  kernel=3, stride=2)  # T→T/2
        self.enc2 = ConvBlock1d(16,    32,  kernel=3, stride=2)  # T/2→T/4
        self.enc3 = ConvBlock1d(32,    64,  kernel=3, stride=2)  # T/4→T/8
        self.enc4 = ConvBlock1d(64,    128, kernel=3, stride=2)  # T/8→T/16
        self.enc5 = ConvBlock1d(128,   128, kernel=3, stride=2)  # T/16→T/32

        # stride=1 细化（大核安全）
        self.ref1 = ConvBlockLarge(16,  16,  kernel=15)
        self.ref2 = ConvBlockLarge(32,  32,  kernel=15)
        self.ref3 = ConvBlockLarge(64,  64,  kernel=15)
        self.ref4 = ConvBlockLarge(128, 128, kernel=7)
        self.ref5 = ConvBlockLarge(128, 128, kernel=5)

        # ── FiLM 调制（每个 encoder 层后）───────────────
        self.film1 = FiLM(z_dim, 16)
        self.film2 = FiLM(z_dim, 32)
        self.film3 = FiLM(z_dim, 64)
        self.film4 = FiLM(z_dim, 128)
        self.film5 = FiLM(z_dim, 128)

        # ── Transformer Bottleneck（在 enc5 尺寸上）─────
        # ✅ 不再二次下采样，直接做 Attention
        self.bn_conv = nn.Sequential(
            ConvBlockLarge(128, 128, kernel=5),
            ConvBlockLarge(128, 128, kernel=3),
        )
        self.bn_attn = MultiHeadSelfAttention1d(128, num_heads)
        self.bn_film = FiLM(z_dim, 128)

        # ── Decoder（4 层，Upsample×2 + skip concat）────
        # ✅ 输入通道 = 上采样特征 + skip 特征
        self.dec4 = nn.Sequential(
            ConvBlock1d(128 + 128, 128, kernel=3, stride=1),
            ConvBlockLarge(128, 64, kernel=15),
        )
        self.dec3 = nn.Sequential(
            ConvBlock1d(64 + 64, 64, kernel=3, stride=1),
            ConvBlockLarge(64, 32, kernel=15),
        )
        self.dec2 = nn.Sequential(
            ConvBlock1d(32 + 32, 32, kernel=3, stride=1),
            ConvBlockLarge(32, 16, kernel=15),
        )
        self.dec1 = nn.Sequential(
            ConvBlock1d(16 + 16, 16, kernel=3, stride=1),
            ConvBlockLarge(16, 16, kernel=15),
        )
        self.dec0 = nn.Sequential(
            ConvBlock1d(16, 16, kernel=3, stride=1),
            ConvBlockLarge(16, 16, kernel=7),
        )

        # FiLM for decoder
        self.film_d4 = FiLM(z_dim, 64)
        self.film_d3 = FiLM(z_dim, 32)
        self.film_d2 = FiLM(z_dim, 16)
        self.film_d1 = FiLM(z_dim, 16)

        # ── 输出层 ────────────────────────────────────────
        self.out_conv = nn.Conv1d(16, in_ch, kernel_size=1)
        self.out_act  = nn.Tanh()

        # ── 质量评分头 ────────────────────────────────────
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _align(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        ✅ 核心修复：精确对齐时间维度
        使用 F.interpolate 而非 Upsample，支持任意目标尺寸
        """
        if x.shape[-1] != target.shape[-1]:
            x = F.interpolate(
                x,
                size=target.shape[-1],
                mode='linear',
                align_corners=False,
            )
        return x

    @staticmethod
    def _upsample2x(x: torch.Tensor) -> torch.Tensor:
        """上采样 ×2"""
        return F.interpolate(
            x,
            scale_factor=2,
            mode='linear',
            align_corners=False,
        )

    def forward(
        self,
        x:       torch.Tensor,   # [B, 3, T]
        z_noise: torch.Tensor,   # [B, z_dim]
    ) -> tuple:

        # ── Encoder ──────────────────────────────────────
        # 每层：Conv(stride=2) → 细化(stride=1) → FiLM
        e1 = self.film1(self.ref1(self.enc1(x)),  z_noise)  # [B, 16,  T/2]
        e2 = self.film2(self.ref2(self.enc2(e1)), z_noise)  # [B, 32,  T/4]
        e3 = self.film3(self.ref3(self.enc3(e2)), z_noise)  # [B, 64,  T/8]
        e4 = self.film4(self.ref4(self.enc4(e3)), z_noise)  # [B, 128, T/16]
        e5 = self.film5(self.ref5(self.enc5(e4)), z_noise)  # [B, 128, T/32]

        # ── Bottleneck（在 e5 尺寸上做 Attention）────────
        bn = self.bn_conv(e5)                  # [B, 128, T/32]
        bn = bn.permute(0, 2, 1)               # [B, T/32, 128]
        bn = self.bn_attn(bn)                  # [B, T/32, 128]
        bn = bn.permute(0, 2, 1)               # [B, 128, T/32]
        bn = self.bn_film(bn, z_noise)         # FiLM 调制

        # ── Decoder ──────────────────────────────────────
        # ✅ 每次 upsample 后立即对齐到对应 encoder 层尺寸

        # d4: bn(T/32) → ×2 → 对齐e4(T/16) → cat(e4) → conv
        d4 = self._upsample2x(bn)              # [B, 128, ~T/16]
        d4 = self._align(d4, e4)               # ✅ 精确对齐
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  # [B, 64, T/16]
        d4 = self.film_d4(d4, z_noise)

        # d3: d4(T/16) → ×2 → 对齐e3(T/8) → cat(e3) → conv
        d3 = self._upsample2x(d4)              # [B, 64, ~T/8]
        d3 = self._align(d3, e3)               # ✅ 精确对齐
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # [B, 32, T/8]
        d3 = self.film_d3(d3, z_noise)

        # d2: d3(T/8) → ×2 → 对齐e2(T/4) → cat(e2) → conv
        d2 = self._upsample2x(d3)              # [B, 32, ~T/4]
        d2 = self._align(d2, e2)               # ✅ 精确对齐
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # [B, 16, T/4]
        d2 = self.film_d2(d2, z_noise)

        # d1: d2(T/4) → ×2 → 对齐e1(T/2) → cat(e1) → conv
        d1 = self._upsample2x(d2)              # [B, 16, ~T/2]
        d1 = self._align(d1, e1)               # ✅ 精确对齐
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # [B, 16, T/2]
        d1 = self.film_d1(d1, z_noise)

        # d0: d1(T/2) → ×2 → 对齐x(T) → conv
        d0 = self._upsample2x(d1)              # [B, 16, ~T]
        d0 = self._align(d0, x)                # ✅ 精确对齐到输入尺寸
        d0 = self.dec0(d0)                     # [B, 16, T]

        # ── 输出 ──────────────────────────────────────────
        quality = self.quality_head(d0)        # [B, 1]
        out     = self.out_act(
            self.out_conv(d0)
        )                                      # [B, 3, T]

        return out, quality

# ============================================================
#  完整模型
# ============================================================
class NoiseAwareDenoiserV3(nn.Module):
    """
    Stage 1: NoiseEncoder(z_cond) → z_noise
    Stage 2: CondDenoiserV3(x, z_noise) → clean, quality
    """

    def __init__(
        self,
        in_ch:     int = 3,
        z_dim:     int = 128,
        cond_len:  int = 400,
        num_heads: int = 8,
    ):
        super().__init__()
        self.noise_encoder = NoiseEncoder(
            in_ch=in_ch, z_dim=z_dim,
            cond_len=cond_len, num_heads=num_heads,
        )
        self.denoiser = CondDenoiserV3(
            in_ch=in_ch, z_dim=z_dim, num_heads=num_heads,
        )

    def forward(
        self,
        x:      torch.Tensor,   # [B, 3, T]
        z_cond: torch.Tensor,   # [B, 3, C]
    ) -> tuple:
        """返回: (clean [B,3,T], quality [B,1], z_noise [B,z_dim])"""
        z_noise        = self.noise_encoder(z_cond)
        clean, quality = self.denoiser(x, z_noise)
        return clean, quality, z_noise

    def encode_noise(self, z_cond: torch.Tensor) -> torch.Tensor:
        return self.noise_encoder(z_cond)

    def denoise(self, x: torch.Tensor, z_noise: torch.Tensor) -> tuple:
        return self.denoiser(x, z_noise)

# ============================================================
#  尺寸验证（运行此文件直接测试）
# ============================================================
if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = NoiseAwareDenoiserV3(
        in_ch=3, z_dim=128, cond_len=400, num_heads=8
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {n_params/1e6:.2f} M")

    # 测试不同 signal_len
    for T in [6000, 3000, 8000]:
        x      = torch.randn(2, 3, T).to(device)
        z_cond = torch.randn(2, 3, 400).to(device)
        try:
            clean, quality, z_noise = model(x, z_cond)
            print(f"✅ T={T:5d} → clean={tuple(clean.shape)}, "
                  f"quality={tuple(quality.shape)}, "
                  f"z_noise={tuple(z_noise.shape)}")
        except Exception as e:
            print(f"❌ T={T}: {e}")
            sys.exit(1)

    print("\n[✅ 所有尺寸测试通过]")