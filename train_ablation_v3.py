# v3/train_v3.py  ── 完整训练主程序 V3（支持消融实验）
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"/home/featurize/work/denoise/part1")

from v3.dataset_v3 import STEADDatasetV3
from v3.model_v3   import NoiseAwareDenoiserV3
from v3.loss_v3    import DenoiserLossV3

# ============================================================
#  基础配置
# ============================================================
CONFIG = {
    # 数据路径
    "event_h5":   "D:/X/p_wave/data/chunk2.hdf5",
    "event_csv":  "D:/X/p_wave/data/chunk2.csv",
    "noise_h5":   "D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv":  "D:/X/p_wave/data/chunk1.csv",
    "raw_h5":     None,
    "raw_csv":    None,

    "save_dir":   "v3/checkpoints_v3",

    # 模型参数
    "z_dim":      128,
    "signal_len": 6000,
    "cond_len":   400,
    "num_heads":  8,

    # 训练参数
    "epochs":        50,
    "batch_size":    8,
    "lr":            1e-4,
    "weight_decay":  1e-4,
    "num_workers":   2,
    "val_frac":      0.1,
    "seed":          42,

    # 损失权重（基础值）
    "alpha_recon":       1.0,
    "alpha_freq":        0.15,
    "alpha_grad":        0.15,
    "alpha_quality":     0.10,
    "alpha_identity":    30.0,
    "alpha_consistency": 0.20,
    "alpha_contrast":    0.05,
    "valid_weight":      3.0,
    "bg_weight":         0.3,
    "identity_snr_thr":  6.0,

    # 数据增强
    "snr_range":    (0.1, 20.0),
    "clean_prob":   0.10,
    "part_b_ratio": 0.3,

    # ============================================================
    #  消融实验开关（默认全开 = 完整模型）
    # ============================================================
    "ablation": {
        "exp_name":            "full_model",   # 实验名，影响 save_dir
        "use_noise_condition": True,           # 噪声条件编码器 z_cond
        "use_quality_branch":  True,           # quality 分支损失
        "use_freq_loss":       True,           # 频域损失
        "use_grad_loss":       True,           # 梯度损失
        "use_identity_loss":   True,           # identity loss
        "use_contrast_loss":   True,           # 噪声对比损失
        "use_consistency_loss":True,           # 无监督一致性损失
    },
}

# ============================================================
#  命令行参数解析（支持从命令行覆盖消融开关）
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="V3 训练脚本（支持消融实验）")

    # 消融开关
    parser.add_argument("--exp_name",              type=str,  default=None)
    parser.add_argument("--use_noise_condition",   type=int,  default=None, help="0/1")
    parser.add_argument("--use_quality_branch",    type=int,  default=None, help="0/1")
    parser.add_argument("--use_freq_loss",         type=int,  default=None, help="0/1")
    parser.add_argument("--use_grad_loss",         type=int,  default=None, help="0/1")
    parser.add_argument("--use_identity_loss",     type=int,  default=None, help="0/1")
    parser.add_argument("--use_contrast_loss",     type=int,  default=None, help="0/1")
    parser.add_argument("--use_consistency_loss",  type=int,  default=None, help="0/1")

    # 训练参数覆盖
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--save_dir",    type=str,   default=None)

    # 数据路径覆盖
    parser.add_argument("--event_h5",   type=str, default=None)
    parser.add_argument("--event_csv",  type=str, default=None)
    parser.add_argument("--noise_h5",   type=str, default=None)
    parser.add_argument("--noise_csv",  type=str, default=None)

    return parser.parse_args()

def apply_args_to_config(args, config):
    """将命令行参数覆盖到 CONFIG"""
    # 消融开关
    ablation_keys = [
        "use_noise_condition", "use_quality_branch",
        "use_freq_loss",       "use_grad_loss",
        "use_identity_loss",   "use_contrast_loss",
        "use_consistency_loss",
    ]
    for key in ablation_keys:
        val = getattr(args, key, None)
        if val is not None:
            config["ablation"][key] = bool(val)

    if args.exp_name is not None:
        config["ablation"]["exp_name"] = args.exp_name

    # 训练参数
    if args.epochs     is not None: config["epochs"]     = args.epochs
    if args.batch_size is not None: config["batch_size"] = args.batch_size
    if args.lr         is not None: config["lr"]         = args.lr
    if args.save_dir   is not None: config["save_dir"]   = args.save_dir

    # 数据路径
    if args.event_h5  is not None: config["event_h5"]  = args.event_h5
    if args.event_csv is not None: config["event_csv"] = args.event_csv
    if args.noise_h5  is not None: config["noise_h5"]  = args.noise_h5
    if args.noise_csv is not None: config["noise_csv"] = args.noise_csv

    return config

# ============================================================
#  消融：动态构建 Loss 权重
# ============================================================
def build_loss_weights(config):
    """
    根据消融开关，将对应 loss 的 alpha 设为 0
    """
    ab = config["ablation"]
    return {
        "alpha_recon":       config["alpha_recon"],
        "alpha_freq":        config["alpha_freq"]        if ab["use_freq_loss"]        else 0.0,
        "alpha_grad":        config["alpha_grad"]        if ab["use_grad_loss"]        else 0.0,
        "alpha_quality":     config["alpha_quality"]     if ab["use_quality_branch"]   else 0.0,
        "alpha_identity":    config["alpha_identity"]    if ab["use_identity_loss"]    else 0.0,
        "alpha_consistency": config["alpha_consistency"] if ab["use_consistency_loss"] else 0.0,
        "alpha_contrast":    config["alpha_contrast"]    if ab["use_contrast_loss"]    else 0.0,
        "valid_weight":      config["valid_weight"],
        "bg_weight":         config["bg_weight"],
        "identity_snr_thr":  config["identity_snr_thr"],
    }

# ============================================================
#  消融：动态处理 batch 输入
# ============================================================
def prepare_batch(batch, ablation_cfg, device):
    """
    根据消融配置动态修改 batch 输入：
      - use_noise_condition=False → z_cond 替换为全零（噪声编码器接收空信号）
    """
    x          = batch['x'].to(device)
    y_clean    = batch['y_clean'].to(device)
    z_cond     = batch['z_cond'].to(device)
    valid_mask = batch['valid_mask'].to(device)
    has_target = batch['has_target'].to(device)

    # 消融：禁用噪声条件编码器
    if not ablation_cfg["use_noise_condition"]:
        z_cond = torch.zeros_like(z_cond)

    return x, y_clean, z_cond, valid_mask, has_target

# ============================================================
#  打印消融配置摘要
# ============================================================
def print_ablation_summary(ablation_cfg):
    print("\n" + "=" * 60)
    print("  消融实验配置 (Ablation Configuration)")
    print("=" * 60)
    status = {True: "✅ ON ", False: "❌ OFF"}
    print(f"  实验名称          : {ablation_cfg['exp_name']}")
    print(f"  噪声条件编码器     : {status[ablation_cfg['use_noise_condition']]}")
    print(f"  Quality 分支损失   : {status[ablation_cfg['use_quality_branch']]}")
    print(f"  频域损失           : {status[ablation_cfg['use_freq_loss']]}")
    print(f"  梯度损失           : {status[ablation_cfg['use_grad_loss']]}")
    print(f"  Identity 损失      : {status[ablation_cfg['use_identity_loss']]}")
    print(f"  对比损失           : {status[ablation_cfg['use_contrast_loss']]}")
    print(f"  一致性损失         : {status[ablation_cfg['use_consistency_loss']]}")
    print("=" * 60 + "\n")

# ============================================================
#  划分数据集
# ============================================================
def split_dataset(csv_path, val_frac=0.1, seed=42, prefix="v3"):
    df       = pd.read_csv(csv_path, low_memory=False)
    val_df   = df.sample(frac=val_frac, random_state=seed)
    train_df = df.drop(val_df.index)
    train_path = f"{prefix}_train.csv"
    val_path   = f"{prefix}_val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    print(f"[Split] Train={len(train_df)}, Val={len(val_df)}")
    return train_path, val_path

# ============================================================
#  SNR 评估
# ============================================================
def compute_snr_batch(clean, residual, valid_mask):
    mask      = valid_mask.unsqueeze(1)
    n         = mask.sum(dim=[1, 2]) * clean.shape[1] + 1e-10
    sig_power = (clean    ** 2 * mask).sum(dim=[1, 2]) / n
    res_power = (residual ** 2 * mask).sum(dim=[1, 2]) / n + 1e-10
    snr       = 10.0 * torch.log10(sig_power / res_power)
    return torch.clamp(snr, -50, 50)

# ============================================================
#  梯度检查
# ============================================================
def _check_grad_nan(model):
    try:
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad.detach().cpu()).all():
                return True
        return False
    except Exception:
        return True

# ============================================================
#  训练一个 epoch
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion,
                    device, epoch, ablation_cfg):
    model.train()
    total_loss    = 0.0
    total_samples = 0
    loss_details  = {}
    nan_count     = 0

    for batch_idx, batch in enumerate(loader):

        # ── 消融：动态处理输入 ────────────────────────────
        x, y_clean, z_cond, valid_mask, has_target = prepare_batch(
            batch, ablation_cfg, device
        )

        # 输入检查
        if not all(torch.isfinite(t).all() for t in [x, y_clean, z_cond]):
            nan_count += 1
            continue

        optimizer.zero_grad()

        try:
            pred, quality, z_noise = model(x, z_cond)
        except Exception as e:
            nan_count += 1
            if nan_count <= 3:
                print(f"  ⚠ Batch {batch_idx} 前向异常: {e}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if not torch.isfinite(pred.detach().cpu()).all():
            nan_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        try:
            loss, detail = criterion(
                pred       = pred,
                target     = y_clean,
                quality    = quality,
                x_input    = x,
                z_noise    = z_noise,
                valid_mask = valid_mask,
                has_target = has_target,
            )
        except Exception as e:
            nan_count += 1
            if nan_count <= 3:
                print(f"  ⚠ Batch {batch_idx} loss 异常: {e}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if not torch.isfinite(loss.detach().cpu()).all():
            nan_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        if _check_grad_nan(model):
            nan_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs             = x.size(0)
        total_loss    += loss.item() * bs
        total_samples += bs
        for k, v in detail.items():
            if np.isfinite(v):
                loss_details[k] = loss_details.get(k, 0.0) + v * bs

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"loss={loss.item():.4f} "
                  f"recon={detail.get('recon', 0):.4f} "
                  f"identity={detail.get('identity', 0):.4f} "
                  f"(skip={nan_count})")

    if total_samples == 0:
        return float('nan'), {}

    avg_loss    = total_loss / total_samples
    avg_details = {k: v / total_samples for k, v in loss_details.items()}
    if nan_count > 0:
        print(f"  ⚠ 跳过 {nan_count} 个异常 batch")
    return avg_loss, avg_details

# ============================================================
#  验证一个 epoch
# ============================================================
def validate_one_epoch(model, loader, criterion,
                       device, ablation_cfg):
    model.eval()
    total_loss    = 0.0
    total_samples = 0
    loss_details  = {}
    all_snr_gain  = []
    all_snr_in    = []
    all_snr_out   = []
    all_quality   = []

    with torch.no_grad():
        for batch in loader:

            # ── 消融：动态处理输入 ────────────────────────
            x, y_clean, z_cond, valid_mask, has_target = prepare_batch(
                batch, ablation_cfg, device
            )

            if not all(torch.isfinite(t).all() for t in [x, y_clean, z_cond]):
                continue

            try:
                pred, quality, z_noise = model(x, z_cond)
            except Exception:
                continue

            if not torch.isfinite(pred.detach().cpu()).all():
                continue

            try:
                loss, detail = criterion(
                    pred=pred, target=y_clean, quality=quality,
                    x_input=x, z_noise=z_noise,
                    valid_mask=valid_mask, has_target=has_target,
                )
            except Exception:
                continue

            if not torch.isfinite(loss.detach().cpu()).all():
                continue

            bs             = x.size(0)
            total_loss    += loss.item() * bs
            total_samples += bs
            for k, v in detail.items():
                if np.isfinite(v):
                    loss_details[k] = loss_details.get(k, 0.0) + v * bs

            # SNR 评估（只对有监督样本）
            sup = has_target.bool()
            if sup.any():
                noise_in  = x[sup]    - y_clean[sup]
                noise_out = pred[sup] - y_clean[sup]
                try:
                    snr_in  = compute_snr_batch(
                        y_clean[sup], noise_in,  valid_mask[sup]
                    )
                    snr_out = compute_snr_batch(
                        y_clean[sup], noise_out, valid_mask[sup]
                    )
                    all_snr_in.append(snr_in.cpu().numpy())
                    all_snr_out.append(snr_out.cpu().numpy())
                    all_snr_gain.append((snr_out - snr_in).cpu().numpy())
                except Exception:
                    pass

            all_quality.append(quality.cpu().numpy().mean())

    if total_samples == 0:
        return float('nan'), float('nan'), {}

    avg_loss    = total_loss / total_samples
    avg_details = {k: v / total_samples for k, v in loss_details.items()}

    all_snr_gain = np.concatenate(all_snr_gain) if all_snr_gain else np.array([])
    valid_gain   = all_snr_gain[
        np.isfinite(all_snr_gain) & (np.abs(all_snr_gain) < 50)
    ] if len(all_snr_gain) > 0 else np.array([])

    mean_gain    = float(valid_gain.mean())     if len(valid_gain) > 0 else float('nan')
    median_gain  = float(np.median(valid_gain)) if len(valid_gain) > 0 else float('nan')
    mean_in      = float(np.nanmean(np.concatenate(all_snr_in)))  if all_snr_in  else float('nan')
    mean_out     = float(np.nanmean(np.concatenate(all_snr_out))) if all_snr_out else float('nan')
    mean_quality = float(np.mean(all_quality))  if all_quality else float('nan')

    print(f"  Val Loss    = {avg_loss:.4f}")
    print(f"  Input  SNR  = {mean_in:.2f} dB  →  Output SNR = {mean_out:.2f} dB")
    print(f"  SNR Gain    = {mean_gain:.2f} dB (Median={median_gain:.2f})")
    print(f"  Quality     = {mean_quality:.4f}")
    for k in ['recon', 'freq', 'grad', 'identity', 'consistency', 'contrast']:
        v = avg_details.get(k, float('nan'))
        if np.isfinite(v):
            print(f"    {k:12s} = {v:.4f}")

    return avg_loss, mean_gain, avg_details

# ============================================================
#  主函数
# ============================================================
def main():
    # ── 解析命令行参数 ────────────────────────────────────
    args = parse_args()
    apply_args_to_config(args, CONFIG)

    # ── 根据实验名自动设置 save_dir ───────────────────────
    exp_name = CONFIG["ablation"]["exp_name"]
    CONFIG["save_dir"] = os.path.join(
        "v3/checkpoints_ablation", exp_name
    )

    ablation_cfg = CONFIG["ablation"]

    # ── 打印消融配置 ──────────────────────────────────────
    print_ablation_summary(ablation_cfg)

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # 保存本次实验完整配置
    save_config = {k: str(v) for k, v in CONFIG.items() if k != "ablation"}
    save_config["ablation"] = ablation_cfg
    with open(os.path.join(CONFIG["save_dir"], "config.json"), "w") as f:
        json.dump(save_config, f, indent=2, ensure_ascii=False)

    # ── 数据集 ────────────────────────────────────────────
    train_csv, val_csv = split_dataset(
        CONFIG["event_csv"],
        val_frac = CONFIG["val_frac"],
        seed     = CONFIG["seed"],
        prefix   = f"v3_{exp_name}",
    )

    def make_ds(csv_path, aug=True):
        return STEADDatasetV3(
            event_h5_path  = CONFIG["event_h5"],
            event_csv_path = csv_path,
            noise_h5_path  = CONFIG["noise_h5"],
            noise_csv_path = CONFIG["noise_csv"],
            raw_h5_path    = CONFIG["raw_h5"],
            raw_csv_path   = CONFIG["raw_csv"],
            signal_len     = CONFIG["signal_len"],
            cond_len       = CONFIG["cond_len"],
            snr_range      = CONFIG["snr_range"],
            clean_prob     = CONFIG["clean_prob"] if aug else 0.0,
            part_b_ratio   = CONFIG["part_b_ratio"] if aug else 0.0,
            seed           = CONFIG["seed"],
        )

    train_ds = make_ds(train_csv, aug=True)
    val_ds   = make_ds(val_csv,   aug=False)

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"],
        shuffle=True,  num_workers=CONFIG["num_workers"], pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,   batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=False,
    )

    # ── 设备 & 模型 ───────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device   : {device}")
    print(f"[INFO] Save Dir : {CONFIG['save_dir']}")

    model = NoiseAwareDenoiserV3(
        z_dim     = CONFIG["z_dim"],
        cond_len  = CONFIG["cond_len"],
        num_heads = CONFIG["num_heads"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 模型参数量: {n_params/1e6:.2f} M")

    # 加载已有权重（断点续训）
    ckpt_path = os.path.join(CONFIG["save_dir"], f"best_{exp_name}.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[INFO] 恢复权重: {ckpt_path}")

    # ── 损失函数（消融控制 alpha）────────────────────────
    loss_weights = build_loss_weights(CONFIG)
    print("\n[INFO] 损失权重:")
    for k, v in loss_weights.items():
        if k.startswith("alpha"):
            status = "✅" if v > 0 else "❌"
            print(f"  {status} {k:25s} = {v}")

    criterion = DenoiserLossV3(**loss_weights)

    # ── 优化器 & 调度器 ───────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6,
    )

    best_val_loss = float('inf')
    history       = []

    print("\n" + "=" * 60)
    print(f"  开始训练：{exp_name}")
    print("=" * 60)

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n[Epoch {epoch}/{CONFIG['epochs']}]")

        train_loss, train_detail = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, ablation_cfg,
        )
        val_loss, val_snr_gain, val_detail = validate_one_epoch(
            model, val_loader, criterion,
            device, ablation_cfg,
        )

        # 学习率调度（手动打印变化）
        if np.isfinite(val_loss):
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  📉 LR 调整: {old_lr:.2e} → {new_lr:.2e}")

        # 记录历史
        history.append({
            "epoch":      epoch,
            "train_loss": train_loss if np.isfinite(train_loss) else None,
            "val_loss":   val_loss   if np.isfinite(val_loss)   else None,
            "snr_gain":   float(val_snr_gain) if np.isfinite(val_snr_gain) else None,
            **{f"train_{k}": v for k, v in train_detail.items()},
        })

        # 保存最优模型
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [✓] 保存最优模型 → {ckpt_path}  val_loss={val_loss:.4f}")

        # 每 5 epoch 保存一次 checkpoint
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(CONFIG["save_dir"], f"ckpt_epoch{epoch}.pth"),
            )

        # 保存历史
        with open(os.path.join(CONFIG["save_dir"], "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        tl = f"{train_loss:.4f}" if np.isfinite(train_loss) else "nan"
        vl = f"{val_loss:.4f}"   if np.isfinite(val_loss)   else "nan"
        print(f"  Train={tl} | Val={vl} | Best={best_val_loss:.4f} | "
              f"LR={optimizer.param_groups[0]['lr']:.2e}")

    print(f"\n[✅ 训练完成] 实验={exp_name}  最优 Val Loss={best_val_loss:.4f}")

if __name__ == "__main__":
    main()