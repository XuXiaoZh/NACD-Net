import numpy as np


def batch_generate(n_samples=10000, fs=100, duration=60.0, seed=42):
    """
    批量生成微震数据集
    参数范围覆盖 Mw=-2 ~ Mw=2 的微震
    """
    rng = np.random.default_rng(seed)
    dataset = []

    for i in range(n_samples):
        # 随机采样参数
        Mw = rng.uniform(-2.0, 2.0)
        distance = rng.uniform(100, 5000)  # 100m ~ 5km
        azimuth = rng.uniform(0, 360)
        takeoff = rng.uniform(10, 80)
        strike = rng.uniform(0, 360)
        dip = rng.uniform(10, 90)
        rake = rng.uniform(-180, 180)

        # 噪声水平（模拟不同SNR）
        snr_db = rng.uniform(-5, 20)  # dB
        snr_lin = 10 ** (snr_db / 10)

        wave, info = synthesize_microseismic(
            Mw=Mw, distance=distance,
            azimuth=azimuth, takeoff=takeoff,
            strike=strike, dip=dip, rake=rake,
            fs=fs, duration=duration,
        )

        # 计算并叠加对应SNR的噪声
        sig_power = np.mean(wave ** 2)
        noise_power = sig_power / snr_lin
        noise = rng.normal(0, np.sqrt(noise_power), wave.shape)

        dataset.append({
            'clean': wave.astype(np.float32),
            'noisy': (wave + noise).astype(np.float32),
            'Mw': Mw,
            'distance': distance,
            'fc': info['fc'],
            'snr_db': snr_db,
            'tp': info['tp'],
            'ts': info['ts'],
        })

        if (i + 1) % 1000 == 0:
            print(f"已生成 {i + 1}/{n_samples}")

    return dataset


# 生成并保存
dataset = batch_generate(n_samples=10000)

# 保存为 HDF5
import h5py

with h5py.File('synthetic_microseismic.h5', 'w') as f:
    grp = f.create_group('data')
    for i, s in enumerate(dataset):
        sub = grp.create_group(f'sample_{i:06d}')
        sub.create_dataset('clean', data=s['clean'])
        sub.create_dataset('noisy', data=s['noisy'])
        for k in ['Mw', 'distance', 'fc', 'snr_db', 'tp', 'ts']:
            sub.attrs[k] = s[k]