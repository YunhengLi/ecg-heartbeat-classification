"""Run the published RDDM checkpoint with independent wrist-window sampling."""

from hashlib import blake2b

import neurokit2 as nk
import numpy as np
from scipy.signal import resample_poly
import torch

from .rddm import AUTHOR_RATE, AUTHOR_WINDOW, author_scaled


def window_seed(window, input_rate):
    """Derive a stable seed from only the sampled PPG and its sampling rate."""
    samples = np.ascontiguousarray(window, dtype="<f8")
    digest = blake2b(digest_size=8, person=b"AFWristRDDM-v1")
    digest.update(int(input_rate).to_bytes(8, "little"))
    digest.update(samples.tobytes())
    return int.from_bytes(digest.digest(), "little")


def _predict_window(chunks, models, device, generator):
    diffusion, first, second = models
    pulse = torch.from_numpy(np.ascontiguousarray(chunks[:, None], dtype=np.float32)).to(device)

    def noise():
        return torch.randn(pulse.shape, generator=generator, dtype=pulse.dtype, device="cpu").to(device)

    # The pinned upstream sample loop draws from global Torch state and does not
    # accept a generator. Keep its reverse steps and trained modules, with a
    # CPU generator private to this window supplying all its diffusion noise.
    with torch.inference_mode():
        condition, region_condition = first(pulse), second(pulse)
        estimate = noise()
        for step in range(diffusion.n_T, 0, -1):
            extra_noise = noise() if step > 1 else 0
            time = torch.full((len(pulse),), step / diffusion.n_T, device=device, dtype=pulse.dtype)
            estimate = diffusion.region_model(estimate, region_condition, time)
            eps = diffusion.eps_model(estimate, condition, time)
            estimate = (
                diffusion.oneover_sqrta[step] * (estimate - eps * diffusion.mab_over_sqrtmab[step])
                + diffusion.sqrt_beta_t[step] * extra_noise
            )
    return estimate.cpu().numpy()[:, 0]


def generate_window_seeded_ecg(ppg, input_rate, models, *, batch_size, device):
    """Prepare groups of windows and evaluate each window in a fixed-shape call."""
    windows = np.asarray(ppg, dtype=float)
    if (windows.ndim != 2 or not windows.shape[0] or not windows.shape[1] or not np.isfinite(windows).all()
            or input_rate <= 0 or windows.shape[1] % (input_rate * AUTHOR_WINDOW) or batch_size < 1):
        raise ValueError("Provide finite PPG windows spanning whole RDDM chunks, a sampling rate, and a positive batch size")

    forecasts = np.empty_like(windows, dtype=np.float32)
    for start in range(0, len(windows), batch_size):
        section = windows[start:start + batch_size]
        scaled = author_scaled(section, input_rate).reshape(len(section), -1, AUTHOR_RATE * AUTHOR_WINDOW)
        for offset, (window, chunks) in enumerate(zip(section, scaled, strict=True)):
            cleaned = np.stack([nk.ppg_clean(chunk, sampling_rate=AUTHOR_RATE) for chunk in chunks])
            generator = torch.Generator(device="cpu").manual_seed(window_seed(window, input_rate))
            estimate = _predict_window(cleaned, models, device, generator).reshape(-1)
            forecasts[start + offset] = resample_poly(estimate, input_rate, AUTHOR_RATE, padtype="line")
        if len(windows) > batch_size:
            print(f"RDDM wrist windows: {start + len(section)}/{len(windows)}", flush=True)
    if not np.isfinite(forecasts).all():
        raise RuntimeError("RDDM did not produce finite output on the original time grid")
    return forecasts
