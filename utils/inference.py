import gc
import inspect
import numpy as np, soundfile as sf
import threading, torch
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

STEM_NAMES = ["bass", "drums", "guitar", "piano"]
INITIAL_MAX_BATCH_SIZE = 1 # primary GPU. less because also holds the VAE and vocoder
MAX_BATCH_SIZE = 1 # worker GPUs


def _save_stems(out_dir: Path, audio, mels, sr) -> None:
    """Save one result: per-stem WAVs and mels.npz into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(STEM_NAMES):
        sf.write(str(out_dir / f"{name}.wav"), audio[i], sr, subtype="PCM_16")
    np.savez_compressed(str(out_dir / "mels.npz"), mels=mels.cpu().numpy())


def _resolve_devices(devices=None):
    if devices is not None:
        return [torch.device(d) for d in devices]
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


def _build_worker_sampler(sampler, device):
    dm = sampler.get_diffusion_model()
    return type(sampler)(dm.fork(device))


def _cleanup_workers(w_samplers: list) -> None:
    """Move worker models off GPU, break reference cycles, then flush CUDA cache"""
    for ws in w_samplers[1:]:
        ws.get_diffusion_model().to('cpu')
    w_samplers[1:] = []
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def _run_sampler(
    sampler,
    cond,
    *,
    bs,
    steps=200,
    ddim_discretize="uniform",
    eta=1.0,
    cfg=1.0,
    rho=7.0,
    s_churn=0.0,
    s_min=0.0,
    s_max=float("inf"),
    s_noise=1.0,
):
    """
    Core diffusion sampling

    Returns: samples (bs, num_stems, z_channels, T, F) in scaled latent space.
    """
    dm = sampler.get_diffusion_model()
    shape = (dm.num_stems, dm.z_channels, dm.latent_t_size, dm.latent_f_size)

    all_kwargs = dict(
        shape=shape,
        conditioning=cond,
        steps=steps,
        batch_size=bs,
        eta=eta,
        verbose=False,
        cfg_scale=cfg,
        ddim_discretize=ddim_discretize,
        rho=rho,
        s_churn=s_churn,
        s_min=s_min,
        s_max=s_max,
        s_noise=s_noise,
    )

    sig = inspect.signature(sampler.sample)
    named = {
        name for name, p in sig.parameters.items()
        if p.kind != inspect.Parameter.VAR_KEYWORD
    }
    return sampler.sample(**{k: v for k, v in all_kwargs.items() if k in named})


def _decode(samples, dm, vae, vocoder):
    # (B, S, C, T, F) -> audio (B, S, num_samples), mels (B, S, 1, n_mels, T_mel)
    B, S, C, T, F = samples.shape
    z = (samples / dm.scale_factor).reshape(B * S, C, T, F)
    m = vae.decode(z)
    w = vocoder.mel_to_audio(m.squeeze(1))
    return w.reshape(B, S, -1), m.reshape(B, S, *m.shape[1:])


def _allot(n_samples, n_workers):
    jobs = []
    if INITIAL_MAX_BATCH_SIZE == 0:
        n_active = n_workers - 1
        if n_active == 0:
            raise RuntimeError("INITIAL_MAX_BATCH_SIZE=0 requires at least 2 GPUs.")
        base, extra = divmod(n_samples, n_active)
        for idx, wi in enumerate(range(1, n_workers)):
            count = base + (1 if idx < extra else 0)
            for s in range(0, count, MAX_BATCH_SIZE):
                jobs.append((wi, min(MAX_BATCH_SIZE, count - s)))
    else:
        base, extra = divmod(n_samples, n_workers)
        for wi in range(n_workers):
            count = base + (1 if wi < extra else 0)
            bs = INITIAL_MAX_BATCH_SIZE if wi == 0 else MAX_BATCH_SIZE
            for s in range(0, count, bs):
                jobs.append((wi, min(bs, count - s)))
    return jobs  # [(worker_idx, batch_size)]


def _dispatch(jobs, primary_dm, vae, vocoder, primary_dev, leave_as_latent):
    raw = [None] * len(jobs)
    errors = []
    groups = defaultdict(list)
    for i, (wi, *_) in enumerate(jobs):
        groups[wi].append(i)

    def run(indices):
        try:
            print(f"Worker {indices[0]} starting with {len(indices)} batches...")
            for i in indices:
                _wi, samp, cond, bs, kwargs = jobs[i]
                with torch.no_grad():
                    raw[i] = _run_sampler(samp, cond, bs=bs, **kwargs).cpu()
                torch.cuda.empty_cache()
            print(f"Worker {indices[0]} finished.")
        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"Worker {indices[0]} OOM")
            else:
                print(f"Worker {indices[0]} encountered an error: {e}")
            torch.cuda.empty_cache()
            errors.append(e)

    if len(groups) == 1:
        run(next(iter(groups.values())))
    else:
        ts = [threading.Thread(target=run, args=(groups[wi],), daemon=True) for wi in groups]
        for t in ts: t.start()
        for t in ts: t.join()

    if errors:
        raise errors[0]

    if leave_as_latent:
        return raw

    out = []
    for idx in range(len(raw)):
        with torch.no_grad():
            audio, mels = _decode(raw[idx].to(primary_dev), primary_dm, vae, vocoder)
        raw[idx] = None
        for j in range(audio.shape[0]):
            out.append((audio[j], mels[j].cpu()))
        del audio, mels
        torch.cuda.empty_cache()
    return out


def generate_stems(
    sampler,
    vae,
    vocoder,
    *,
    n_samples=1,
    devices=None,
    ddim_steps=200,
    ddim_discretize="uniform",
    ddim_eta=1.0,
    edm_rho=7.0,
    edm_s_churn=0.0,
    edm_s_min=0.0,
    edm_s_max=float("inf"),
    edm_s_noise=1.0,
    leave_as_latent=False,
    out_dir: Optional[str] = None,
    start_from: Optional[int] = 0,
) -> List[Tuple]:
    """
    Generate n_samples unconditional samples.

    Args:
        ddim_discretize : timestep spacing for DDIM/Heun, "uniform" or "quad".

    Returns List of n_samples (audio, mels):
        audio : int16 numpy (num_stems, num_samples)
        mels : float tensor (num_stems, 1, n_mels, T_mel)
    If leave_as_latent is True, returns List of n_samples tensors
        (num_stems, z_channels, latent_t_size, latent_f_size)
    If out_dir is given, each sample is saved to out_dir/sample_NNNN/.
    """
    resolved = _resolve_devices(devices)
    primary = resolved[0]
    primary_dm = sampler.get_diffusion_model().to(primary).eval()
    vae.to(primary).eval()
    vocoder.to(primary).eval()

    allot = _allot(n_samples, min(len(resolved), n_samples))
    n_workers = max(wi for wi, _ in allot) + 1
    w_samplers = [sampler] + [_build_worker_sampler(sampler, resolved[i]) for i in range(1, n_workers)]

    shape = (primary_dm.num_stems, primary_dm.z_channels, primary_dm.latent_t_size, primary_dm.latent_f_size)

    jobs = []
    sample_idx = 0
    for wi, bs in allot:
        with torch.no_grad():
            cond = w_samplers[wi].get_diffusion_model().cond_stage_model.get_unconditional_condition(bs)
        kwargs = dict(
            steps=ddim_steps,
            ddim_discretize=ddim_discretize,
            eta=ddim_eta,
            cfg=1.0,
            rho=edm_rho,
            s_churn=edm_s_churn,
            s_min=edm_s_min,
            s_max=edm_s_max,
            s_noise=edm_s_noise,
        )
        jobs.append((wi, w_samplers[wi], cond, bs, kwargs))
        sample_idx += bs

    try:
        results = _dispatch(jobs, primary_dm, vae, vocoder, primary, leave_as_latent=leave_as_latent)
    finally:
        del jobs
        _cleanup_workers(w_samplers)

    if out_dir is not None and not leave_as_latent:
        root = Path(out_dir)
        for i, (audio, mels) in enumerate(results):
            _save_stems(root / f"sample_{(i + start_from):04d}", audio, mels, vocoder.sample_rate)

    return results


def separate_mixture(
    mixture_audio,
    sr,
    sampler,
    vae,
    vocoder,
    mel_extractor,
    *,
    devices=None,
    ddim_steps=200,
    ddim_discretize="uniform",
    ddim_eta=1.0,
    edm_rho=7.0,
    edm_s_churn=0.0,
    edm_s_min=0.0,
    edm_s_max=float("inf"),
    edm_s_noise=1.0,
    cfg_scale=3.0,
    target_length=1024,
    leave_as_latent=False,
    out_dir: Optional[str] = None,
    start_from: Optional[int] = 0,
) -> List[Tuple]:
    """
    mixture_audio: float/int16 numpy (num_samples,) or List of such arrays.

    Args:
        ddim_discretize : timestep spacing for DDIM/Heun, "uniform" or "quad".
        cfg_scale : guidance strength; 1 = none, 3-5 = recommended for separation.
        target_length : mel frames. Default 1024 = latent_t_size * VAE stride at 16 kHz hop=160.

    Returns List of n_samples (audio, mels):
        audio : int16 numpy  (num_stems, num_samples)
        mels  : float tensor (num_stems, 1, n_mels, T_mel)
    If leave_as_latent is True, returns List of n_samples tensors
        (num_stems, z_channels, latent_t_size, latent_f_size)
    If out_dir is given, each result is saved to out_dir/sample_NNNN/.
    """
    resolved = _resolve_devices(devices)
    primary = resolved[0]
    primary_dm = sampler.get_diffusion_model().to(primary).eval()
    vae.to(primary).eval()
    vocoder.to(primary).eval()

    mixtures = mixture_audio if isinstance(mixture_audio, list) else [mixture_audio]
    n_samples = len(mixtures)

    def _mel(mix):
        m = mel_extractor.audio_to_mel(mix, sr)  # (1, n_mels, T)
        T = m.shape[-1]
        if T < target_length: m = torch.nn.functional.pad(m, (0, target_length - T))
        else: m = m[..., :target_length]
        return m

    cond_cache = {}
    with torch.no_grad():
        for mix in mixtures:
            if id(mix) not in cond_cache:
                mel = _mel(mix).to(primary)
                cond_cache[id(mix)] = primary_dm.get_conditioning({"fbank": mel.unsqueeze(0)})  # (1, S, C, T, F)

    allot = _allot(n_samples, min(len(resolved), n_samples))
    n_workers = max(wi for wi, _ in allot) + 1
    w_samplers = [sampler] + [_build_worker_sampler(sampler, resolved[i]) for i in range(1, n_workers)]

    shape = (primary_dm.num_stems, primary_dm.z_channels, primary_dm.latent_t_size, primary_dm.latent_f_size)

    jobs, cursor = [], 0
    for wi, bs in allot:
        batch_conds = [cond_cache[id(m)] for m in mixtures[cursor: cursor + bs]]
        cond = torch.cat(batch_conds, dim=0).to(resolved[wi])  # (bs, S, C, T, F)
        kwargs = dict(
            steps=ddim_steps,
            ddim_discretize=ddim_discretize,
            eta=ddim_eta,
            cfg=cfg_scale,
            rho=edm_rho,
            s_churn=edm_s_churn,
            s_min=edm_s_min,
            s_max=edm_s_max,
            s_noise=edm_s_noise,
        )
        jobs.append((wi, w_samplers[wi], cond, bs, kwargs))
        cursor += bs

    try:
        results = _dispatch(jobs, primary_dm, vae, vocoder, primary, leave_as_latent=leave_as_latent)
    finally:
        del jobs
        cond_cache.clear()
        del cond_cache
        _cleanup_workers(w_samplers)

    if out_dir is not None and not leave_as_latent:
        root = Path(out_dir)
        for i, (audio, mels) in enumerate(results):
            _save_stems(root / f"sample_{(i + start_from):04d}", audio, mels, vocoder.sample_rate)

    return results


def change_max_batches(start_new_max, new_max):
    global INITIAL_MAX_BATCH_SIZE, MAX_BATCH_SIZE
    INITIAL_MAX_BATCH_SIZE = start_new_max
    MAX_BATCH_SIZE = new_max


def load_saved_stems(path: str):
    """Load generated stems from a previous run, given the path to the sample directory."""
    path = Path(path)

    mels_path = path / "mels.npz"
    if not mels_path.exists():
        raise FileNotFoundError(f"Expected mels.npz file not found: {mels_path}")
    mels_data = np.load(mels_path)
    mels = mels_data["mels"] # (num_stems, 1, n_mels, T_mel)

    res = {}
    for i, name in enumerate(STEM_NAMES):
        wav_path = path / f"{name}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Expected WAV file not found: {wav_path}")
        stem_audio, sr = sf.read(wav_path, dtype="int16")
        res[name] = (stem_audio, mels[i], sr)

    return res
