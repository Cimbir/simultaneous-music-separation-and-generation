
import copy, threading, torch
from collections import defaultdict
from typing import List, Tuple

STEM_NAMES = ["bass", "drums", "guitar", "piano"]
INITIAL_MAX_BATCH_SIZE = 1 # primary GPU. less because also holds the VAE and vocoder
MAX_BATCH_SIZE = 4 # worker GPUs


def _resolve_devices(devices=None):
    if devices is not None:
        return [torch.device(d) for d in devices]
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


def _build_worker_sampler(sampler, device):
    dm = sampler.get_diffusion_model()
    dm = copy.deepcopy(dm).to(device).eval()
    return type(sampler)(dm)


def _run_ddim(
    sampler, 
    cond, 
    *, 
    steps, 
    eta, 
    cfg, 
    bs
):
    dm = sampler.get_diffusion_model()
    shape = (dm.num_stems, dm.z_channels, dm.latent_t_size, dm.latent_f_size)
    return sampler.sample(steps=steps, 
                          batch_size=bs, 
                          shape=shape,
                          conditioning=cond, 
                          eta=eta, 
                          verbose=False, 
                          cfg_scale=cfg)


def _decode(samples, dm, vae, vocoder):
    # (B, S, C, T, F) -> audio (B, S, num_samples), mels (B, S, 1, n_mels, T_mel)
    B, S, C, T, F = samples.shape
    z = (samples / dm.scale_factor).reshape(B * S, C, T, F)
    m = vae.decode(z)
    w = vocoder.mel_to_audio(m.squeeze(1))
    return w.reshape(B, S, -1), m.reshape(B, S, *m.shape[1:])


def _allot(n_samples, n_workers):
    base, extra = divmod(n_samples, n_workers)
    jobs = []
    for wi in range(n_workers):
        count = base + (1 if wi < extra else 0)
        bs = INITIAL_MAX_BATCH_SIZE if wi == 0 else MAX_BATCH_SIZE
        for s in range(0, count, bs):
            jobs.append((wi, min(bs, count - s)))
    return jobs # [(worker_idx, batch_size)]


def _dispatch(jobs, primary_dm, vae, vocoder, primary_dev):
    raw = [None] * len(jobs)
    errors = []
    groups = defaultdict(list)
    for i, (wi, *_) in enumerate(jobs):
        groups[wi].append(i)

    def run(indices):
        try:
            for i in indices:
                wi, samp, cond, bs, steps, eta, cfg = jobs[i]
                with torch.no_grad():
                    raw[i] = _run_ddim(samp, cond, steps=steps, eta=eta, cfg=cfg, bs=bs).cpu()
        except Exception as e:
            errors.append(e)

    if len(groups) == 1:
        run(groups[0])
    else:
        ts = [threading.Thread(target=run, args=(groups[wi],), daemon=True) for wi in groups]
        for t in ts: t.start()
        for t in ts: t.join()

    if errors:
        raise errors[0]

    out = []
    for s in raw:
        with torch.no_grad():
            audio, mels = _decode(s.to(primary_dev), primary_dm, vae, vocoder)
        for j in range(audio.shape[0]):
            out.append((audio[j], mels[j].cpu()))
    return out


def generate_stems(
    sampler, 
    vae, 
    vocoder, 
    *, 
    n_samples=1,
    devices=None,
    ddim_steps=200, 
    ddim_eta=1.0
) -> List[Tuple]:
    """
    Returns List of n_samples (audio, mels):
        audio : int16 numpy (num_stems, num_samples)
        mels : float tensor (num_stems, 1, n_mels, T_mel)
    """
    resolved = _resolve_devices(devices)
    primary = resolved[0]
    primary_dm = sampler.get_diffusion_model().to(primary).eval()
    vae.to(primary).eval()
    vocoder.to(primary).eval()

    allot = _allot(n_samples, min(len(resolved), n_samples))
    n_workers = max(wi for wi, _ in allot) + 1
    w_samplers = [sampler] + [_build_worker_sampler(sampler, resolved[i]) for i in range(1, n_workers)]

    jobs = []
    for wi, bs in allot:
        with torch.no_grad():
            cond = w_samplers[wi].get_diffusion_model().cond_stage_model.get_unconditional_condition(bs)
        jobs.append((wi, w_samplers[wi], cond, bs, ddim_steps, ddim_eta, 1.0))

    return _dispatch(jobs, primary_dm, vae, vocoder, primary)


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
    ddim_eta=1.0,
    cfg_scale=3.0, 
    target_length=1024
) -> List[Tuple]:
    """
    mixture_audio: int16 numpy (num_samples,) or List of int16 numpy (num_samples,)
    Returns List of n_samples (audio, mels):
        audio : int16 numpy  (num_stems, num_samples)
        mels  : float tensor (num_stems, 1, n_mels, T_mel)
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

    jobs, cursor = [], 0
    for wi, bs in allot:
        batch_conds = [cond_cache[id(m)] for m in mixtures[cursor: cursor + bs]]
        cond = torch.cat(batch_conds, dim=0).to(resolved[wi])  # (bs, S, C, T, F)
        jobs.append((wi, w_samplers[wi], cond, bs, ddim_steps, ddim_eta, cfg_scale))
        cursor += bs

    return _dispatch(jobs, primary_dm, vae, vocoder, primary)

def change_max_batches(start_new_max, new_max):
    global INITIAL_MAX_BATCH_SIZE, MAX_BATCH_SIZE
    INITIAL_MAX_BATCH_SIZE = start_new_max
    MAX_BATCH_SIZE = new_max