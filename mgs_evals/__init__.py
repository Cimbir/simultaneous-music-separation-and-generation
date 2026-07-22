"""
mgs_evals  evaluation metrics for simultaneous music generation & separation

Metric groups
---
Separation (require paired audio/mel):

    SISDR()                     SI-SDR on waveforms (dB, higher better)
                                Do NOT use after a vocoder
    MelMSE()                    MSE on mel-spectrograms (lower better)
                                Preferred for LDM + HiFi-GAN pipelines

    papers:                     https://www.jonathanleroux.org/pdf/LeRoux2019ICASSP05sdr.pdf

Generation (distribution-level, no pairing needed):

    FAD(backend="vggish")       Frechet Audio Distance, VGGish embeddings
    FAD(backend="clap")         Frechet Audio Distance, CLAP embeddings (CLAP-FAD)
    KAD(backend="vggish")       Kernel Audio Distance - preferred over FAD
    KAD(backend="clap")         KAD with CLAP embeddings
    CLAPScore()                 Audio-text cosine similarity in CLAP space

    papers:                     https://arxiv.org/abs/1812.08466
                                https://arxiv.org/abs/2311.01616
                                https://arxiv.org/abs/2502.15602
                                https://arxiv.org/abs/2211.06687

Beat / rhythm (SyncTrack paper, folder-based, require madmom):

    IRS()                       Intra-Rhythmic Stability. beat-interval std per stem (lower better)
    CBS()                       Cross-track Beat Sync. fraction of stems beating together (higher better)
    CBD()                       Cross-track Beat Dispersion. normalized cross-stem beat error (lower better)

    paper:                      https://arxiv.org/abs/2603.01101

Coherence (stem-to-accompaniment, require contrastive_model / beat_this):

    COCOLA(checkpoint_path="path/to/ckpt.ckpt")
                                HPSS contrastive harmonic+rhythmic coherence (higher better)
                                compare pred score vs cocola_random baseline
    BeatAlignment()             beat_this + madmom F-measure (higher better)

Dependencies
---
Core (always required):
    numpy, scipy, torchmetrics, librosa, torch, torchaudio, madmom

Backend-specific (installed separately):
    VGGish  :  pip install torchvggish
    CLAP    :  pip install laion-clap
               OR call utils.deps.add_msg_ld_to_path() for the local copy

Usage example
---

    import numpy as np
    from mgs_evals import SISDR, MelMSE, FAD, KAD, IRS, CBS, CBD, EvalSuite

    # Separation
    si  = SISDR().compute(estimates=est_wav, references=ref_wav)
    mse = MelMSE().compute(estimates_mel=est_mel, references_mel=ref_mel)

    # Generation
    fad = FAD(backend="vggish").compute(generated=gen_list, reference=ref_list, sr=16_000)
    kad = KAD(backend="vggish").compute(generated=gen_list, reference=ref_list)

    # Beat metrics (folder-based, stem_0 / stem_1 / stem_2 / stem_3 subfolders)
    irs = IRS().compute(folder="/path/to/stems/")
    cbs = CBS().compute(folder="/path/to/stems/")
    cbd = CBD().compute(folder="/path/to/stems/")

    # Suite (pass all kwargs; each metric only uses what it needs)
    suite = EvalSuite([SISDR(), MelMSE(), FAD(backend="vggish")])
    results = suite.run(
        estimates=est_wav,       # -> SISDR
        references=ref_wav,      # -> SISDR
        estimates_mel=est_mel,   # -> MelMSE
        references_mel=ref_mel,  # -> MelMSE
        generated=gen_list,      # -> FAD
        reference=ref_list,      # -> FAD
        sr=16_000,
    )
"""

from .base import Metric
from .suite import EvalSuite

try:
    from .separation import SISDR, MelMSE
    _separation_available = True
except (ImportError, OSError):
    _separation_available = False

try:
    from .generation import FAD, KAD, CLAPScore, AudioEmbedder, VGGishEmbedder, CLAPEmbedder, get_embedder
    _generation_available = True
except (ImportError, OSError):
    _generation_available = False

from .harmonic import HC, TC, harmonic_compatibility, tonal_coherence, tc_short_clip

try:
    from .beats import IRS, CBS, CBD
    _beats_available = True
except ImportError:
    _beats_available = False

try:
    from .coherence import COCOLA, BeatAlignment
    _coherence_available = True
except (ImportError, OSError):
    _coherence_available = False

__all__ = [
    # base
    "Metric",
    "EvalSuite",
    # separation
    "SISDR",
    "MelMSE",
    # generation
    "FAD",
    "KAD",
    "CLAPScore",
    # harmonic (custom)
    "HC",
    "TC",
    "harmonic_compatibility",
    "tonal_coherence",
    "tc_short_clip",
    # beats
    "IRS",
    "CBS",
    "CBD",
    # coherence
    "COCOLA",
    "BeatAlignment",
    # embedding backends (for advanced usage)
    "AudioEmbedder",
    "VGGishEmbedder",
    "CLAPEmbedder",
    "get_embedder",
]
