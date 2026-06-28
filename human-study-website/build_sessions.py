#!/usr/bin/env python3
import json
import random
from collections import defaultdict
from pathlib import Path

MODELS = ["MSDM", "MSLDM", "MSG-LD", "MGE-LDM"]
GROUND_TRUTH_DIR = "ground_truth"
AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".m4a", ".flac")

TARGET_PARTICIPANTS = 20
NUM_SESSIONS = 40
REAL_TRIALS = 6
CATCH_TRIALS = 1
REPEAT_TRIALS = 1
REPEAT_MIN_GAP = 3
PLAYS_PER_CLIP = 3
SEED = 42

FALLBACK_CLIPS_PER_MODEL = 10
FALLBACK_GT_CLIPS = 10

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio"
DATA = ROOT / "data"


def scan(folder):
    if not folder.is_dir():
        return []
    return sorted(f.name for f in folder.iterdir() if f.suffix.lower() in AUDIO_EXTS)


def discover_pool():
    pool, used_fallback = {}, False
    for model in MODELS:
        files = scan(AUDIO / model)
        if not files:
            files = [f"{i:02d}.mp3" for i in range(1, FALLBACK_CLIPS_PER_MODEL + 1)]
            used_fallback = True
        pool[model] = [
            {"clip_id": f"{model}/{Path(f).stem}", "model": model, "src": f"audio/{model}/{f}"}
            for f in files
        ]

    gt_files = scan(AUDIO / GROUND_TRUTH_DIR)
    if not gt_files:
        gt_files = [f"{i:02d}.mp3" for i in range(1, FALLBACK_GT_CLIPS + 1)]
        used_fallback = True
    ground_truth = [
        {"clip_id": f"GT/{Path(f).stem}", "model": "GROUND_TRUTH",
         "src": f"audio/{GROUND_TRUTH_DIR}/{f}"}
        for f in gt_files
    ]
    return pool, ground_truth, used_fallback


def pick_least_seen(candidates, counts, exclude):
    eligible = [c for c in candidates if c["clip_id"] not in exclude]
    if not eligible:
        raise RuntimeError("No eligible clip left, increase clips per model.")
    lowest = min(counts[c["clip_id"]] for c in eligible)
    return random.choice([c for c in eligible if counts[c["clip_id"]] == lowest])


def reshuffled(clips):
    original = [c["clip_id"] for c in clips]
    shuffled = [dict(c) for c in clips]
    for _ in range(10):
        random.shuffle(shuffled)
        if [c["clip_id"] for c in shuffled] != original:
            break
    return shuffled


def ratings_after(sessions, k):
    counts = defaultdict(int)
    for session in sessions[:k]:
        for trial in session["trials"]:
            if trial.get("is_repeat"):
                continue
            for clip in trial["clips"]:
                if clip["model"] != "GROUND_TRUTH":
                    counts[clip["clip_id"]] += 1
    return list(counts.values())


def build():
    random.seed(SEED)
    pool, ground_truth, used_fallback = discover_pool()

    counts = defaultdict(int)
    gt_counts = defaultdict(int)
    sessions = []

    for participant in range(NUM_SESSIONS):
        used = set()
        real_trials = []
        for _ in range(REAL_TRIALS):
            trial_clips = []
            for model in MODELS:
                clip = pick_least_seen(pool[model], counts, used)
                counts[clip["clip_id"]] += 1
                used.add(clip["clip_id"])
                trial_clips.append(dict(clip))
            random.shuffle(trial_clips)
            real_trials.append({"is_catch": False, "is_repeat": False, "clips": trial_clips})

        dropped_model = random.choice(MODELS)
        catch_clips = []
        for model in MODELS:
            if model == dropped_model:
                continue
            clip = pick_least_seen(pool[model], counts, used)
            counts[clip["clip_id"]] += 1
            used.add(clip["clip_id"])
            catch_clips.append(dict(clip))
        gt_clip = pick_least_seen(ground_truth, gt_counts, set())
        gt_counts[gt_clip["clip_id"]] += 1
        catch_clips.append(dict(gt_clip))
        random.shuffle(catch_clips)
        catch_trial = {"is_catch": True, "is_repeat": False, "clips": catch_clips}

        trials = real_trials[:]

        repeat_source = random.choice(real_trials[:REAL_TRIALS - REPEAT_MIN_GAP])
        repeat_trial = {"is_catch": False, "is_repeat": True,
                        "clips": reshuffled(repeat_source["clips"])}
        source_index = trials.index(repeat_source)
        trials.insert(random.randint(source_index + REPEAT_MIN_GAP, len(trials)), repeat_trial)

        trials.insert(random.randint(0, len(trials)), catch_trial)

        for number, trial in enumerate(trials, start=1):
            trial["trial_number"] = number
        repeat_trial["repeat_of"] = repeat_source["trial_number"]

        sessions.append({"session_index": participant, "trials": trials})

    practice_clips = [dict(pool[model][0]) for model in MODELS]

    DATA.mkdir(exist_ok=True)
    (DATA / "sessions.json").write_text(json.dumps({
        "num_sessions": NUM_SESSIONS,
        "target_participants": TARGET_PARTICIPANTS,
        "real_trials": REAL_TRIALS,
        "catch_trials": CATCH_TRIALS,
        "repeat_trials": REPEAT_TRIALS,
        "trials_per_session": REAL_TRIALS + CATCH_TRIALS + REPEAT_TRIALS,
        "plays_per_clip": PLAYS_PER_CLIP,
        "models": MODELS,
        "seed": SEED,
        "practice": {"clips": practice_clips},
        "sessions": sessions,
    }, indent=2))

    prefix_counts = ratings_after(sessions, TARGET_PARTICIPANTS)
    print(f"Wrote {len(sessions)} sessions to data/sessions.json "
          f"(headroom for {NUM_SESSIONS} participants, goal {TARGET_PARTICIPANTS})")
    print(f"Trials per session: {REAL_TRIALS + CATCH_TRIALS + REPEAT_TRIALS} "
          f"({REAL_TRIALS} real, {CATCH_TRIALS} catch, {REPEAT_TRIALS} repeat)")
    print("Clips per model: " + ", ".join(f"{m}={len(pool[m])}" for m in MODELS)
          + f", GT={len(ground_truth)}")
    if prefix_counts:
        target = TARGET_PARTICIPANTS * REAL_TRIALS / len(pool[MODELS[0]])
        print(f"Ratings per clip at {TARGET_PARTICIPANTS} participants: "
              f"min={min(prefix_counts)} max={max(prefix_counts)} target~={target:.1f}")
    if used_fallback:
        print("\n[!] Some audio folders were empty, used placeholder clip names. "
              "Drop real clips into audio/<MODEL>/ and audio/ground_truth/, then re-run.")


if __name__ == "__main__":
    build()
