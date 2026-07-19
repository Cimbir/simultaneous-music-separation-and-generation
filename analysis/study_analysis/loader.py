from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from study_analysis.config import Config, GROUND_TRUTH

LONG_COLUMNS = [
    "participant_id", "session_index", "trial_number", "is_catch", "is_repeat",
    "repeat_of", "model", "clip_id", "display_position", "realism_rank",
    "coherence_rank", "replays", "duration_s",
]

PARTICIPANT_COLUMNS = [
    "participant_id", "session_index", "user_agent", "started_at", "finished_at",
    "completed", "n_trials", "reached_catch", "catch_passed",
]


@dataclass(frozen=True)
class StudyData:
    long: pd.DataFrame
    participants: pd.DataFrame

    def real_trials(self) -> pd.DataFrame:
        return self.long[~self.long["is_catch"] & ~self.long["is_repeat"]]

    def clean(self) -> "StudyData":
        good = set(self.participants.loc[self.participants["catch_passed"], "participant_id"])
        return StudyData(
            long=self.long[self.long["participant_id"].isin(good)].reset_index(drop=True),
            participants=self.participants[
                self.participants["participant_id"].isin(good)
            ].reset_index(drop=True),
        )


def build_study_data(
    participants_raw: list[dict[str, Any]],
    responses_raw: list[dict[str, Any]],
    config: Config,
) -> StudyData:
    session_meta = _load_session_meta(config.sessions_json)
    session_of = {p["id"]: p.get("session_index") for p in participants_raw}

    rows: list[dict[str, Any]] = []
    for response in responses_raw:
        session_index = session_of.get(response["participant_id"])
        meta = session_meta.get((session_index, response["trial_number"]), {})
        rows.extend(_explode_response(response, session_index, meta))

    long = pd.DataFrame(rows, columns=LONG_COLUMNS)
    return StudyData(long=long, participants=_participant_table(participants_raw, long))


def _explode_response(
    response: dict[str, Any], session_index: int | None, meta: dict[str, Any]
) -> list[dict[str, Any]]:
    realism = _rank_lookup(response.get("musicality"))
    coherence = _rank_lookup(response.get("coherence"))
    replays = response.get("replays") or {}
    duration = _duration_seconds(response.get("trial_started_at"), response.get("submitted_at"))

    return [
        {
            "participant_id": response["participant_id"],
            "session_index": session_index,
            "trial_number": response["trial_number"],
            "is_catch": bool(response.get("is_catch")),
            "is_repeat": bool(meta.get("is_repeat", False)),
            "repeat_of": meta.get("repeat_of"),
            "model": clip["model"],
            "clip_id": clip["clip_id"],
            "display_position": clip.get("position"),
            "realism_rank": realism.get(clip["clip_id"]),
            "coherence_rank": coherence.get(clip["clip_id"]),
            "replays": replays.get(clip["clip_id"]),
            "duration_s": duration,
        }
        for clip in response.get("clips", [])
    ]


def _rank_lookup(best_first_clip_ids: list[str] | None) -> dict[str, int]:
    return {clip_id: i + 1 for i, clip_id in enumerate(best_first_clip_ids or [])}


def _duration_seconds(started: str | None, submitted: str | None) -> float | None:
    if not started or not submitted:
        return None
    return (pd.Timestamp(submitted) - pd.Timestamp(started)).total_seconds()


def _participant_table(
    participants_raw: list[dict[str, Any]], long: pd.DataFrame
) -> pd.DataFrame:
    catch_passed = _catch_pass_flags(long)
    n_trials = long.groupby("participant_id")["trial_number"].nunique()
    return pd.DataFrame(
        [{
            "participant_id": p["id"],
            "session_index": p.get("session_index"),
            "user_agent": p.get("user_agent"),
            "started_at": p.get("started_at"),
            "finished_at": p.get("finished_at"),
            "completed": p.get("finished_at") is not None,
            "n_trials": int(n_trials.get(p["id"], 0)),
            "reached_catch": p["id"] in catch_passed,
            "catch_passed": bool(catch_passed.get(p["id"], False)),
        } for p in participants_raw],
        columns=PARTICIPANT_COLUMNS,
    )


def _catch_pass_flags(long: pd.DataFrame) -> dict[str, bool]:
    catch = long[long["is_catch"] & (long["model"] == GROUND_TRUTH)]
    return {r.participant_id: r.realism_rank == 1
            for r in catch.itertuples() if pd.notna(r.realism_rank)}


def _load_session_meta(sessions_json: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not sessions_json.is_file():
        return {}
    data = json.loads(sessions_json.read_text())
    return {
        (session["session_index"], trial["trial_number"]): {
            "is_repeat": trial.get("is_repeat", False),
            "repeat_of": trial.get("repeat_of"),
        }
        for session in data.get("sessions", [])
        for trial in session["trials"]
    }
