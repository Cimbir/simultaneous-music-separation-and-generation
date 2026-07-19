from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MODELS: tuple[str, ...] = ("MSDM", "MSLDM", "MSG-LD", "MSG-LD-ext", "MGE-LDM")
GROUND_TRUTH = "GROUND_TRUTH"
KEY_PAIR: tuple[str, str] = ("MSG-LD", "MSG-LD-ext")
CLIPS_PER_TRIAL = 5

MODEL_COLORS: dict[str, str] = {
    "MSDM": "#2a78d6",
    "MSLDM": "#008300",
    "MSG-LD": "#eda100",
    "MSG-LD-ext": "#eb6834",
    "MGE-LDM": "#4a3aa7",
    GROUND_TRUTH: "#52514e",
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e6e5e1"
SURFACE = "#fcfcfb"


@dataclass(frozen=True)
class Config:
    supabase_url: str = ""
    supabase_service_key: str = ""
    sessions_json: Path = field(
        default_factory=lambda: _repo_root() / "human-study-website" / "data" / "sessions.json"
    )
    objective_metrics_csv: Path | None = None
    out_dir: Path = field(default_factory=lambda: _repo_root() / "analysis" / "out")
    n_bootstrap: int = 10_000
    seed: int = 42

    @classmethod
    def from_env(cls) -> "Config":
        _load_dotenv(_repo_root() / "analysis" / ".env")
        metrics_csv = os.environ.get("OBJECTIVE_METRICS_CSV")
        return cls(
            supabase_url=os.environ.get("SUPABASE_URL", "").rstrip("/"),
            supabase_service_key=os.environ.get("SUPABASE_SERVICE_KEY", ""),
            objective_metrics_csv=Path(metrics_csv) if metrics_csv else None,
        )

    def require_credentials(self) -> None:
        missing = [n for n, v in (("SUPABASE_URL", self.supabase_url),
                                  ("SUPABASE_SERVICE_KEY", self.supabase_service_key)) if not v]
        if missing:
            raise RuntimeError(
                f"Missing {', '.join(missing)}. Set them in the env or analysis/.env. "
                "Reading needs the service_role key (anon can only insert)."
            )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
