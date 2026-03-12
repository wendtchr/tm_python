from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    output_base_dir: Path


def get_settings() -> Settings:
    default_base = Path(__file__).resolve().parents[2] / "outputs"
    configured = os.getenv("TM_OUTPUT_BASE_DIR", str(default_base))
    return Settings(output_base_dir=Path(configured).resolve())
