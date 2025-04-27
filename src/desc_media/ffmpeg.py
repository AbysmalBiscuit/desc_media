from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


def extract_keyframes(path: Path, out_dir: Path) -> None:
    """Extracts keyframes from a video."""
    try:
        _ = subprocess.run(
            [
                "ffmpeg",
                "-skip_frame",
                "nokey",
                "-i",
                str(path.resolve()),
                "-vsync",
                "vfr",
                "-frame_pts",
                "true",
                f"{out_dir.resolve()!s}/frame_%06d.png",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        logger.exception("Failed to extract keyframes for: %s", str(path.resolve()))
