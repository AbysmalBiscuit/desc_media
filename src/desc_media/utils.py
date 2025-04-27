from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
import json
import logging
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Literal

import av
import filetype
from PIL import Image
from tqdm import tqdm

IMAGE_EXT = (".jpeg", ".jpg", ".png", ".gif", ".webp", ".tiff", ".web")
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi")

TRANSLATION_DICT = {
    "\n": " ",
    "#": " ",
    # "\\": None,
}

TRANSLATION_TABLE = str.maketrans(TRANSLATION_DICT)

logger = logging.getLogger(__name__)

# Disable printing ffmpeg logs
logging.getLogger("libav").setLevel(50)


def is_media(path: Path) -> bool:
    path = Path(path)
    if path.is_file():
        guessed_type = filetype.guess(path)
        # guessed_type will not be None if the file is valid
        return guessed_type is not None
    return False


def check_video_exists(path: Path, files_dict: dict[str, list[Path]]) -> bool:
    if path.name in files_dict:
        return all(is_media(f) for f in files_dict[path.name])
    return False


def is_video(path: Path) -> bool:
    path = Path(path)
    if path.is_file():
        try:
            with av.open(str(path.absolute())) as container:
                return len(container.streams.video) > 0
        except av.error.InvalidDataError:  # type: ignore
            logger.exception("Failed to open video %s with error:\n", path)
            logger.debug(traceback.format_exc())
    return False


def is_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()  # Verify that it is an image
    except (OSError, SyntaxError) as e:
        logger.debug("The following error occurred while opening image: %s", e)
        return False
    else:
        return True


def is_video_fast(path: Path) -> bool:
    return path.stem in VIDEO_EXT


def is_image_fast(path: Path) -> bool:
    return path.stem in IMAGE_EXT


def post_process_model_result(result: str) -> list[str]:
    result = result.translate(TRANSLATION_TABLE)
    if "," in result:
        result_list = [s.strip() for s in result.split(",")]
    else:
        result_list = [s.strip() for s in result.split(" ")]
    return result_list


def find_files_fd(path: Path, filter_type: Literal["image", "video"] | None = None) -> list[Path]:
    fd = shutil.which("fd") or shutil.which("fdfind") or shutil.which("fd-find")
    if fd is None:
        msg = "fd is not ava.lable on this system."
        raise ValueError(msg)

    logger.info("Using fd to search for files.")
    filter_ = "."
    if filter_type == "image":
        filter_ = r".(" + "|".join(["\\" + ext for ext in IMAGE_EXT]) + ")$"
    elif filter_type == "video":
        filter_ = r".(" + "|".join(["\\" + ext for ext in VIDEO_EXT]) + ")$"

    result = subprocess.run(
        [
            fd,
            "-t",
            "f",
            f"{filter_}",
            str(path.resolve()),
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [Path(f) for f in result.stdout.decode().splitlines()]


def find_files(path: Path, filter_type: Literal["image", "video"] | None = None) -> list[Path]:
    fd = shutil.which("fd") or shutil.which("fdfind") or shutil.which("fd-find")
    if fd is not None:
        logger.info("Using fd to search for files.")
        if filter_type == "image":
            filter_ = "(" + "|".join(IMAGE_EXT) + ")$"
        elif filter_type == "video":
            filter_ = "(" + "|".join(VIDEO_EXT) + ")$"

        result = subprocess.run(
            [
                fd,
                "-t",
                "f",
                ".",
                str(path.resolve()),
            ],
            check=True,
            stdout=subprocess.PIPE,
        )
        return [Path(f) for f in result.stdout.decode().splitlines()]
    logger.info("Using Python to search for files.")
    return [f for f in tqdm(path.rglob("*"), desc="Finding files") if f.is_file()]


def get_counter_most_common_keys[T: Hashable](desc: Counter[T]) -> list[T]:
    return [item[0] for item in desc.most_common()]


def save_descriptions(path: Path, descriptions: dict[str, list[str]]) -> None:
    path.write_text(json.dumps(descriptions, indent=2, sort_keys=False))
