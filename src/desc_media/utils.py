from __future__ import annotations

import logging
import traceback
from pathlib import Path

import av
import filetype
from PIL import Image

IMG_EXT = {".jpeg", ".jpg", ".png", ".gif", ".webp", ".tiff", ".web"}
VIDEO_EXT = {".mp4", ".mov"}

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


def post_process_model_result(result: str) -> list[str]:
    result = result.translate(TRANSLATION_TABLE)
    if "," in result:
        result_list = [s.strip() for s in result.split(",")]
    else:
        result_list = [s.strip() for s in result.split(" ")]
    return result_list
