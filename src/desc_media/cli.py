from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import cast

import click
import coloredlogs
import ollama
import verboselogs
from filetype import is_image, is_video
from tqdm import tqdm

from desc_media import ROOT_LOGGER
from desc_media.process import process_image, process_video
from desc_media.utils import (
    find_files,
    find_files_fd,
    get_counter_most_common_keys,
    save_descriptions,
)

__version__ = "0.0.1"


logger = verboselogs.VerboseLogger(__name__)


@click.group(invoke_without_command=True)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (can be used additively)",
)
@click.option(
    "--quiet",
    "-q",
    count=True,
    help="Decrease verbosity (can be used additively)",
)
@click.version_option(version=__version__, prog_name="")
@click.pass_context
def cli(ctx: click.Context, verbose: int, quiet: int) -> None:
    """
    \bDescribe video and image files.
    """  # noqa: D200, D415, D212, D205, D301
    # The verbosity flag handling is partially taken from `hatch`:
    # https://github.com/pypa/hatch/tree/master
    # Configure logging
    verbosity: int = 4 + verbose - quiet
    verbosity_levels: dict[int, int] = {
        0: logging.ERROR,
        1: verboselogs.SUCCESS,
        2: logging.WARNING,
        3: verboselogs.NOTICE,
        4: logging.INFO,
        5: verboselogs.VERBOSE,
        6: logging.DEBUG,
        7: verboselogs.SPAM,
        8: logging.NOTSET,
    }
    coloredlogs.set_level(verbosity_levels.get(verbosity, logging.NOTSET))


@cli.command("describe")
@click.option(
    "extra_prompt",
    "--extra",
    "-e",
    type=str,
    help="Extra text to be added to the prompt.",
)
@click.option(
    "video_batch_size",
    "--video-batch-size",
    "-vbs",
    type=int,
    default=5,
    help="Batch size for video processing.",
)
@click.argument(
    "path",
    type=click.Path(exists=True, path_type=Path),
)
def describe(
    *,
    extra_prompt: str,
    video_batch_size: int,
    path: Path,
) -> None:
    """Describe the given image, video, or all images/videos in a folder."""
    client: ollama.Client = ollama.Client(host="http://127.0.0.1:11434", timeout=5)
    if path.is_file():
        if is_image(path):
            desc = process_image(
                client=client,
                path=path,
                extra_prompt=extra_prompt,
            )
            logger.info("%s: %s", str(path), get_counter_most_common_keys(desc))
        elif is_video(path):
            desc = process_video(
                client=client,
                path=path,
                batch_size=video_batch_size,
                extra_prompt=extra_prompt,
            )
            logger.info("%s: %s", str(path), get_counter_most_common_keys(desc))
    else:
        files: list[Path] = [file for file in path.rglob("*") if file.is_file()]
        image_files: list[Path] = [file for file in files if is_image(file)]
        video_files: list[Path] = [file for file in files if is_video(file)]
        if len(files) == 0:
            logger.info("No files found in: '%s'", path)
            return
        logger.info("Found %s files", len(files))
        logger.info("Found %s image files", len(image_files))
        logger.info("Found %s video files", len(video_files))
        for image in image_files:
            desc = process_image(
                client=client,
                path=image,
                extra_prompt=extra_prompt,
            )
            logger.info("%s: %s", str(image), get_counter_most_common_keys(desc))
        for video in video_files:
            desc = process_video(
                client=client,
                path=video,
                batch_size=video_batch_size,
                extra_prompt=extra_prompt,
            )
            logger.info("%s: %s", str(video), get_counter_most_common_keys(desc))


@cli.command("save")
@click.option(
    "extra_prompt",
    "--extra",
    "-e",
    type=str,
    help="Extra text to be added to the prompt.",
)
@click.option(
    "video_batch_size",
    "--video-batch-size",
    "-vbs",
    type=int,
    default=5,
    help="Batch size for video processing.",
)
@click.option(
    "--timeout",
    "-t",
    type=int,
    default=5,
    help="Batch size for video processing.",
)
@click.option(
    "incremental",
    "--incremental",
    "-i",
    is_flag=True,
    default=False,
    help="Skip files already in descfile",
)
@click.argument(
    "path",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "descfile",
    type=click.Path(path_type=Path),
)
def save(
    *,
    extra_prompt: str,
    video_batch_size: int,
    timeout: int,
    incremental: bool,
    path: Path,
    descfile: Path,
) -> None:
    """Describe the given image, video, or all images/videos in a folder."""
    client: ollama.Client = ollama.Client(host="http://127.0.0.1:11434", timeout=timeout)
    if descfile == Path(".") or descfile.is_dir():
        descfile = descfile / "descmedia.json"

    log_file: Path = descfile.parent / "descmedia.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(verboselogs.VERBOSE)
    file_formatter = logging.Formatter(
        "%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    ROOT_LOGGER.addHandler(file_handler)

    descriptions: dict[str, list[str]] = {}

    if incremental:
        descriptions = cast(dict[str, list[str]], json.loads(descfile.read_text()))

    if path.is_file():
        if is_image(path):
            desc = process_image(
                client=client,
                path=path,
                extra_prompt=extra_prompt,
            )

            logger.debug("%s: %s", str(path), [item[0] for item in desc.most_common()])
            descriptions[str(path.resolve())] = get_counter_most_common_keys(desc)
        elif is_video(path):
            desc = process_video(
                client=client,
                path=path,
                batch_size=video_batch_size,
                extra_prompt=extra_prompt,
            )
            logger.debug("%s: %s", str(path), [item[0] for item in desc.most_common()])
            descriptions[str(path.resolve())] = get_counter_most_common_keys(desc)
    else:
        # files: list[Path] = find_files(path)
        # logger.debug("Found %s files", len(files))
        image_files: list[Path] = find_files_fd(path, filter_type="image")
        video_files: list[Path] = find_files_fd(path, filter_type="video")
        if len(image_files) == 0 and len(video_files) == 0:
            logger.warning("No media files found in: '%s'", path)
            return
        logger.debug("Found %s image files", len(image_files))
        logger.debug("Found %s video files", len(video_files))

        for idx, image in tqdm(
            enumerate(image_files),
            desc="Images",
            total=len(image_files),
            leave=False,
        ):
            if (
                incremental
                and (file_path := str(image.resolve())) in descriptions
                and len(descriptions[file_path]) > 0
            ):
                logger.verbose("Skipping %s", file_path)
                continue
            desc = process_image(
                client=client,
                path=image,
                extra_prompt=extra_prompt,
            )
            most_common = get_counter_most_common_keys(desc)
            logger.debug("%s: %s", str(image), most_common)
            descriptions[str(image.resolve())] = most_common
            if idx % 50 == 0:
                save_descriptions(descfile, descriptions)
        for idx, video in tqdm(
            enumerate(video_files),
            desc="Videos",
            total=len(video_files),
            leave=False,
        ):
            if incremental and str(video.resolve()) in descriptions:
                continue
            desc = process_video(
                client=client,
                path=video,
                batch_size=video_batch_size,
                extra_prompt=extra_prompt,
            )
            most_common = get_counter_most_common_keys(desc)
            logger.debug("%s: %s", str(video), most_common)
            descriptions[str(video.resolve())] = most_common
            if idx % 50 == 0:
                save_descriptions(descfile, descriptions)

    save_descriptions(path=descfile, descriptions=descriptions)


def main() -> int:
    try:
        cli()
    except Exception:
        logger.exception("An unexpected error occurred.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
