from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import coloredlogs
import ollama
from filetype import is_image, is_video
from tqdm import tqdm

from desc_media.process import process_image, process_video
from desc_media.utils import find_files, find_files_fd

__version__ = "0.0.1"

coloredlogs.install()
logger: logging.Logger = logging.getLogger(__name__)


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
    verbosity = 2 + verbose - quiet
    verbosity_levels = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
        4: logging.NOTSET,
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
            logger.info("%s: %s", str(path), [item[0] for item in desc.most_common()])
        elif is_video(path):
            desc = process_video(
                client=client,
                path=path,
                batch_size=video_batch_size,
                extra_prompt=extra_prompt,
            )
            logger.info("%s: %s", str(path), [item[0] for item in desc.most_common()])
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
            logger.info("%s: %s", str(image), [item[0] for item in desc.most_common()])
        for video in video_files:
            desc = process_video(
                client=client,
                path=video,
                batch_size=video_batch_size,
                extra_prompt=extra_prompt,
            )
            logger.info("%s: %s", str(video), [item[0] for item in desc.most_common()])


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
    incremental: bool,
    path: Path,
    descfile: Path,
) -> None:
    """Describe the given image, video, or all images/videos in a folder."""
    client: ollama.Client = ollama.Client(host="http://127.0.0.1:11434", timeout=5)
    if descfile == Path(".") or descfile.is_dir():
        descfile = descfile / "descmedia.json"
    descriptions: dict[str, list[tuple[str, int]]] = {}

    if incremental:
        descriptions = json.loads(descfile.read_text())

    if path.is_file():
        if is_image(path):
            desc = process_image(
                client=client,
                path=path,
                extra_prompt=extra_prompt,
            )
            logger.debug("%s: %s", str(path), [item[0] for item in desc.most_common()])
            descriptions[str(path.resolve())] = desc.most_common()
        elif is_video(path):
            desc = process_video(
                client=client,
                path=path,
                batch_size=video_batch_size,
                extra_prompt=extra_prompt,
            )
            logger.debug("%s: %s", str(path), [item[0] for item in desc.most_common()])
            descriptions[str(path.resolve())] = desc.most_common()
    else:
        # files: list[Path] = find_files(path)
        # logger.debug("Found %s files", len(files))
        image_files: list[Path] = find_files_fd(path, filter_type="image")
        video_files: list[Path] = find_files_fd(path, filter_type="video")
        if len(image_files) == 0 and len(video_files) == 0:
            logger.debug("No files found in: '%s'", path)
            return
        logger.debug("Found %s image files", len(image_files))
        logger.debug("Found %s video files", len(video_files))

        for idx, image in tqdm(
            enumerate(image_files),
            desc="Images",
            total=len(image_files),
            leave=False,
        ):
            if incremental and str(image.resolve()) in descriptions:
                continue
            desc = process_image(
                client=client,
                path=image,
                extra_prompt=extra_prompt,
            )
            logger.debug("%s: %s", str(image), [item[0] for item in desc.most_common()])
            descriptions[str(image.resolve())] = desc.most_common()
            if idx % 50 == 0:
                descfile.write_text(json.dumps(descriptions, indent=4, sort_keys=False))
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
            logger.debug("%s: %s", str(video), [item[0] for item in desc.most_common()])
            descriptions[str(video.resolve())] = desc.most_common()
            if idx % 50 == 0:
                descfile.write_text(json.dumps(descriptions, indent=4, sort_keys=False))

    descfile.write_text(json.dumps(descriptions, indent=4, sort_keys=False))


def main() -> int:
    try:
        cli()
    except Exception:
        logger.exception("An unexpected error occurred.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
