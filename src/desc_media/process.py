from __future__ import annotations

import logging
import shutil
from collections import Counter
from typing import TYPE_CHECKING

import httpx
import ollama
from PIL import Image
from tqdm import tqdm

from desc_media.ffmpeg import extract_keyframes
from desc_media.utils import post_process_model_result

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(50)


def _describe(
    client: ollama.Client,
    messages: list[dict[str, str | list[str]]],
    max_retries: int = 5,
) -> ollama.ChatResponse | None:
    for _ in range(max_retries):
        try:
            response: ollama.ChatResponse = client.chat(model="llava", messages=messages)
        except (ollama.ResponseError, httpx.ReadTimeout):
            pass
        else:
            return response
        logger.debug("Model timed out, trying again.")
    return None


def process_video(
    client: ollama.Client,
    path: Path,
    batch_size: int = 5,
    extra_prompt: str = "",
    max_retries: int = 5,
) -> Counter[str]:
    """Generate keywords to describe a video."""
    if extra_prompt:
        extra_prompt = f"\n{extra_prompt}"

    frames_dir = path.parent / "_temp_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    extract_keyframes(path=path, out_dir=frames_dir)

    frames = sorted([str(f.resolve()) for f in frames_dir.glob("*") if f.is_file()])
    start = 0
    responses: list[str] = []

    slide = len(frames) > batch_size
    if slide:
        batch_size -= 2
    for start in tqdm(range(0, len(frames), batch_size), desc="Batches", leave=False):
        if slide and start > 0:
            selected = frames[start - 2 : start + batch_size]
        else:
            selected = frames[start : start + batch_size]
        response = _describe(
            client=client,
            messages=[
                {
                    "role": "user",
                    "content": f"""\
Analyze the following images and output a list of 10-50 keywords or hashtags that best describe the
visual content.
The images are part of a video, so they should flow_together.{extra_prompt}
Do not format the output as a numbered list.
Do not format the output as a list.
Do not include punctuation other than commas.
Only output the keywords, separated by commas.""",
                    "images": selected,
                },
            ],
            max_retries=max_retries,
        )
        if response is None:
            logger.error("Failed to analyze part of the video: '%s'", selected)
            continue
        logger.debug(
            f"Description for batch {start // batch_size + 1}: {response['message']['content']}"
        )
        responses.append(response["message"]["content"])
    shutil.rmtree(frames_dir)
    return Counter([resp for response in responses for resp in post_process_model_result(response)])


def process_image(
    client: ollama.Client,
    path: Path,
    extra_prompt: str = "",
) -> Counter[str]:
    """Generate keywords to describe an image."""
    if extra_prompt:
        extra_prompt = f"\n{extra_prompt}"

    created_temp: bool = False
    temp_path: Path = path
    if path.stem.casefold() not in (".jpeg", ".jpg", ".png"):
        created_temp = True
        temp_path = path.parent / f"{path.name}_temp.jpg"
        im = Image.open(str(path))
        im = im.convert("RGB")
        im.save(str(temp_path))

    response = _describe(
        client=client,
        messages=[
            {
                "role": "user",
                "content": f"""\
Analyze this image and output a list of 10-50 keywords or hashtags that best describe the visual content.{extra_prompt}
Do not format the output as a numbered list.
Do not format the output as a list.
Do not include punctuation other than commas.
Only output the keywords, separated by commas.""",
                "images": [str(temp_path.resolve())],
            },
        ],
    )
    if created_temp:
        temp_path.unlink()

    if response is None:
        logger.error("Failed to analyze the video: '%s'", str(path))
        return Counter([])

    result = response["message"]["content"]
    logger.debug(f"Description for image {path!s}: {result}")
    return Counter(post_process_model_result(result))
