"""Stage 6: Restyle the finished composite as anime.

One image edit call per run, on the composited trail rather than on each cutout, so
the model sees the whole afterimage effect at once and stylises it coherently.
"""

import base64
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import APIError, OpenAI

from projection_sorcery.config import (
    ANIME_DIR,
    ANIME_FILENAME,
    ANIME_IMAGE_SIZE,
    ANIME_MAX_RETRIES,
    ANIME_MODEL,
    ANIME_PROMPT,
    ANIME_RETRY_BACKOFF_S,
    COMPOSITE_DIR,
    COMPOSITE_FILENAME,
    OPENAI_API_KEY_VAR,
)

PNG_SUFFIX = ".png"
UPLOAD_FILENAME = f"composite{PNG_SUFFIX}"


def build_client() -> OpenAI:
    load_dotenv()

    api_key = os.getenv(OPENAI_API_KEY_VAR)
    if not api_key:
        raise ValueError(f"{OPENAI_API_KEY_VAR} is not set - copy .env.example to .env first")

    return OpenAI(api_key=api_key)


# Send the composite off for restyling. Both guards fire before any billable call.
def stylise(image: np.ndarray, client: OpenAI, prompt: str = ANIME_PROMPT) -> np.ndarray:
    if not prompt.strip():
        raise ValueError("ANIME_PROMPT is empty - set it in config.py before running this stage")

    upload = io.BytesIO(_encode_png(image))
    upload.name = UPLOAD_FILENAME

    response = _edit_with_retries(client, upload, prompt)

    return _decode_png(base64.b64decode(response.data[0].b64_json))


# Transient API failures are common enough on image edits to be worth a bounded retry.
def _edit_with_retries(client: OpenAI, upload: io.BytesIO, prompt: str):
    for attempt in range(1, ANIME_MAX_RETRIES + 1):
        upload.seek(0)

        try:
            return client.images.edit(
                model=ANIME_MODEL,
                image=upload,
                prompt=prompt,
                size=ANIME_IMAGE_SIZE,
            )
        except APIError as error:
            if attempt == ANIME_MAX_RETRIES:
                raise

            print(f"Image edit failed (attempt {attempt}/{ANIME_MAX_RETRIES}): {error}")
            time.sleep(ANIME_RETRY_BACKOFF_S * attempt)

    raise AssertionError("unreachable: retry loop always returns or raises")


def _encode_png(image: np.ndarray) -> bytes:
    ret, buffer = cv2.imencode(PNG_SUFFIX, image)
    if not ret:
        raise ValueError("imencode failed for the composite")

    return buffer.tobytes()


def _decode_png(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode the image returned by the API")

    return image


def save_anime(image: np.ndarray, out_dir=ANIME_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ANIME_FILENAME

    if not cv2.imwrite(str(path), image):
        raise ValueError(f"imwrite failed for: {path}")

    return path


def load_composite(in_dir=COMPOSITE_DIR) -> np.ndarray:
    path = in_dir / COMPOSITE_FILENAME

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"no composite at {path} - run composite.py first")

    return image


def main():
    composite = load_composite()
    client = build_client()
    stylised = stylise(composite, client)
    path = save_anime(stylised)
    print(f"Wrote stylised frame to {path}")


if __name__ == "__main__":
    main()
