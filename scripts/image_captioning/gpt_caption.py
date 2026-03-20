"""
Lightweight GPT vision caption helper.
Used by inference scripts as an alternative to local InternVL3.

Usage:
    from scripts.image_captioning.gpt_caption import build_gpt_client, caption_image_gpt

    client = build_gpt_client()
    response = caption_image_gpt(client, image_path="/path/to/frame.jpg",
                                 question="Describe this image.")
"""

import base64
import time
import uuid
from pathlib import Path

from openai import AzureOpenAI

# ── Default config (matches caption_images.py) ───────────────────────────────
GPT_API_KEY  = "6myxZoOuBuOEyasZ82gEwXd2yWVh8O7K"
GPT_ENDPOINT = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
GPT_MODEL    = "gpt-5.4-2026-03-05" # "gpt-5.2-2025-12-11"
GPT_API_VER  = "2024-03-01-preview"


def build_gpt_client(
    api_key: str = GPT_API_KEY,
    endpoint: str = GPT_ENDPOINT,
    api_version: str = GPT_API_VER,
) -> AzureOpenAI:
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def _encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_str, mime_type) for a local image file."""
    ext = Path(image_path).suffix.lower()
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".gif": "image/gif",  ".bmp":  "image/bmp",
    }.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def caption_image_gpt(
    client: AzureOpenAI,
    *,
    image_path: str | None = None,
    image_url: str | None = None,
    question: str = "Describe this image in detail.",
    model: str = GPT_MODEL,
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> str:
    """
    Call GPT vision API and return the response string.

    Exactly one of `image_path` (local file, encoded as base64) or
    `image_url` (publicly accessible URL) must be provided.
    """
    if image_path:
        b64, mime = _encode_image(image_path)
        url_payload = f"data:{mime};base64,{b64}"
    elif image_url:
        url_payload = image_url
    else:
        raise ValueError("Either image_path or image_url must be provided")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": url_payload, "detail": "auto"}},
            ],
        }
    ]

    log_id = str(uuid.uuid4())
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=False,
                extra_headers={"X-TT-LOGID": log_id},
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [GPT RETRY {attempt}/{max_retries}] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                raise
