"""
Batch image captioning / labeling via Azure OpenAI GPT vision API.

Reads images from a local directory (or a text file of URLs), sends each to
GPT for captioning, and writes results as sidecar .txt / aggregated .jsonl.

Usage:
    # Caption all images in a folder
    python caption_images.py --image_dir /path/to/images --output_dir /path/to/output

    # Caption images listed by URL in a text file
    python caption_images.py --url_list urls.txt --output_dir /path/to/output

    # Custom prompt & concurrency
    python caption_images.py --image_dir ./imgs --output_dir ./out \
        --prompt "Describe this game screenshot in detail, including environment, lighting, camera angle, and any characters or objects visible." \
        --max_workers 8 --max_tokens 1024
"""

import argparse
import base64
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import AzureOpenAI

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

DEFAULT_PROMPT = (
    "Describe this image in detail. Include the main subject, background, "
    "colors, lighting, style, and any text visible in the image."
)
DEFAULT_MODEL = "gpt-5.4-2026-03-05"
DEFAULT_ENDPOINT = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
DEFAULT_API_KEY = "6myxZoOuBuOEyasZ82gEwXd2yWVh8O7K"


def build_client(api_key: str, endpoint: str, api_version: str) -> AzureOpenAI:
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def guess_mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    return mapping.get(ext, "image/jpeg")


def mask_secret(secret: str, prefix: int = 6, suffix: int = 6) -> str:
    if len(secret) <= prefix + suffix:
        return secret
    return f"{secret[:prefix]}...{secret[-suffix:]}"


def caption_single_image(
    client: AzureOpenAI,
    model: str,
    prompt: str,
    *,
    image_path: str | None = None,
    image_url: str | None = None,
    detail: str = "auto",
    max_tokens: int = 500,
    max_retries: int = 3,
) -> str:
    """Send one image to GPT and return the caption text."""
    if image_path:
        b64 = encode_image_base64(image_path)
        mime = guess_mime(image_path)
        url_payload = f"data:{mime};base64,{b64}"
    elif image_url:
        url_payload = image_url
    else:
        raise ValueError("Either image_path or image_url must be provided")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": url_payload, "detail": detail},
                },
            ],
        }
    ]

    log_id = str(uuid.uuid4())
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=False,
                extra_headers={"X-TT-LOGID": log_id},
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [RETRY {attempt}/{max_retries}] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                raise


def collect_local_images(image_dir: str) -> list[str]:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        print(f"ERROR: image_dir does not exist: {image_dir}")
        sys.exit(1)
    paths = sorted(
        str(p) for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths


def collect_urls(url_list: str) -> list[str]:
    with open(url_list) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def process_local_image(
    client, model, prompt, image_path, output_dir, detail, max_tokens, max_retries
):
    name = Path(image_path).stem
    txt_path = Path(output_dir) / f"{name}.txt"

    if txt_path.exists():
        return name, None, True

    try:
        caption = caption_single_image(
            client,
            model,
            prompt,
            image_path=image_path,
            detail=detail,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        txt_path.write_text(caption, encoding="utf-8")
        return name, caption, False
    except Exception as e:
        return name, f"ERROR: {e}", False


def process_url_image(
    client, model, prompt, url, output_dir, detail, max_tokens, max_retries, idx
):
    name = f"url_{idx:06d}"
    txt_path = Path(output_dir) / f"{name}.txt"

    if txt_path.exists():
        return name, None, True

    try:
        caption = caption_single_image(
            client,
            model,
            prompt,
            image_url=url,
            detail=detail,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        txt_path.write_text(caption, encoding="utf-8")
        return name, caption, False
    except Exception as e:
        return name, f"ERROR: {e}", False


def main():
    parser = argparse.ArgumentParser(
        description="Batch image captioning via Azure OpenAI GPT vision API"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image_dir", type=str, help="Directory of local images")
    src.add_argument("--url_list", type=str, help="Text file with one image URL per line")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--detail", type=str, default="auto", choices=["low", "high", "auto"])
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
    )
    parser.add_argument("--api_version", type=str, default="2024-03-01-preview")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    env_api_key = os.environ.get("OPENAI_API_KEY")
    api_key = args.api_key or env_api_key or DEFAULT_API_KEY
    api_key_source = (
        "--api_key"
        if args.api_key
        else "OPENAI_API_KEY"
        if env_api_key
        else "script default"
    )

    print(f"Using model: {args.model}")
    print(f"Using endpoint: {args.endpoint}")
    print(f"Using api key source: {api_key_source}")
    print(f"Using api key: {mask_secret(api_key)}")

    client = build_client(api_key, args.endpoint, args.api_version)

    # Collect work items
    if args.image_dir:
        images = collect_local_images(args.image_dir)
        print(f"Found {len(images)} images in {args.image_dir}")
    else:
        urls = collect_urls(args.url_list)
        print(f"Found {len(urls)} URLs in {args.url_list}")

    jsonl_path = Path(args.output_dir) / "captions.jsonl"
    done, skipped, failed = 0, 0, 0

    with (
        ThreadPoolExecutor(max_workers=args.max_workers) as pool,
        open(jsonl_path, "a", encoding="utf-8") as jsonl_f,
    ):
        futures = {}
        if args.image_dir:
            for img in images:
                fut = pool.submit(
                    process_local_image,
                    client, args.model, args.prompt,
                    img, args.output_dir, args.detail,
                    args.max_tokens, args.max_retries,
                )
                futures[fut] = img
        else:
            for idx, url in enumerate(urls):
                fut = pool.submit(
                    process_url_image,
                    client, args.model, args.prompt,
                    url, args.output_dir, args.detail,
                    args.max_tokens, args.max_retries, idx,
                )
                futures[fut] = url

        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            name, caption, was_skipped = fut.result()
            if was_skipped:
                skipped += 1
                print(f"[{i}/{total}] {name} — skipped (already exists)")
                continue
            if caption and caption.startswith("ERROR:"):
                failed += 1
                print(f"[{i}/{total}] {name} — {caption}")
            else:
                done += 1
                source = futures[fut]
                record = {"name": name, "source": source, "caption": caption}
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_f.flush()
                print(f"[{i}/{total}] {name} — OK ({len(caption)} chars)")

    print(f"\nDone: {done}  Skipped: {skipped}  Failed: {failed}")
    print(f"Results: {args.output_dir}")
    print(f"JSONL:   {jsonl_path}")


if __name__ == "__main__":
    main()
