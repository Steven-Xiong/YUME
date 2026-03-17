#!/usr/bin/env python3
"""
Batch inference launcher for YUME1.5 (sample_5b.py).

It keeps the same core arguments as infer.sh, but runs multiple (image, prompt)
items from one manifest file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List


SUPPORTED_INPUT_SUFFIXES = {".jsonl", ".csv", ".tsv"}


@dataclass
class BatchItem:
    image: Path
    prompt: str
    name: str | None = None
    seed: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone batch inference launcher for YUME1.5"
    )

    parser.add_argument("--input_file", type=str, required=True, help="jsonl/csv/tsv manifest file")
    parser.add_argument("--sample_script", type=str, default="fastvideo/sample/sample_5b.py")
    parser.add_argument("--video_output_dir", type=str, default="./outputs/yume15_batch")
    parser.add_argument("--caption_path", type=str, default="./caption_self.txt")
    parser.add_argument("--test_data_dir", type=str, default="./val")

    # Keep defaults aligned with current infer.sh as much as possible.
    parser.add_argument("--nproc_per_node", type=int, default=8)
    parser.add_argument("--master_port", type=int, default=9600)
    parser.add_argument("--seed", type=int, default=43, help="Default seed if item seed is not provided")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_sample_steps", type=int, default=600000)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_euler_timesteps", type=int, default=8)
    parser.add_argument("--rand_num_img", type=float, default=0.6)

    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable --gradient_checkpointing (enabled by default)",
    )
    parser.add_argument(
        "--disable_tf32",
        action="store_true",
        help="Disable --allow_tf32 (enabled by default)",
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands, do not execute")

    return parser.parse_args()


def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:80] if name else "item"


def load_jsonl(path: Path) -> List[BatchItem]:
    items: List[BatchItem] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i}: {e}") from e

            if "image" not in obj or "prompt" not in obj:
                raise ValueError(f"Line {i} must contain 'image' and 'prompt'")

            seed = obj.get("seed", None)
            if seed is not None:
                seed = int(seed)

            name = obj.get("name", None)
            items.append(
                BatchItem(
                    image=Path(str(obj["image"])),
                    prompt=str(obj["prompt"]),
                    name=str(name) if name is not None else None,
                    seed=seed,
                )
            )
    return items


def load_csv_or_tsv(path: Path) -> List[BatchItem]:
    items: List[BatchItem] = []
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        required = {"image", "prompt"}
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            raise ValueError(
                f"{path.name} must have header columns: image,prompt "
                f"(optional: name,seed). Found: {sorted(fields)}"
            )

        for row_idx, row in enumerate(reader, start=2):
            image = (row.get("image") or "").strip()
            prompt = row.get("prompt") or ""
            if not image:
                raise ValueError(f"{path.name}:{row_idx} missing image")
            if not prompt:
                raise ValueError(f"{path.name}:{row_idx} missing prompt")

            seed_raw = (row.get("seed") or "").strip()
            seed = int(seed_raw) if seed_raw else None
            name = (row.get("name") or "").strip() or None

            items.append(BatchItem(image=Path(image), prompt=prompt, name=name, seed=seed))
    return items


def load_items(input_path: Path) -> List[BatchItem]:
    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(
            f"Unsupported input file type: {suffix}. Use one of: {sorted(SUPPORTED_INPUT_SUFFIXES)}"
        )
    if suffix == ".jsonl":
        return load_jsonl(input_path)
    return load_csv_or_tsv(input_path)


def ensure_item_image_exists(item: BatchItem, base_dir: Path) -> Path:
    image_path = item.image if item.image.is_absolute() else (base_dir / item.image)
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path


def build_command(
    args: argparse.Namespace,
    image_dir: Path,
    output_dir: Path,
    prompt: str,
    seed: int,
) -> List[str]:
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--master_port",
        str(args.master_port),
        args.sample_script,
        "--seed",
        str(seed),
        "--train_batch_size",
        str(args.train_batch_size),
        "--max_sample_steps",
        str(args.max_sample_steps),
        "--mixed_precision",
        args.mixed_precision,
        "--video_output_dir",
        str(output_dir),
        "--caption_path",
        args.caption_path,
        "--test_data_dir",
        args.test_data_dir,
        "--num_euler_timesteps",
        str(args.num_euler_timesteps),
        "--rand_num_img",
        str(args.rand_num_img),
        "--jpg_dir",
        str(image_dir),
        "--prompt",
        prompt,
    ]

    if not args.no_gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if not args.disable_tf32:
        cmd.append("--allow_tf32")
    return cmd


def create_single_image_dir(image_path: Path) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="yume15_batch_"))
    dst = tmp_dir / image_path.name
    try:
        os.symlink(image_path, dst)
    except OSError:
        shutil.copy2(image_path, dst)
    return tmp_dir


def run_batch(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    input_path = Path(args.input_file).resolve()
    output_root = Path(args.video_output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    items = load_items(input_path)
    if not items:
        print(f"[ERROR] No items found in {input_path}")
        return 1

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    total = len(items)
    failures: List[str] = []

    for idx, item in enumerate(items, start=1):
        try:
            image_path = ensure_item_image_exists(item, input_path.parent)
            run_name = item.name or f"{idx:04d}_{image_path.stem}"
            run_name = sanitize_name(run_name)
            run_output_dir = output_root / run_name
            run_output_dir.mkdir(parents=True, exist_ok=True)
            seed = item.seed if item.seed is not None else args.seed

            image_dir = create_single_image_dir(image_path)
            try:
                cmd = build_command(
                    args=args,
                    image_dir=image_dir,
                    output_dir=run_output_dir,
                    prompt=item.prompt,
                    seed=seed,
                )
                print(f"\n[{idx}/{total}] Running: {run_name}")
                print("Command:", " ".join(cmd))

                if args.dry_run:
                    continue

                result = subprocess.run(cmd, env=env, check=False)
                if result.returncode != 0:
                    msg = f"{run_name} failed with exit code {result.returncode}"
                    failures.append(msg)
                    print(f"[ERROR] {msg}")
                    if not args.continue_on_error:
                        return result.returncode
            finally:
                shutil.rmtree(image_dir, ignore_errors=True)

        except Exception as e:  # pylint: disable=broad-except
            msg = f"item {idx} failed: {e}"
            failures.append(msg)
            print(f"[ERROR] {msg}")
            if not args.continue_on_error:
                return 1

    if failures:
        print("\nBatch finished with failures:")
        for f in failures:
            print("-", f)
        return 1

    print(f"\nBatch finished successfully. Outputs: {output_root}")
    return 0


def main() -> None:
    args = parse_args()
    rc = run_batch(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
