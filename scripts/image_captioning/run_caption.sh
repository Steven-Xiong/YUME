#!/usr/bin/bash
set -euo pipefail

# ── Image captioning with GPT-5.4-pro vision API ──
# Office network endpoint: https://aidp-i18ntt-sg.tiktok-row.net
# Prod/datacenter endpoint: https://aidp-i18ntt-sg.byteintl.net

IMAGE_DIR="${1:?Usage: $0 <image_dir> [output_dir]}"
OUTPUT_DIR="${2:-${IMAGE_DIR}/captions}"
API_KEY="${OPENAI_API_KEY:-6myxZoOuBuOEyasZ82gEwXd2yWVh8O7K}"

echo "Using endpoint: https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
echo "Using model: gpt-5.2-2025-12-11"
echo "Using api key prefix: ${API_KEY:0:6}...${API_KEY: -6}"

python "$(dirname "$0")/caption_images.py" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --prompt "Describe this image in detail. Include the main subject, background, colors, lighting, style, and any text visible in the image." \
    --model "gpt-5.4-2026-03-05" \
    --detail "auto" \
    --max_tokens 500 \
    --max_workers 4 \
    --max_retries 3 \
    --endpoint "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl" \
    --api_key "$API_KEY"
