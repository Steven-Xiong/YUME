#!/usr/bin/env bash

# Batch inference launcher based on infer.sh
# Supports both:
#   - i2v: image-to-video
#   - t2v: text-to-video
#
# Input formats:
#   i2v TSV: image<TAB>prompt (optional header)
#   t2v: one prompt per line (optional header "prompt"). Lines starting with '#' are ignored.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TOKENIZERS_PARALLELISM=false
# Workaround for protobuf 4+/5+ vs databus/wandb generated _pb2.py (Descriptors cannot be created directly)
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

MODE=""
INPUT_FILE=""
SAMPLE_SCRIPT="fastvideo/sample/sample_5b.py"
NPROC_PER_NODE=8
MASTER_PORT=9600
SEED=43
TRAIN_BATCH_SIZE=1
MAX_SAMPLE_STEPS=600000
MIXED_PRECISION="bf16"
VIDEO_OUTPUT_DIR="./outputs/batch"
CAPTION_PATH=""
TEST_DATA_DIR="./val"
NUM_EULER_TIMESTEPS=""
RAND_NUM_IMG=0.6

USE_GRADIENT_CHECKPOINTING=1
USE_TF32=1
CONTINUE_ON_ERROR=0
DRY_RUN=0

print_usage() {
  cat <<'EOF'
Usage:
  bash infer_batch.sh --mode i2v|t2v --input_file <path.tsv> [options]

Required:
  --mode                      i2v or t2v
  --input_file                TSV file for batch inputs

Options (defaults are based on infer.sh):
  --sample_script             fastvideo/sample/sample_5b.py
  --nproc_per_node            8
  --master_port               9600
  --seed                      43 (can be overridden per item)
  --train_batch_size          1
  --max_sample_steps          600000
  --mixed_precision           bf16
  --video_output_dir          ./outputs/batch
  --caption_path              auto: ./caption_self.txt(i2v), ./caption_re.txt(t2v)
  --test_data_dir             ./val
  --num_euler_timesteps       auto: 8(i2v), 4(t2v)
  --rand_num_img              0.6
  --no_gradient_checkpointing disable --gradient_checkpointing
  --disable_tf32              disable --allow_tf32
  --continue_on_error         keep running after one item fails
  --dry_run                   print commands only
  -h, --help                  show this help

Input format:
  i2v TSV: image<TAB>prompt
  t2v TSV: prompt (one prompt per line, optional header "prompt")

Examples:
  bash infer_batch.sh --mode i2v --input_file batch_inputs.example.tsv
  bash infer_batch.sh --mode t2v --input_file batch_t2v.example.tsv --video_output_dir ./outputs/t2v_batch
EOF
}

sanitize_name() {
  local raw="$1"
  raw="$(echo "$raw" | tr -cs '[:alnum:]_.-' '_' | sed 's/^_//;s/_$//')"
  if [[ -z "$raw" ]]; then
    raw="item"
  fi
  echo "${raw:0:80}"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mode) MODE="${2:-}"; shift 2 ;;
      --input_file) INPUT_FILE="${2:-}"; shift 2 ;;
      --sample_script) SAMPLE_SCRIPT="${2:-}"; shift 2 ;;
      --nproc_per_node) NPROC_PER_NODE="${2:-}"; shift 2 ;;
      --master_port) MASTER_PORT="${2:-}"; shift 2 ;;
      --seed) SEED="${2:-}"; shift 2 ;;
      --train_batch_size) TRAIN_BATCH_SIZE="${2:-}"; shift 2 ;;
      --max_sample_steps) MAX_SAMPLE_STEPS="${2:-}"; shift 2 ;;
      --mixed_precision) MIXED_PRECISION="${2:-}"; shift 2 ;;
      --video_output_dir) VIDEO_OUTPUT_DIR="${2:-}"; shift 2 ;;
      --caption_path) CAPTION_PATH="${2:-}"; shift 2 ;;
      --test_data_dir) TEST_DATA_DIR="${2:-}"; shift 2 ;;
      --num_euler_timesteps) NUM_EULER_TIMESTEPS="${2:-}"; shift 2 ;;
      --rand_num_img) RAND_NUM_IMG="${2:-}"; shift 2 ;;
      --no_gradient_checkpointing) USE_GRADIENT_CHECKPOINTING=0; shift ;;
      --disable_tf32) USE_TF32=0; shift ;;
      --continue_on_error) CONTINUE_ON_ERROR=1; shift ;;
      --dry_run) DRY_RUN=1; shift ;;
      -h|--help) print_usage; exit 0 ;;
      *)
        echo "[ERROR] Unknown arg: $1"
        print_usage
        exit 1
        ;;
    esac
  done
}

ensure_defaults_by_mode() {
  if [[ "$MODE" == "i2v" ]]; then
    if [[ -z "$CAPTION_PATH" ]]; then CAPTION_PATH="./caption_self.txt"; fi
    if [[ -z "$NUM_EULER_TIMESTEPS" ]]; then NUM_EULER_TIMESTEPS=8; fi
  elif [[ "$MODE" == "t2v" ]]; then
    if [[ -z "$CAPTION_PATH" ]]; then CAPTION_PATH="./caption_re.txt"; fi
    if [[ -z "$NUM_EULER_TIMESTEPS" ]]; then NUM_EULER_TIMESTEPS=4; fi
  else
    echo "[ERROR] --mode must be i2v or t2v"
    exit 1
  fi
}

validate_args() {
  if [[ -z "$INPUT_FILE" ]]; then
    echo "[ERROR] --input_file is required"
    exit 1
  fi
  if [[ ! -f "$INPUT_FILE" ]]; then
    echo "[ERROR] input file not found: $INPUT_FILE"
    exit 1
  fi
  if [[ ! -f "$SAMPLE_SCRIPT" ]]; then
    echo "[ERROR] sample script not found: $SAMPLE_SCRIPT"
    exit 1
  fi
  mkdir -p "$VIDEO_OUTPUT_DIR"
}

run_one() {
  local idx="$1"
  local image_path="$2"
  local prompt="$3"
  local run_name="$4"
  local item_seed="$5"
  local run_output_dir="$VIDEO_OUTPUT_DIR/$run_name"
  mkdir -p "$run_output_dir"

  local -a cmd=(
    torchrun
    --nproc_per_node "$NPROC_PER_NODE"
    --master_port "$MASTER_PORT"
    "$SAMPLE_SCRIPT"
    --seed "$item_seed"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --max_sample_steps "$MAX_SAMPLE_STEPS"
    --mixed_precision "$MIXED_PRECISION"
    --video_output_dir "$run_output_dir"
    --caption_path "$CAPTION_PATH"
    --test_data_dir "$TEST_DATA_DIR"
    --num_euler_timesteps "$NUM_EULER_TIMESTEPS"
    --rand_num_img "$RAND_NUM_IMG"
    --prompt "$prompt"
  )

  if [[ "$USE_GRADIENT_CHECKPOINTING" -eq 1 ]]; then
    cmd+=(--gradient_checkpointing)
  fi
  if [[ "$USE_TF32" -eq 1 ]]; then
    cmd+=(--allow_tf32)
  fi

  local tmp_img_dir=""
  if [[ "$MODE" == "i2v" ]]; then
    tmp_img_dir="$(mktemp -d -t yume_i2v_batch_XXXXXX)"
    local image_abs=""
    if [[ "$image_path" = /* ]]; then
      image_abs="$image_path"
    else
      image_abs="$(cd "$(dirname "$INPUT_FILE")" && pwd)/$image_path"
    fi
    if [[ ! -f "$image_abs" ]]; then
      echo "[ERROR] image not found for item $idx: $image_abs"
      rm -rf "$tmp_img_dir"
      return 1
    fi
    ln -s "$image_abs" "$tmp_img_dir/$(basename "$image_abs")" 2>/dev/null || cp "$image_abs" "$tmp_img_dir/"
    cmd+=(--jpg_dir "$tmp_img_dir")
  else
    cmd+=(--T2V)
  fi

  echo ""
  echo "[$idx] mode=$MODE name=$run_name"
  echo "Command: ${cmd[*]}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${cmd[@]}"
    local rc=$?
    if [[ -n "$tmp_img_dir" ]]; then rm -rf "$tmp_img_dir"; fi
    return $rc
  fi

  if [[ -n "$tmp_img_dir" ]]; then rm -rf "$tmp_img_dir"; fi
  return 0
}

run_batch_i2v() {
  local idx=0
  local failed=0

  while IFS=$'\t' read -r image prompt name seed _rest; do
    [[ -z "${image}${prompt}${name}${seed}" ]] && continue
    image="${image%$'\r'}"
    prompt="${prompt%$'\r'}"
    name="${name%$'\r'}"
    seed="${seed%$'\r'}"

    [[ -z "$image" ]] && continue
    [[ "${image:0:1}" == "#" ]] && continue
    if [[ "$image" == "image" && "$prompt" == "prompt" ]]; then
      continue
    fi
    if [[ -z "$prompt" ]]; then
      echo "[ERROR] i2v item missing prompt, image=$image"
      failed=1
      [[ "$CONTINUE_ON_ERROR" -eq 1 ]] || return 1
      continue
    fi

    idx=$((idx + 1))
    local item_seed="$SEED"
    if [[ -n "$seed" ]]; then item_seed="$seed"; fi

    local run_name="$name"
    if [[ -z "$run_name" ]]; then
      run_name="$(basename "${image%.*}")"
      run_name="$(sanitize_name "${idx}_$run_name")"
    else
      run_name="$(sanitize_name "$run_name")"
    fi

    run_one "$idx" "$image" "$prompt" "$run_name" "$item_seed"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
      failed=1
      echo "[ERROR] item $idx failed with code $rc"
      [[ "$CONTINUE_ON_ERROR" -eq 1 ]] || return $rc
    fi
  done < "$INPUT_FILE"

  if [[ "$idx" -eq 0 ]]; then
    echo "[ERROR] no valid i2v rows found in $INPUT_FILE"
    return 1
  fi
  [[ "$failed" -eq 0 ]]
}

run_batch_t2v() {
  local idx=0
  local failed=0

  while IFS= read -r prompt; do
    prompt="${prompt%$'\r'}"
    [[ -z "$prompt" ]] && continue
    [[ "${prompt:0:1}" == "#" ]] && continue
    if [[ "$prompt" == "prompt" ]]; then
      continue
    fi

    idx=$((idx + 1))
    local run_name="$(sanitize_name "${idx}_t2v")"
    run_one "$idx" "" "$prompt" "$run_name" "$SEED"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
      failed=1
      echo "[ERROR] item $idx failed with code $rc"
      [[ "$CONTINUE_ON_ERROR" -eq 1 ]] || return $rc
    fi
  done < "$INPUT_FILE"

  if [[ "$idx" -eq 0 ]]; then
    echo "[ERROR] no valid t2v rows found in $INPUT_FILE"
    return 1
  fi
  [[ "$failed" -eq 0 ]]
}

main() {
  parse_args "$@"
  ensure_defaults_by_mode
  validate_args

  echo "Batch start: mode=$MODE input=$INPUT_FILE output=$VIDEO_OUTPUT_DIR"
  if [[ "$MODE" == "i2v" ]]; then
    run_batch_i2v
  else
    run_batch_t2v
  fi
  local rc=$?

  if [[ $rc -eq 0 ]]; then
    echo "Batch finished successfully."
  else
    echo "Batch finished with errors (code=$rc)."
  fi
  return $rc
}

main "$@"
