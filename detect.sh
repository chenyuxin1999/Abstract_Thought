#!/bin/bash

RATIO="0.01"
LANGS="en,zh,th,sw,fr,de"
SAMPLE_SIZE=100
TASK="gsm"


MODELS=(
  "Qwen2.5-7B"
  "Qwen2.5-3B"
  "Qwen2.5-0.5B"
  "Qwen2.5-1.5B"
  "Qwen1.5-7B"
  "Qwen2-7B"
  "Llama-3.2-1B"
  "Llama-3.2-3B"
  "Llama-3.1-8B"
  "Llama-1-7b"
  "Llama-2-7b-hf"
  "gemma-7b"
  "gemma-2-9b"
  "Llama-3-8B"
  "Qwen2-0.5B"
  "Qwen2-1.5B"
  "Qwen1.5-0.5B"
  "Qwen1.5-1.8B"
  "Qwen1.5-4B"
  "gemma-3-4b-pt"
)


MODEL_BASE_PATH="/huggingface_model"


for MODEL in "${MODELS[@]}"; do
  MODEL_PATH="${MODEL_BASE_PATH}/${MODEL}"
  echo "🔍 Detecting neurons for: $MODEL"

  CUDA_VISIBLE_DEVICES=3 python detect.py \
    --base "$MODEL_PATH" \
    --atten_ratio "$RATIO" \
    --ffn_ratio "$RATIO" \
    --lang "$LANGS" \
    --sample_size "$SAMPLE_SIZE" \
    --task "$TASK"

  echo "✅ Detection complete for: $MODEL"
done

echo "🎉 All models processed for neuron detection."
