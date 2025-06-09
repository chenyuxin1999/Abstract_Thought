#!/bin/bash

# 模型列表
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


RATIO="0.01"
NEURONS_PATH="./neuron_deactivation"
SAVE_PATH="./deactivate_model_param"


for MODEL in "${MODELS[@]}"; do
  echo "🔧 Processing model: $MODEL"

  python deactivation.py \
    --model_name "$MODEL" \
    --ratio "$RATIO" \
    --neurons_path "$NEURONS_PATH" \
    --save_path "$SAVE_PATH"

  echo "✅ Done: $MODEL"
done

echo "🎉 All models processed."
