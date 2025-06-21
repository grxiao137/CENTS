#!/bin/bash

MODEL_DIRS_PATHS=(
  "./gpt3.5"
  "./gpt4o"
)

for MODEL_DIR_PATH in "${MODEL_DIRS_PATHS[@]}"; do
  if [ -d "$MODEL_DIR_PATH" ]; then
    echo "Model --- $MODEL_DIR_PATH ---"
    cd "$MODEL_DIR_PATH"
    bash run_eval.sh
    cd ..
  else
    echo "Directory $MODEL_DIR_PATH does not exist."
  fi
done
