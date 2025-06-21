#!/bin/bash

MODEL_DIRS_PATHS=(
  "./gpt3.5"
  "./gpt4o"
  "./deepseek"
  "./tabgptv2"
)

for MODEL_DIR_PATH in "${MODEL_DIRS_PATHS[@]}"; do
  if [ -d "$MODEL_DIR_PATH" ]; then
    echo "Running $MODEL_DIR_PATH"
    cd "$MODEL_DIR_PATH"
    bash run.sh
    cd ..
  else
    echo "Directory $MODEL_DIR_PATH does not exist."
  fi
done
