#!/bin/bash

TASK_DIRS_PATHS=(
  "./CTA"
  "./RE"
  "./SA"
)

for TASK_DIRS_PATH in "${TASK_DIRS_PATHS[@]}"; do
  if [ -d "$TASK_DIRS_PATH" ]; then
    echo "Working on evaluating ------------$TASK_DIRS_PATH---------------"
    cd "$TASK_DIRS_PATH"
    bash run_eval_all.sh
    cd ..
  else
    echo "Directory $TASK_DIRS_PATH does not exist."
  fi
done
