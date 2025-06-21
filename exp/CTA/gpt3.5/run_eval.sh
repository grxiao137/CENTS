#!/bin/bash

EVAL_SCRIPT_PATH="../cta_eval.py"     
BASE_OUT_DIR="./out"                  

if [[ ! -f "$EVAL_SCRIPT_PATH" ]]; then
  echo "Error: $EVAL_SCRIPT_PATH not found." >&2
  exit 1
fi

for OUT_DIR in "$BASE_OUT_DIR"/*/; do
  [[ -e "$OUT_DIR" ]] || { 
      echo "No folders found under $BASE_OUT_DIR"; 
      exit 1; 
  }

  [[ -d "$OUT_DIR" ]] || continue    

  echo $OUT_DIR
  python "$EVAL_SCRIPT_PATH" --dir "$OUT_DIR" 
  echo                                 
done

