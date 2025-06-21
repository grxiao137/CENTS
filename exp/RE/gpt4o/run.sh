#!/bin/bash

OUT_DIR='./out/'
TABLE_DIR='../raw/'
python ./main.py \
	--model gpt-4o \
	--data "$TABLE_DIR" \
	--outdir "$OUT_DIR" \
	--ntables 7026 \
