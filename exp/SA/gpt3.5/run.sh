#!/bin/bash

OUT_DIR='./out/'
TABLE_DIR='../raw/'
LABEL_PATH='../../../src/data/topk-turl.pkl'


python ./main.py \
	--model gpt-3.5-turbo \
	--data "$TABLE_DIR" \
	--topk 50 \
	--label "$LABEL_PATH" \
	--outdir "$OUT_DIR" \
	--ntables 4646 \
