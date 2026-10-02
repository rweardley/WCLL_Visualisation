#!/bin/bash

./make_dummy_checkpoints.sh --seed 42

../link_checkpoints.sh \
  --start 0 --end 18 \
  --input `pwd`/"dummy_runs" \
  --output "linked_dummy_runs" \
  --batch-size 20 \
  --force \
  # --verbose

../link_checkpoints.sh \
  --start 0 --end 18 \
  --input `pwd`/"dummy_runs" \
  --output "linked_dummy_runs_stride" \
  --batch-size 20 \
  --stride 4 \
  --force \
  # --verbose
