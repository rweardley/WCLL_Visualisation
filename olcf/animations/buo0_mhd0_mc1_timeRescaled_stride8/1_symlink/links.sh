#!/bin/bash

../../link_checkpoints.sh \
  --start 0 --end 37 \
  --input "/lustre/orion/proj-shared/fus166/aldeia2/00_m3_no_buo_s2" \
  --output "/lustre/orion/fus166/proj-shared/rweb/gb_final/animation/buo0_mhd0_mc1_timeRescaled_stride8" \
  --stride 8 \
  --batch-size 20 \
  --force
