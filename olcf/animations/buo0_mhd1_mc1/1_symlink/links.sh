#!/bin/bash

../../link_checkpoints.sh \
  --start 0 --end 11 \
  --input "/lustre/orion/nfu106/proj-shared/dalinger/inputs/lm-mhd/2.1-nekrs/05_m3_mhd_no_buo" \
  --output "/lustre/orion/fus166/proj-shared/rweb/gb_final/animation/buo0_mhd1_mc1" \
  --batch-size 20 \
  --force
