#!/bin/bash

../../link_checkpoints.sh \
  --start 0 --end 18 \
  --input "/lustre/orion/nfu106/proj-shared/gottems/GB26_finalist/03_m3_no_buo/03_m3_no_buo" \
  --output "/lustre/orion/fus166/proj-shared/rweb/gb_final/animation/buo0_mhd0_mc1" \
  --batch-size 20 \
  --force
