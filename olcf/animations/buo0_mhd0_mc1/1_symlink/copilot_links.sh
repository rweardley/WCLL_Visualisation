#!/usr/bin/env bash

create_checkpoint_links() {
    local casedir="$1"
    local outdir="$2"
    local checkpoint_prefix="$3"
    local chunksize="$4"

    mkdir -p "$outdir"

    shopt -s nullglob

    for rundir in "${casedir}"/[0-9][0-9]run; do
        [[ -d "$rundir" ]] || continue

        local run
        run=$(basename "$rundir")

        # Pick up all checkpoint files in lexical order
        local checkpoints=(
            "$rundir"/${checkpoint_prefix}*0.f*
        )

        local nfiles=${#checkpoints[@]}
  
casedir="/lustre/orion/nfu106/proj-shared/gottems/GB26_finalist/03_m3_no_buo/03_m3_no_buo"
outdir="/lustre/orion/fus166/proj-shared/rweb/gb_final/animation/buo0_mhd0_mc1/"
checkpoint_prefix=""
chunksize=20

create_checkpoint_links $casedir $outfile $logfile_prefix $checkpoint_prefix