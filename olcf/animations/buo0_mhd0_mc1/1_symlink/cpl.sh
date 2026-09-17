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

        local checkpoints=(
            "$rundir"/${checkpoint_prefix}*0.f*
        )

        local nfiles=${#checkpoints[@]}
        (( nfiles > 0 )) || continue

        echo "$run: found $nfiles checkpoint files"

        local mesh_file="${checkpoints[0]}"
        local batch=1

        for ((i=1; i<nfiles; i+=chunksize)); do

            local targetdir="${outdir}/${run}_${batch}"
            mkdir -p "$targetdir"

            ln -sf \
                "$(realpath "$mesh_file")" \
                "$targetdir/$(basename "$mesh_file")"

            local local_idx=1

            for ((j=i; j<i+chunksize && j<nfiles; j++)); do

                local src="${checkpoints[j]}"
                local base
                base=$(basename "$src")

                newbase="${base%*.f*}.f$(printf '%05d' "$local_idx")"

                echo $base "->" $newbase

                # ln -sf "$(realpath "$src")" "$   ((batch++))
        done
    done
}

casedir="/lustre/orion/nfu106/proj-shared/gottems/GB26_finalist/03_m3_no_buo/03_m3_no_buo"
outdir="/lustre/orion/fus166/proj-shared/rweb/gb_final/animation/buo0_mhd0_mc1/"
checkpoint_prefix=""
chunksize=20

create_checkpoint_links \
    "$casedir" \
    "$outdir" \
    "$checkpoint_prefix" \
    "$chunksize"