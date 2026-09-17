#!/usr/bin/env bash

casedir="/lustre/orion/nfu106/proj-shared/gottems/GB26_finalist/03_m3_no_buo/03_m3_no_buo"
outfile="run_endtimes.tsv"
logfile_prefix="nekRS_pink"

# Header
printf "run\tendtime\n" > "$outfile"

for rundir in ${casedir}/[0-9][0-9]run; do
    echo $rundir
    [[ -d "$rundir" ]] || continue

    run=$(basename "$rundir" | sed 's/run$//')

    logfile=$(ls "$rundir"/${logfile_prefix}*.out 2>/dev/null | head -n1)
    [[ -f "$logfile" ]] || continue

    endtime=$(awk '/t=/ {t=$3} END{print t}' "$logfile")

    checkpoints=$(ls "$rundir"/*0.f* | wc -l)

    printf "%s\t%s\t%s\n" "$run" "$endtime" "$checkpoints" >> "$outfile"
done