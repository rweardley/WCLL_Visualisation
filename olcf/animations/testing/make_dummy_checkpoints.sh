#!/usr/bin/env bash
#
# make_dummy_checkpoints.sh -- generate fake NekRS checkpoint output for
# testing link_checkpoints.sh locally.
#
# Creates <output>/00run .. 18run, each holding <prefix>.f00000 ..
# <prefix>.f000NN, where the number of files per run is random in [min, max].
# Files are tiny placeholders. A matching <casename>.nek5000 is also written
# in each run directory so the metadata-copying path gets exercised.

set -euo pipefail

output_dir="./dummy_runs"
prefix="turbPipe0"
run_start=0
run_end=18
run_suffix="run"
min_ckpt=57
max_ckpt=65
seed=""
no_nek5000=0

usage() {
    cat <<'EOF'
Usage: make_dummy_checkpoints.sh [options]

  -o, --output DIR    Where to create the run directories (default: ./dummy_runs)
  -p, --prefix NAME   Field-file prefix, must end in a digit (default: turbPipe0)
      --start N       First run index (default: 0)
      --end N         Last run index, inclusive (default: 18)
      --suffix S      Text after the two-digit index (default: run)
      --min N         Minimum files per run, including f00000 (default: 57)
      --max N         Maximum files per run, including f00000 (default: 65)
      --seed N        Seed the RNG for reproducible output
      --no-nek5000    Don't write a .nek5000 file in each run directory
  -h, --help          Show this help
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)  output_dir="${2:?missing value for $1}"; shift 2 ;;
        -p|--prefix)  prefix="${2:?missing value for $1}";     shift 2 ;;
        --start)      run_start="${2:?missing value for $1}";  shift 2 ;;
        --end)        run_end="${2:?missing value for $1}";    shift 2 ;;
        --suffix)     run_suffix="${2-}";                      shift 2 ;;
        --min)        min_ckpt="${2:?missing value for $1}";   shift 2 ;;
        --max)        max_ckpt="${2:?missing value for $1}";   shift 2 ;;
        --seed)       seed="${2:?missing value for $1}";       shift 2 ;;
        --no-nek5000) no_nek5000=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            usage >&2; die "unknown argument: $1" ;;
    esac
done

[[ "$prefix" =~ [0-9]$ ]] || die "prefix must end in a digit (got '$prefix')"
[[ "$min_ckpt" =~ ^[0-9]+$ && "$max_ckpt" =~ ^[0-9]+$ ]] || die "--min/--max must be integers"
(( min_ckpt >= 1 && max_ckpt >= min_ckpt )) || die "need 1 <= min <= max"

# Seeding RANDOM makes the sequence reproducible.
[[ -n "$seed" ]] && RANDOM=$seed

casename="${prefix%0}"
total_files=0

mkdir -p "$output_dir"

for ((i = run_start; i <= run_end; i++)); do
    run_name="$(printf '%02d%s' "$i" "$run_suffix")"
    run_dir="$output_dir/$run_name"
    mkdir -p "$run_dir"

    n=$(( min_ckpt + RANDOM % (max_ckpt - min_ckpt + 1) ))

    for ((k = 0; k < n; k++)); do
        fname="$(printf '%s.f%05d' "$prefix" "$k")"
        printf 'dummy checkpoint %s/%s\n' "$run_name" "$fname" > "$run_dir/$fname"
    done

    if [[ $no_nek5000 -eq 0 ]]; then
        cat > "$run_dir/${casename}.nek5000" <<EOF
filetemplate: ${casename}%01d.f%05d
firsttimestep: 0
numtimesteps: $n
EOF
    fi

    total_files=$((total_files + n))
    printf '%s: %d files (%s.f00000 .. %s.f%05d)\n' \
        "$run_name" "$n" "$prefix" "$prefix" "$((n - 1))"
done

printf '\nDone: %d checkpoint files under %s\n' "$total_files" "$output_dir"
