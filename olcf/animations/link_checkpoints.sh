#!/usr/bin/env bash
#
# link_checkpoints.sh -- split NekRS checkpoint output into batched symlink dirs.
#
# For each run directory (e.g. 00run .. 18run) this creates
#   <output>/<run>_1, <output>/<run>_2, ...
# each containing at most <batch-size> checkpoint symlinks, renumbered from
# .f00001, and preceded by a link to the mesh-carrying .f00000 file.
#
# Example (batch size 20, sources .f00000 .. .f00057):
#   18run_1/  f00000 -> 18run/f00000   f00001..f00020 -> 18run/f00001..f00020
#   18run_2/  f00000 -> 18run/f00000   f00001..f00020 -> 18run/f00021..f00040
#   18run_3/  f00000 -> 18run/f00000   f00001..f00017 -> 18run/f00041..f00057

set -euo pipefail

# ---------------------------------------------------------------- defaults ---
batch_size=20
input_dir="."
output_dir="./checkpoint_links"
prefix=""
run_start=0
run_end=18
run_suffix="run"
runs=()
force=0
dry_run=0
relative=0

usage() {
    cat <<'EOF'
Usage: link_checkpoints.sh [options]

Options:
  -b, --batch-size N     Checkpoints per batch directory, excluding the mesh
                         link (default: 20).
  -i, --input DIR        Directory containing the XXrun folders (default: .).
  -o, --output DIR       Where the batch directories are created
                         (default: ./checkpoint_links).
  -p, --prefix NAME      Field-file prefix, i.e. the part before ".f00000"
                         (e.g. "turbPipe0"). Auto-detected if omitted.
  -r, --runs "A B C"     Explicit space-separated list of run directory names
                         (e.g. "00run 05run 18run"). Overrides --start/--end.
      --start N          First run index (default: 0).
      --end N            Last run index, inclusive (default: 18).
      --suffix S         Text after the two-digit index (default: "run").
  -R, --relative         Make relative symlinks instead of absolute ones.
  -f, --force            Delete any pre-existing <run>_N directories first.
  -n, --dry-run          Print what would be done, change nothing.
  -h, --help             Show this help.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
warn() { printf 'warning: %s\n' "$*" >&2; }

# ------------------------------------------------------------ arg parsing ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--batch-size) batch_size="${2:?missing value for $1}"; shift 2 ;;
        -i|--input)      input_dir="${2:?missing value for $1}";  shift 2 ;;
        -o|--output)     output_dir="${2:?missing value for $1}"; shift 2 ;;
        -p|--prefix)     prefix="${2:?missing value for $1}";     shift 2 ;;
        -r|--runs)       read -r -a runs <<< "${2:?missing value for $1}"; shift 2 ;;
        --start)         run_start="${2:?missing value for $1}";  shift 2 ;;
        --end)           run_end="${2:?missing value for $1}";    shift 2 ;;
        --suffix)        run_suffix="${2-}";                      shift 2 ;;
        -R|--relative)   relative=1; shift ;;
        -f|--force)      force=1;    shift ;;
        -n|--dry-run)    dry_run=1;  shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               usage >&2; die "unknown argument: $1" ;;
    esac
done

[[ "$batch_size" =~ ^[0-9]+$ && "$batch_size" -gt 0 ]] \
    || die "batch size must be a positive integer (got '$batch_size')"
[[ -d "$input_dir" ]] || die "input directory not found: $input_dir"

# Build the default run list if none was given explicitly.
if [[ ${#runs[@]} -eq 0 ]]; then
    for ((i = run_start; i <= run_end; i++)); do
        runs+=("$(printf '%02d%s' "$i" "$run_suffix")")
    done
fi

run() {  # execute, or just echo under --dry-run
    if [[ $dry_run -eq 1 ]]; then
        printf '  [dry-run] %s\n' "$*"
    else
        "$@"
    fi
}

# write_metadata <batch-dir> <field-prefix> <n-timesteps> <run-dir>
#
# Produces <casename>.nek5000 inside the batch directory, where <casename> is
# the field prefix with its trailing output-index digit removed, e.g. field
# files "pink5m3_no_buo0.f00000" -> "pink5m3_no_buo.nek5000".  If the source
# run directory already has a .nek5000, it is copied and its firsttimestep /
# numtimesteps lines rewritten, so any other fields it carries are preserved.
write_metadata() {
    local batch_dir="$1" fprefix="$2" n_ts="$3" src_dir="$4"
    local casename="${fprefix%0}" src meta

    if [[ "$casename" == "$fprefix" ]]; then
        warn "field prefix '$fprefix' does not end in an output index digit;" \
             "using it verbatim in the file template"
    fi

    meta="$batch_dir/${casename}.nek5000"
    src="$src_dir/${casename}.nek5000"

    if [[ $dry_run -eq 1 ]]; then
        printf '  [dry-run] write %s (numtimesteps: %d)\n' "$meta" "$n_ts"
        return
    fi

    if [[ -f "$src" ]]; then
        cp "$src" "$meta"
        sed -i -e 's/^firsttimestep:.*/firsttimestep: 0/' \
               -e "s/^numtimesteps:.*/numtimesteps: $n_ts/" "$meta"
    else
        cat > "$meta" <<EOF
filetemplate: ${casename}%01d.f%05d
firsttimestep: 0
numtimesteps: $n_ts
EOF
    fi
}

input_abs="$(cd "$input_dir" && pwd -P)"
if [[ $dry_run -eq 0 ]]; then
    mkdir -p "$output_dir"
    output_abs="$(cd "$output_dir" && pwd -P)"
else
    output_abs="$output_dir"
fi

# ----------------------------------------------------------------- main -----
total_dirs=0
total_links=0

for run_name in "${runs[@]}"; do
    run_dir="$input_abs/$run_name"

    if [[ ! -d "$run_dir" ]]; then
        warn "skipping $run_name: directory does not exist"
        continue
    fi

    # Collect the field files: <prefix>.fNNNNN
    mapfile -t all_files < <(
        find "$run_dir" -maxdepth 1 -type f \
             -name '*0.f[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | sort
    )

    if [[ ${#all_files[@]} -eq 0 ]]; then
        warn "skipping $run_name: no *0.f????? files found"
        continue
    fi

    # Determine the prefix (everything before the final ".f").
    run_prefix="$prefix"
    if [[ -z "$run_prefix" ]]; then
        mapfile -t found < <(printf '%s\n' "${all_files[@]}" | sed 's/\.f[0-9]\{5\}$//' | sort -u)
        if [[ ${#found[@]} -gt 1 ]]; then
            die "$run_name contains several field prefixes (${found[*]}); rerun with --prefix"
        fi
        run_prefix="${found[0]}"
    fi

    mesh="$run_dir/${run_prefix}.f00000"
    [[ -f "$mesh" ]] || die "$run_name: mesh file ${run_prefix}.f00000 not found"

    # Data checkpoints = everything except .f00000, in numerical order.
    mapfile -t data < <(
        printf '%s\n' "${all_files[@]}" \
            | grep -E "^${run_prefix//./\\.}\.f[0-9]{5}$" \
            | grep -v "\.f00000$"
    )

    n_data=${#data[@]}
    if [[ $n_data -eq 0 ]]; then
        warn "$run_name: only the mesh file is present, nothing to batch"
        continue
    fi

    # With --force, drop every existing <run>_N so a smaller batch count
    # doesn't leave stale directories behind.
    if [[ $force -eq 1 ]]; then
        for stale in "$output_abs/${run_name}"_[0-9]*; do
            [[ -e "$stale" ]] && run rm -rf "$stale"
        done
    fi

    n_batches=$(( (n_data + batch_size - 1) / batch_size ))
    printf '%s: %d checkpoints -> %d batch director%s\n' \
        "$run_name" "$n_data" "$n_batches" \
        "$([[ $n_batches -eq 1 ]] && echo y || echo ies)"

    for ((b = 0; b < n_batches; b++)); do
        batch_dir="$output_abs/${run_name}_$((b + 1))"

        if [[ -e "$batch_dir" ]]; then
            if [[ $force -eq 1 ]]; then
                run rm -rf "$batch_dir"
            else
                die "$batch_dir already exists (use --force to overwrite)"
            fi
        fi
        run mkdir -p "$batch_dir"

        # Link target helper: absolute by default, relative with -R.
        link_to() {  # $1 = source path, $2 = link path
            local src="$1"
            if [[ $relative -eq 1 ]]; then
                src="$(realpath -m --relative-to="$(dirname "$2")" "$1")"
            fi
            run ln -s "$src" "$2"
        }

        # 1. the mesh, always as .f00000
        link_to "$mesh" "$batch_dir/${run_prefix}.f00000"
        total_links=$((total_links + 1))

        # 2. this batch's checkpoints, renumbered from .f00001
        idx=1
        for ((j = b * batch_size; j < (b + 1) * batch_size && j < n_data; j++)); do
            printf -v link_name '%s.f%05d' "$run_prefix" "$idx"
            link_to "$run_dir/${data[j]}" "$batch_dir/$link_name"
            idx=$((idx + 1))
            total_links=$((total_links + 1))
        done

        # 3. the .nek5000 metadata file: idx == mesh link + batch links
        write_metadata "$batch_dir" "$run_prefix" "$idx" "$run_dir"

        printf '  %-16s %s -> %s  (%d checkpoints)\n' \
            "${run_name}_$((b + 1))" "${data[b * batch_size]}" \
            "${data[$(( (b + 1) * batch_size < n_data ? (b + 1) * batch_size - 1 : n_data - 1 ))]}" \
            "$((idx - 1))"

        total_dirs=$((total_dirs + 1))
    done
done

printf '\nDone: %d batch directories, %d symlinks under %s\n' \
    "$total_dirs" "$total_links" "$output_abs"