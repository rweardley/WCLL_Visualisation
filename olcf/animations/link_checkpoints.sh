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
shopt -s nullglob

# ---------------------------------------------------------------- defaults ---
batch_size=20
stride=1
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
  -s, --stride N         Keep every Nth checkpoint in the global sequence
                         (default: 1).
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
        -s|--stride)     stride="${2:?missing value for $1}";     shift 2 ;;
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
[[ "$stride" =~ ^[0-9]+$ && "$stride" -gt 0 ]] \
    || die "stride must be a positive integer (got '$stride')"
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

# write_zero_metadata <batch-dir> <yes|no>
#
# Records whether this run's f00000 was selected by the global stride and
# therefore needs to be retained downstream as an animation checkpoint.
# f00000 itself is always linked into every batch regardless of this flag.
write_zero_metadata() {
    local batch_dir="$1" keep_zero="$2"
    local meta="$batch_dir/zero_checkpoint"

    if [[ $dry_run -eq 1 ]]; then
        printf '  [dry-run] write %s: keep_zero=%s\n' "$meta" "$keep_zero"
        return
    fi

    printf 'keep_zero: %s\n' "$keep_zero" > "$meta"
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

# Position in the complete temporal checkpoint sequence. This deliberately
# spans run boundaries so that --stride produces uniform temporal spacing
# throughout the final animation.
global_index=0

for run_name in "${runs[@]}"; do
    run_dir="$input_abs/$run_name"

    if [[ ! -d "$run_dir" ]]; then
        warn "skipping $run_name: directory does not exist"
        continue
    fi

    # Collect the field files: <prefix>.fNNNNN
    # (shell glob: readdir only, no per-file stat)
    all_files=( "$run_dir"/*0.f[0-9][0-9][0-9][0-9][0-9] )
    all_files=( "${all_files[@]##*/}" )   # strip the directory, keep basenames

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

    # All checkpoints for this run, including f00000, in numerical order.
    # f00000 is both the mesh required to read the run and the run's zeroth
    # checkpoint, so it participates in the global stride sequence.
    mapfile -t run_checkpoints < <(
        printf '%s\n' "${all_files[@]}" \
            | grep -E "^${run_prefix//./\\.}\.f[0-9]{5}$"
    )

    n_run_checkpoints=${#run_checkpoints[@]}

    if [[ $n_run_checkpoints -eq 0 ]]; then
        warn "$run_name: no checkpoints found"
        continue
    fi

    # Apply the stride to the single global temporal sequence spanning all
    # processed runs. The selected array contains only non-zero checkpoints
    # that belong to the animation sequence; the mandatory mesh link is
    # added separately to every batch below.
    selected=()
    zero_selected=0

    for checkpoint_name in "${run_checkpoints[@]}"; do
        if (( global_index % stride == 0 )); then
            if [[ "$checkpoint_name" == "${run_prefix}.f00000" ]]; then
                # f00000 is represented by the mandatory mesh link below;
                # do not add a second link to it as .f00001.
                zero_selected=1
            else
                selected+=("$checkpoint_name")
            fi
        fi

        global_index=$((global_index + 1))
    done

    n_selected=${#selected[@]}

    # With --force, drop every existing <run>_N so a smaller batch count
    # doesn't leave stale directories behind.
    if [[ $force -eq 1 ]]; then
        for stale in "$output_abs/${run_name}"_[0-9]*; do
            [[ -e "$stale" ]] && run rm -rf "$stale"
        done
    fi

    if (( n_selected > 0 )); then
        n_batches=$(( (n_selected + batch_size - 1) / batch_size ))
    else
        # Even when this run contributes no stride-selected checkpoint, one
        # batch is required so its mandatory f00000 mesh is available.
        n_batches=1
    fi

    printf '%s: %d original checkpoints -> %d selected with stride %d -> %d batch director%s\n' \
        "$run_name" "$n_run_checkpoints" "$n_selected" "$stride" "$n_batches" \
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

        # 2. this batch's globally stride-selected checkpoints, renumbered
        #    locally from .f00001. The mesh link above does not consume a
        #    batch slot.
        idx=1
        for ((j = b * batch_size; j < (b + 1) * batch_size && j < n_selected; j++)); do
            printf -v link_name '%s.f%05d' "$run_prefix" "$idx"
            link_to "$run_dir/${selected[j]}" "$batch_dir/$link_name"
            idx=$((idx + 1))
            total_links=$((total_links + 1))
        done

        # 3. the .nek5000 metadata file: idx == mesh link + batch links
        write_metadata "$batch_dir" "$run_prefix" "$idx" "$run_dir"

                if (( n_selected > 0 )); then
            first_idx=$(( b * batch_size ))
            last_idx=$(( (b + 1) * batch_size - 1 ))
            (( last_idx >= n_selected )) && last_idx=$(( n_selected - 1 ))
            first_src="${selected[first_idx]}"
            last_src="${selected[last_idx]}"
        else
            first_src="(mesh only)"
            last_src="(mesh only)"
        fi

        printf '  %-16s %s -> %s  (%d checkpoints)\n' \
            "${run_name}_$((b + 1))" "$first_src" "$last_src" "$((idx - 1))"

        total_dirs=$((total_dirs + 1))
    done
done

printf '\nDone: %d batch directories, %d symlinks under %s\n' \
    "$total_dirs" "$total_links" "$output_abs"