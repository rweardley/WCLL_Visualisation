#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# USER CONFIGURATION
###############################################################################

# Base directory containing all setup directories.
BASE_DIR=`pwd`

# Define as many setups as you need to appear sequentially in the animation.
# All arrays below must have the same number of entries,
# in the order the setups should appear in the animation.
#
# SETUP_NAMES: directory name of each setup, relative to BASE_DIR.
# SETUP_VIS:   visualisation directory inside each setup.
# SETUP_XX:    whitespace-separated XX values to include for that setup.
# RUN_STYLE:   "underscore" for XXrun_Y, or "plain" for XXrunY.
#
# Examples:
#   "buo0_mhd0_mc1_stride8"
#   "buo0_mhd1_mc1"
#   "buo0_mhd2_mc1"
#

SETUP_NAMES=(
    # "buo0_mhd0_mc1_stride8"
    "buo0_mhd0_mc1_timeRescaled_stride8"
    # "buo0_mhd1_mc1"
)

VIS_DIR_NAME="vol_fluids_velocity_all_jet_split"

SETUP_VIS=(
    # ${VIS_DIR_NAME}
    ${VIS_DIR_NAME}
    # ${VIS_DIR_NAME}
)

# SETUP1_XX=($(seq 33 37))
SETUP2_XX=($(seq 0 3))
# SETUP3_XX=(01)

SETUP_XX=(
    # "${SETUP1_XX[*]}"
    "${SETUP2_XX[*]}"
    # "${SETUP3_XX[*]}"
)

RUN_STYLE=(
    # "underscore"
    "underscore"
    # "underscore"
)

# Individual frames to omit. Paths may be absolute or relative to BASE_DIR.
SKIP_FRAMES=(
    # "buo0_mhd0_mc1_stride8/vol_fluids_velocity_all_jet_split/15run_1/frame_0013.png"
    # "buo0_mhd1_mc1/vol_fluids_velocity_all_jet_split/01run_1/frame_0004.png"
)

# Output animation. The extension selects the container/format.
OUTPUT="animation_generalized.mp4"

# Default frame rate for frames not covered by SLOW_RANGES.
DEFAULT_FPS=10

# Slow-motion ranges (inclusive frame numbers).
# Format: "setup_name/visualisation_directory/run_directory:start-end:fps"
SLOW_RANGES=(
    # "${SETUP_NAMES[1]}/${SETUP_VIS[1]}/00run_1:0-0:10"
    # "${SETUP_NAMES[1]}/${SETUP_VIS[1]}/01run_1:0-4:10"
    # "${SETUP_NAMES[1]}/${SETUP_VIS[1]}/01run_2:0-4:10"
    # "${SETUP_NAMES[1]}/${SETUP_VIS[1]}/01run_1:0-3:10"
)

###############################################################################
# CHECK REQUIREMENTS AND CONFIGURATION
###############################################################################

for cmd in ffmpeg awk sort cut find grep basename dirname realpath sed; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command '$cmd' was not found in PATH." >&2
        exit 1
    fi
done

if ! [[ "$DEFAULT_FPS" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk "BEGIN { exit ($DEFAULT_FPS > 0 ? 0 : 1) }"; then
    echo "Error: DEFAULT_FPS must be a positive number." >&2
    exit 1
fi

n_setups=${#SETUP_NAMES[@]}
if (( n_setups == 0 ||
      ${#SETUP_VIS[@]} != n_setups ||
      ${#SETUP_XX[@]} != n_setups ||
      ${#RUN_STYLE[@]} != n_setups )); then
    echo "Error: SETUP_NAMES, SETUP_VIS, SETUP_XX and RUN_STYLE must have the same non-zero number of entries." >&2
    exit 1
fi

for ((i=0; i<n_setups; i++)); do
    if [[ "${RUN_STYLE[i]}" != "underscore" && "${RUN_STYLE[i]}" != "plain" ]]; then
        echo "Error: RUN_STYLE[$i] must be 'underscore' or 'plain'." >&2
        exit 1
    fi

    setup_path="${BASE_DIR}/${SETUP_NAMES[i]}/${SETUP_VIS[i]}"
    if [[ ! -d "$setup_path" ]]; then
        echo "Error: setup directory does not exist: $setup_path" >&2
        exit 1
    fi
done

###############################################################################
# TEMPORARY CONCAT FILE
###############################################################################

CONCAT_FILE=$(mktemp)
trap 'rm -f "$CONCAT_FILE"' EXIT

###############################################################################
# HELPERS
###############################################################################

should_skip_frame() {
    local frame="$1" skip skip_path
    local frame_real

    frame_real=$(realpath "$frame")
    for skip in "${SKIP_FRAMES[@]}"; do
        [[ -z "$skip" ]] && continue
        if [[ "$skip" = /* ]]; then
            skip_path="$skip"
        else
            skip_path="${BASE_DIR}/${skip}"
        fi
        if [[ -e "$skip_path" ]]; then
            skip_path=$(realpath "$skip_path")
        else
            skip_path=$(realpath -m "$skip_path")
        fi
        [[ "$frame_real" == "$skip_path" ]] && return 0
    done
    return 1
}

escape_for_ffmpeg() {
    local path="$1"
    path="${path//\'/\'\\\'\'}"
    printf "'%s'" "$path"
}

get_frame_fps() {
    local frame="$1"
    local frame_dir frame_number range range_run frame_range range_fps
    local range_start range_end range_run_path

    frame_dir=$(dirname "$frame")
    frame_number=$(basename "$frame" | sed -E 's/^frame_([0-9]{4})\.png$/\1/')
    frame_number=$((10#$frame_number))

    for range in "${SLOW_RANGES[@]}"; do
        [[ -z "$range" ]] && continue
        IFS=':' read -r range_run frame_range range_fps <<< "$range"

        if [[ ! "$frame_range" =~ ^[0-9]+-[0-9]+$ ]] ||
           [[ ! "$range_fps" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
           ! awk "BEGIN { exit ($range_fps > 0 ? 0 : 1) }"; then
            echo "Error: invalid SLOW_RANGES entry: $range" >&2
            exit 1
        fi

        IFS='-' read -r range_start range_end <<< "$frame_range"
        range_start=$((10#$range_start))
        range_end=$((10#$range_end))
        if (( range_end < range_start )); then
            echo "Error: range end is before range start: $range" >&2
            exit 1
        fi

        if [[ "$range_run" = /* ]]; then
            range_run_path="$range_run"
        else
            range_run_path="${BASE_DIR}/${range_run}"
        fi

        if [[ "$(realpath "$frame_dir")" == "$(realpath -m "$range_run_path")" ]] &&
           (( frame_number >= range_start && frame_number <= range_end )); then
            printf '%s\n' "$range_fps"
            return
        fi
    done

    printf '%s\n' "$DEFAULT_FPS"
}

add_run_frames() {
    local run_dir="$1"
    local frames=() frame frame_fps duration

    while IFS= read -r -d '' frame; do
        frames+=("$frame")
    done < <(
        find "$run_dir" -maxdepth 1 -type f \
            -regextype posix-extended \
            -regex '.*/frame_[0-9]{4}\.png' -print0
    )

    if (( ${#frames[@]} == 0 )); then
        echo "Warning: no frames found in: $run_dir" >&2
        return
    fi

    # Sort numerically by the four-digit frame number.
    mapfile -t frames < <(
        printf '%s\n' "${frames[@]}" |
        awk '{
            n=$0
            sub(/^.*frame_/, "", n)
            sub(/\.png$/, "", n)
            print n "\t" $0
        }' |
        sort -n -k1,1 |
        cut -f2-
    )

    for frame in "${frames[@]}"; do
        if should_skip_frame "$frame"; then
            echo "Skipping frame: $frame" >&2
            continue
        fi

        frame_fps=$(get_frame_fps "$frame")
        duration=$(awk "BEGIN { printf \"%.12f\", 1/$frame_fps }")

        printf "file %s\n" "$(escape_for_ffmpeg "$frame")" >> "$CONCAT_FILE"
        printf "duration %s\n" "$duration" >> "$CONCAT_FILE"
    done
}

process_setup() {
    local setup_path="$1"
    local xx_values_string="$2"
    local naming_style="$3"
    local -a xx_values=()
    local -a run_dirs=()
    local xx xx_padded pattern run_dir dir_name y

    read -r -a xx_values <<< "$xx_values_string"

    for xx in "${xx_values[@]}"; do
        if [[ ! "$xx" =~ ^[0-9]+$ ]]; then
            echo "Error: XX values must be integers; got '$xx'." >&2
            exit 1
        fi
        xx=$((10#$xx))
        printf -v xx_padded '%02d' "$xx"

        if [[ "$naming_style" == "underscore" ]]; then
            pattern="${xx_padded}run_"*
        else
            pattern="${xx_padded}run"*
        fi

        run_dirs=()
        while IFS= read -r -d '' run_dir; do
            dir_name=$(basename "$run_dir")

            if [[ "$naming_style" == "underscore" ]]; then
                [[ "$dir_name" =~ ^${xx_padded}run_([0-9]+)$ ]] || continue
            else
                [[ "$dir_name" =~ ^${xx_padded}run([0-9]+)$ ]] || continue
            fi
            y="${BASH_REMATCH[1]}"
            run_dirs+=("${y}"$'\t'"${run_dir}")
        done < <(
            find "$setup_path" -mindepth 1 -maxdepth 1 -type d \
                -name "$pattern" -print0
        )

        if (( ${#run_dirs[@]} == 0 )); then
            echo "Warning: no runs found for XX=$xx in: $setup_path" >&2
            continue
        fi

        mapfile -t run_dirs < <(
            printf '%s\n' "${run_dirs[@]}" |
            sort -n -k1,1 |
            cut -f2-
        )

        for run_dir in "${run_dirs[@]}"; do
            echo "Adding run: $run_dir" >&2
            add_run_frames "$run_dir"
        done
    done
}

###############################################################################
# BUILD THE FRAME LIST IN SETUP ORDER
###############################################################################

echo "Building animation frame list..." >&2

for ((i=0; i<n_setups; i++)); do
    setup_path="${BASE_DIR}/${SETUP_NAMES[i]}/${SETUP_VIS[i]}"
    echo "Processing setup $((i + 1))/${n_setups}: ${SETUP_NAMES[i]}" >&2
    process_setup "$setup_path" "${SETUP_XX[i]}" "${RUN_STYLE[i]}"
done

if [[ ! -s "$CONCAT_FILE" ]]; then
    echo "Error: no PNG frames were found after applying the skip list." >&2
    exit 1
fi

FRAME_COUNT=$(grep -c '^file ' "$CONCAT_FILE")
echo "Frames included : $FRAME_COUNT" >&2
echo "Default FPS     : $DEFAULT_FPS" >&2
echo "Output          : $OUTPUT" >&2

###############################################################################
# ENCODE
###############################################################################

# Repeating the final file makes the last concat duration effective in more
# FFmpeg/container combinations. The repeated frame may still affect playback
# by a tiny amount depending on the output container.
last_file_line=$(grep '^file ' "$CONCAT_FILE" | tail -n 1)
printf '%s\n' "$last_file_line" >> "$CONCAT_FILE"
ffmpeg -hide_banner -loglevel info -f concat -safe 0 -i "$CONCAT_FILE" -vsync vfr "$OUTPUT"

echo "Animation written to: $OUTPUT" >&2
