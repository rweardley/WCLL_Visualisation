#!/usr/bin/env bash

set -euo pipefail

###############################################################################
# USER CONFIGURATION
###############################################################################

# Base directory containing the setup directories.
BASE_DIR=`pwd`

# Setup 1
SETUP1="buo0_mhd0_mc1"
VIS1="vol_fluids_velocity_all_jet_split"

# XX values to include from each setup.
#
# setup1 directories have the form:
#     XXrun_Y
#
# XX values are specified as integers here; the script handles the
# zero-padding in the directory names.
SETUP1_XX=(15 16)

# Setup 2
#
# Set SETUP2="" if setup2 is not currently available.
#
# setup2 directories have the form:
#     XXrunY
# SETUP2="buo0_mhd1_mc1"
SETUP2=""
VIS2=$VIS1

# XX values to include from setup2.
SETUP2_XX=()

# Output animation.
#
# The format is determined by the filename extension.
# Examples:
#     animation.mp4
#     animation.webm
#     animation.mkv
OUTPUT="animation.mp4"

# Frame rate in frames per second.
FPS=12

###############################################################################
# END OF USER CONFIGURATION
###############################################################################


###############################################################################
# CHECK REQUIREMENTS
###############################################################################

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg was not found in PATH." >&2
    exit 1
fi

if ! command -v awk >/dev/null 2>&1; then
    echo "Error: awk was not found in PATH." >&2
    exit 1
fi

if ! command -v sort >/dev/null 2>&1; then
    echo "Error: sort was not found in PATH." >&2
    exit 1
fi

if ! [[ "$FPS" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk "BEGIN { exit ($FPS > 0 ? 0 : 1) }"; then
    echo "Error: FPS must be a positive number." >&2
    exit 1
fi

SETUP1_PATH="${BASE_DIR}/${SETUP1}/${VIS1}"
SETUP2_PATH="${BASE_DIR}/${SETUP2}/${VIS2}"

if [[ ! -d "$SETUP1_PATH" ]]; then
    echo "Error: setup1 directory does not exist:" >&2
    echo "  $SETUP1_PATH" >&2
    exit 1
fi

# setup2 is optional.
if [[ -n "$SETUP2" ]]; then
    SETUP2_PATH="${BASE_DIR}/${SETUP2}/${VOL2}"

    if [[ ! -d "$SETUP2_PATH" ]]; then
        echo "Error: setup2 directory does not exist:" >&2
        echo "  $SETUP2_PATH" >&2
        exit 1
    fi
fi


###############################################################################
# TEMPORARY FILE
###############################################################################

CONCAT_FILE=$(mktemp)

cleanup() {
    rm -f "$CONCAT_FILE"
}

trap cleanup EXIT


###############################################################################
# HELPER FUNCTIONS
###############################################################################

# Escape a filename for use in an FFmpeg concat-demuxer file.
escape_for_ffmpeg() {
    local path="$1"

    # Escape single quotes.
    path="${path//\'/\'\\\'\'}"

    printf "'%s'" "$path"
}


###############################################################################
# ADD ALL FRAMES FROM ONE RUN
###############################################################################

add_run_frames() {
    local run_dir="$1"
    local fps="$2"

    local frames=()
    local frame
    local duration

    # Find PNG files matching:
    #
    #     frame_ZZZZ.png
    #
    # where ZZZZ consists of exactly four digits.
    while IFS= read -r -d '' frame; do
        frames+=("$frame")
    done < <(
        find "$run_dir" \
            -maxdepth 1 \
            -type f \
            -regextype posix-extended \
            -regex '.*/frame_[0-9]{4}\.png' \
            -print0
    )

    if (( ${#frames[@]} == 0 )); then
        echo "Warning: no frames found in:" >&2
        echo "  $run_dir" >&2
        return
    fi

    # Sort frames numerically by ZZZZ.
    #
    # This is deliberately numeric rather than relying on ordinary
    # alphabetical sorting.
    mapfile -t frames < <(
        printf '%s\n' "${frames[@]}" |
        awk -F'frame_|\\.png' '{ print $2 "\t" $0 }' |
        sort -n -k1,1 |
        cut -f2-
    )

    # Duration of each frame.
    duration=$(awk "BEGIN { printf \"%.12f\", 1/$fps }")

    for frame in "${frames[@]}"; do
        printf "file %s\n" "$(escape_for_ffmpeg "$frame")" >> "$CONCAT_FILE"
        printf "duration %s\n" "$duration" >> "$CONCAT_FILE"
    done
}


###############################################################################
# PROCESS ONE SETUP
###############################################################################

process_setup() {
    local setup_path="$1"
    local xx_array_name="$2"
    local naming_style="$3"

    # Retrieve the array by name.
    local -n xx_values="$xx_array_name"

    local xx
    local xx_padded
    local pattern
    local run_dir
    local basename
    local y

    local run_dirs=()

    for xx in "${xx_values[@]}"; do

        # XX must be two digits in the directory name.
        #
        # For example:
        #   4  -> 04
        #   14 -> 14
        #   25 -> 25
        printf -v xx_padded '%02d' "$xx"

        if [[ "$naming_style" == "underscore" ]]; then
            # setup1:
            #     XXrun_Y
            pattern="${xx_padded}run_"*
        else
            # setup2:
            #     XXrunY
            pattern="${xx_padded}run"*
        fi

        run_dirs=()

        #######################################################################
        # FIND ALL RUN DIRECTORIES FOR THIS XX
        #######################################################################

        while IFS= read -r -d '' run_dir; do

            basename=$(basename "$run_dir")

            if [[ "$naming_style" == "underscore" ]]; then

                # Expected:
                #     XXrun_Y
                #
                # Capture Y.
                if [[ "$basename" =~ ^${xx_padded}run_([0-9]+)$ ]]; then
                    y="${BASH_REMATCH[1]}"
                else
                    continue
                fi

            else

                # Expected:
                #     XXrunY
                #
                # Capture Y.
                if [[ "$basename" =~ ^${xx_padded}run([0-9]+)$ ]]; then
                    y="${BASH_REMATCH[1]}"
                else
                    continue
                fi

            fi

            # Store Y together with the directory so that it can be
            # numerically sorted.
            run_dirs+=("${y}"$'\t'"${run_dir}")

        done < <(
            find "$setup_path" \
                -mindepth 1 \
                -maxdepth 1 \
                -type d \
                -name "$pattern" \
                -print0
        )

        #######################################################################
        # SORT RUNS NUMERICALLY BY Y
        #######################################################################

        if (( ${#run_dirs[@]} == 0 )); then
            echo "Warning: no runs found for XX=$xx in:" >&2
            echo "  $setup_path" >&2
            continue
        fi

        mapfile -t run_dirs < <(
            printf '%s\n' "${run_dirs[@]}" |
            sort -n -k1,1 |
            cut -f2-
        )

        #######################################################################
        # ADD FRAMES FROM EACH RUN
        #######################################################################

        for run_dir in "${run_dirs[@]}"; do
            echo "Adding: $run_dir" >&2
            add_run_frames "$run_dir" "$FPS"
        done

    done
}


###############################################################################
# BUILD THE COMPLETE FRAME LIST
###############################################################################

echo "============================================================" >&2
echo "Building animation frame list" >&2
echo "============================================================" >&2

echo >&2
echo "Processing setup1..." >&2

# setup1 uses:
#
#     XXrun_Y
#
process_setup \
    "$SETUP1_PATH" \
    SETUP1_XX \
    underscore


# setup2 is optional.
if [[ -n "$SETUP2" ]]; then

    echo >&2
    echo "Processing setup2..." >&2

    # setup2 uses:
    #
    #     XXrunY
    #
    process_setup \
        "$SETUP2_PATH" \
        SETUP2_XX \
        plain

else

    echo >&2
    echo "setup2 not specified; skipping setup2." >&2

fi

###############################################################################
# CHECK THAT FRAMES WERE FOUND
###############################################################################

if [[ ! -s "$CONCAT_FILE" ]]; then
    echo >&2
    echo "Error: no PNG frames were found." >&2
    exit 1
fi

FRAME_COUNT=$(grep -c '^file ' "$CONCAT_FILE")

echo >&2
echo "============================================================" >&2
echo "Frame list complete" >&2
echo "============================================================" >&2
echo "Frames found : $FRAME_COUNT" >&2
echo "Frame rate   : $FPS fps" >&2
echo "Output       : $OUTPUT" >&2
echo >&2


###############################################################################
# CREATE ANIMATION
###############################################################################

ffmpeg \
    -hide_banner \
    -loglevel info \
    -f concat \
    -safe 0 \
    -i "$CONCAT_FILE" \
    -vf "fps=${FPS}" \
    -vsync cfr \
    "$OUTPUT"


###############################################################################
# DONE
###############################################################################

echo >&2
echo "============================================================" >&2
echo "Done" >&2
echo "============================================================" >&2
echo "Animation written to:" >&2
echo "  $OUTPUT" >&2