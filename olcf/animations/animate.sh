#!/usr/bin/env bash

set -euo pipefail

###############################################################################
# USER CONFIGURATION
###############################################################################

# Base directory containing the setup directories.
BASE_DIR=`pwd`


# ============================================================================
# SETUP 1
# ============================================================================

# Name of setup1 directory.
SETUP1="buo0_mhd0_mc1"

# Name of visualisation directory inside setup1.
SETUP1_VIS="vol_fluids_velocity_all_jet_split"

# XX values to include from setup1.
#
# setup1 directories have the form:
#
#     XXrun_Y
#
SETUP1_XX=(13 14 15 16)


# ============================================================================
# SETUP 2
# ============================================================================

# Name of setup2 directory.
#
# Set this to "" if setup2 is not currently available.
#
# SETUP2="buo0_mhd1_mc1"
SETUP2=""

# Name of visualisation directory inside setup2.
SETUP2_VIS=$SETUP1_VIS

# XX values to include from setup2.
#
# setup2 directories have the form:
#
#     XXrunY
#
SETUP2_XX=()


# ============================================================================
# FRAMES TO SKIP
# ============================================================================

# Individual frames that should NOT be included in the animation.
#
# Paths can be either:
#
#   1. Relative to BASE_DIR
#
#      "setup1/vol/14run_1/frame_0012.png"
#
#   2. Absolute
#
#      "/path/to/data/setup1/vol/14run_1/frame_0012.png"
#
# You can add as many frames as necessary.
#
SKIP_FRAMES=(
    "buo0_mhd0_mc1/vol_fluids_velocity_all_jet_split/15run_1/frame_0013.png"
)


# ============================================================================
# OUTPUT
# ============================================================================

# Output animation.
#
# The format is determined by the filename extension.
#
# Examples:
#
#     animation.mp4
#     animation.webm
#     animation.mkv
#
OUTPUT="animation.mp4"

# Frame rate in frames per second.
FPS=24


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


###############################################################################
# SETUP PATHS
###############################################################################

SETUP1_PATH="${BASE_DIR}/${SETUP1}/${SETUP1_VIS}"

if [[ ! -d "$SETUP1_PATH" ]]; then
    echo "Error: setup1 directory does not exist:" >&2
    echo "  $SETUP1_PATH" >&2
    exit 1
fi


# setup2 is optional.
if [[ -n "$SETUP2" ]]; then

    SETUP2_PATH="${BASE_DIR}/${SETUP2}/${SETUP2_VIS}"

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
# FRAME SKIP CHECK
###############################################################################

should_skip_frame() {
    local frame="$1"
    local skip
    local skip_path

    for skip in "${SKIP_FRAMES[@]}"; do

        # Ignore empty entries.
        [[ -z "$skip" ]] && continue

        # Convert relative skip paths into absolute paths.
        if [[ "$skip" = /* ]]; then
            skip_path="$skip"
        else
            skip_path="${BASE_DIR}/${skip}"
        fi

        # Compare canonical absolute paths.
        #
        # realpath is preferable because it handles things such as:
        #   ./foo
        #   ../foo
        #   duplicate slashes
        #
        # If realpath isn't available, fall back to the literal paths.
        if command -v realpath >/dev/null 2>&1; then

            if [[ "$(realpath "$frame")" == "$(realpath -m "$skip_path")" ]]; then
                return 0
            fi

        else

            if [[ "$frame" == "$skip_path" ]]; then
                return 0
            fi

        fi

    done

    return 1
}


###############################################################################
# ESCAPE PATH FOR FFMPEG
###############################################################################

escape_for_ffmpeg() {
    local path="$1"

    # Escape single quotes for the FFmpeg concat demuxer.
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

    ###########################################################################
    # FIND FRAMES
    ###########################################################################

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


    ###########################################################################
    # SORT FRAMES NUMERICALLY
    ###########################################################################

    mapfile -t frames < <(
        printf '%s\n' "${frames[@]}" |
        awk -F'frame_|\\.png' '{ print $2 "\t" $0 }' |
        sort -n -k1,1 |
        cut -f2-
    )


    ###########################################################################
    # FRAME DURATION
    ###########################################################################

    duration=$(awk "BEGIN { printf \"%.12f\", 1/$fps }")


    ###########################################################################
    # ADD FRAMES
    ###########################################################################

    for frame in "${frames[@]}"; do

        # Check whether this frame has been manually excluded.
        if should_skip_frame "$frame"; then
            echo "Skipping: $frame" >&2
            continue
        fi

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


    ###########################################################################
    # PROCESS EACH XX
    ###########################################################################

    for xx in "${xx_values[@]}"; do

        # XX is always two digits in the directory name.
        #
        # Examples:
        #
        #     4  -> 04
        #     14 -> 14
        #     25 -> 25
        #
        printf -v xx_padded '%02d' "$xx"


        #######################################################################
        # DIRECTORY PATTERN
        #######################################################################

        if [[ "$naming_style" == "underscore" ]]; then

            # setup1:
            #
            #     XXrun_Y
            #
            pattern="${xx_padded}run_"*

        else

            # setup2:
            #
            #     XXrunY
            #
            pattern="${xx_padded}run"*

        fi


        run_dirs=()


        #######################################################################
        # FIND ALL RUN DIRECTORIES
        #######################################################################

        while IFS= read -r -d '' run_dir; do

            basename=$(basename "$run_dir")


            if [[ "$naming_style" == "underscore" ]]; then

                # Expected:
                #
                #     XXrun_Y
                #
                if [[ "$basename" =~ ^${xx_padded}run_([0-9]+)$ ]]; then
                    y="${BASH_REMATCH[1]}"
                else
                    continue
                fi

            else

                # Expected:
                #
                #     XXrunY
                #
                if [[ "$basename" =~ ^${xx_padded}run([0-9]+)$ ]]; then
                    y="${BASH_REMATCH[1]}"
                else
                    continue
                fi

            fi


            # Store Y alongside the directory.
            #
            # This allows us to sort numerically by Y.
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
        # CHECK FOR RUNS
        #######################################################################

        if (( ${#run_dirs[@]} == 0 )); then

            echo "Warning: no runs found for XX=$xx in:" >&2
            echo "  $setup_path" >&2

            continue

        fi


        #######################################################################
        # SORT RUNS NUMERICALLY BY Y
        #######################################################################

        mapfile -t run_dirs < <(
            printf '%s\n' "${run_dirs[@]}" |
            sort -n -k1,1 |
            cut -f2-
        )


        #######################################################################
        # PROCESS EACH RUN
        #######################################################################

        for run_dir in "${run_dirs[@]}"; do

            echo "Adding: $run_dir" >&2

            add_run_frames "$run_dir" "$FPS"

        done

    done
}


###############################################################################
# BUILD COMPLETE FRAME LIST
###############################################################################

echo "============================================================" >&2
echo "Building animation frame list" >&2
echo "============================================================" >&2


###############################################################################
# SETUP 1
###############################################################################

echo >&2
echo "Processing setup1..." >&2

process_setup \
    "$SETUP1_PATH" \
    SETUP1_XX \
    underscore


###############################################################################
# SETUP 2 (OPTIONAL)
###############################################################################

if [[ -n "$SETUP2" ]]; then

    echo >&2
    echo "Processing setup2..." >&2

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
    echo "Error: no PNG frames were found after applying skip list." >&2
    exit 1

fi


FRAME_COUNT=$(grep -c '^file ' "$CONCAT_FILE")


###############################################################################
# SUMMARY
###############################################################################

echo >&2
echo "============================================================" >&2
echo "Frame list complete" >&2
echo "============================================================" >&2
echo "Frames included : $FRAME_COUNT" >&2
echo "Frame rate      : $FPS fps" >&2
echo "Output          : $OUTPUT" >&2
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