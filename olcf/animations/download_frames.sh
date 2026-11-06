#!/bin/bash

source="riker.olcf.ornl.gov:/lustre/orion/fus166/proj-shared/rweb/gb_final/animation/"

cases=(
    # "buo0_mhd0_mc1/vol_fluids_velocity_all_jet_split/"
    # "buo0_mhd1_mc1/vol_fluids_velocity_all_jet_split/"
    # "buo0_mhd0_mc1_stride8/vol_fluids_velocity_all_jet_split/"
    # "buo0_mhd0_mc1_timeRescaled_stride8/vol_fluids_velocity_all_jet_split/"
    "buo0_mhd1_mc1_stride8/vol_fluids_velocity_all_jet_split/"
)


for case in "${cases[@]}"; do
    rsync -av $source/$case $case
done