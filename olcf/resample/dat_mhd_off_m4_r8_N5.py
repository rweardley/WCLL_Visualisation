# pvbatch script for resampling NEK5000 data
import sys
import resample

# Get command-line arguments
ts_idx = int(sys.argv[1])
print(f"[{ts_idx}] >>> Timestep index: {ts_idx}")
filter_domains = bool(sys.argv[2])
print(f"[{ts_idx}] >>> Filter domains: {filter_domains}")

# File path and output configuration
input_file = "/lustre/orion/fus166/proj-shared/ylan/vis_rupert_gb26/dat_mhd_off_m4_r8_N5/pink.nek5000"
output_file = "/lustre/orion/fus166/proj-shared/rweb/dat_mhd_off_m4_r8_N5_x400"
full_sampling_dimensions = [11097, 400, 1042]
spectralIDs = [-1, 38733372, 140869280, 151935592, 245637464, 257676220]
domainNames = ["water_TBM", "PbLi", "water_shield", "solid_TBM", "solid_shield"]

resample.nekrs_resample_to_image(input_file, output_file, full_sampling_dimensions, ts_idx, filter_domains, spectralIDs, domainNames)