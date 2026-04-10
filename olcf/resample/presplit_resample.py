import sys
import resample

# Get command-line arguments
ts_idx = int(sys.argv[1])
print(f"[{ts_idx}] !!! Timestep index: {ts_idx}")
filter_domains = bool(sys.argv[2])
print(f"[{ts_idx}] !!! Filter domains: {filter_domains}")
input_file = str(sys.argv[3])
print(f"[{ts_idx}] !!! Input file: {input_file}")
output_file = str(sys.argv[4])
print(f"[{ts_idx}] !!! Output file: {output_file}")

spectralIDs = None
domainNames = ["PRESPLIT"]

# calculate full-domain sampling resolution

input_bounds = [-0.461538, 37.1299, -5.63846, 5.63846, -14.6846, 14.6846]
input_ranges = [
    abs(input_bounds[1] - input_bounds[0]),
    abs(input_bounds[3] - input_bounds[2]),
    abs(input_bounds[5] - input_bounds[4]),
]
min_range = min(input_ranges)
res = 400
full_sampling_dimensions = [
    int(res * input_range / min_range) for input_range in input_ranges
]

print(f"[{ts_idx}] !!! Sampling dimensions for res={res}: {full_sampling_dimensions}")

resample.nekrs_resample_to_image(
    input_file,
    output_file,
    full_sampling_dimensions,
    ts_idx,
    filter_domains,
    spectralIDs,
    domainNames,
    input_bounds,
)
