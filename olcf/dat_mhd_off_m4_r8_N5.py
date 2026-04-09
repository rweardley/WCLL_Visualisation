# pvbatch script for resampling NEK5000 data
from paraview.simple import *
import time
import gc

t_init = time.time()

print(f">>> Checkpoint: 1, time={time.time()-t_init:.2f} s")

# File path and output configuration

input_file = "/lustre/orion/fus166/proj-shared/ylan/vis_rupert_gb26/dat_mhd_off_m4_r8_N5/pink.nek5000"
output_file = "/lustre/orion/fus166/proj-shared/rweb/dat_mhd_off_m4_r8_N5_x400"
full_sampling_dimensions = [11097, 400, 1042]
point_arrays = None
#point_arrays = ["Velocity Magnitude"]
spectralIDs = [-1, 38733372, 140869280, 151935592, 245637464, 257676220]
domainNames = ["water_TBM", "PbLi", "water_shield", "solid_TBM", "solid_shield"]
process_timesteps = [0, 5, 10, 14]

print(f">>> Checkpoint: 2, time={time.time()-t_init:.2f} s")

# Load NekRS data
nek5000_data = Nek5000Reader(FileName=input_file)
if point_arrays:
    nek5000_data.PointArrays = point_arrays
nek5000_data.AddSpectralElementIdsasCellData = 1
full_domain_bounds = nek5000_data.GetDataInformation().GetBounds()

print(f">>> Checkpoint: 3, time={time.time()-t_init:.2f} s")

print(">>> Commencing loop through timesteps")

for ts_idx in range(len(nek5000_data.TimestepValues)):
    print(f">>> Processing timestep {ts_idx} ({ts_idx + 1}/{len(nek5000_data.TimestepValues)})")
    if ts_idx not in process_timesteps:
        print(">>> Skipping timestep {ts_idx}")
    else:

        # Update pipeline to execute data loading for current timestep
        nek5000_data.UpdatePipeline(time=nek5000_data.TimestepValues[ts_idx])

        # Loop through domains and resample each
        print(">>> Commencing loop through domains")

        for domain in range(len(domainNames)):
            threshold = Threshold(registrationName="Threshold", Input=nek5000_data)
            print(f">>> Domain {domain}: {domainNames[domain]}, time={time.time()-t_init:.2f} s")
            print(f">>> Filtering by Spectral Element ID, time={time.time()-t_init:.2f} s")
            threshold.Set(
                Scalars=["CELLS", "spectral element id"],
                LowerThreshold=spectralIDs[domain]+1,
                UpperThreshold=spectralIDs[domain + 1],
            )
            threshold.UpdatePipeline()

            # Get domain SamplingBounds and SamplingDimensions
            domain_bounds = threshold.GetDataInformation().GetBounds()
            x_ratio = abs((domain_bounds[1] - domain_bounds[0]) / (
                full_domain_bounds[1] - full_domain_bounds[0]
            ))
            y_ratio = abs((domain_bounds[3] - domain_bounds[2]) / (
                full_domain_bounds[3] - full_domain_bounds[2]
            ))
            z_ratio = abs((domain_bounds[5] - domain_bounds[4]) / (
                full_domain_bounds[5] - full_domain_bounds[4]
            ))
            domain_sampling_dimensions = [
                int(full_sampling_dimensions[0] * x_ratio),
                int(full_sampling_dimensions[1] * y_ratio),
                int(full_sampling_dimensions[2] * z_ratio),
            ]

            print(f">>> Checkpoint: 4 ({domain}), time={time.time()-t_init:.2f} s")
            print(">>> Resample to Image")

            # Apply Resample To Image filter
            resample = ResampleToImage(Input=threshold)
            resample.Set(
                UseInputBounds=0,
                SamplingBounds=domain_bounds,
                SamplingDimensions=domain_sampling_dimensions,
            )

            print(f">>> Checkpoint: 5 ({domain}), time={time.time()-t_init:.2f} s")

            # Update pipeline for resampling
            resample.UpdatePipeline()

            print(f">>> Checkpoint: 6 ({domain}), time={time.time()-t_init:.2f} s")

            # Save output as .vti (VTK ImageData format)
            output_filename = f"{output_file}_{domainNames[domain]}_"
            if point_arrays:
                output_filename += '_'.join(point_arrays).replace(" ", "")
            output_filename += f'{ts_idx:03d}.pvti'
            SaveData(output_filename, proxy=resample)

            # Clean up
            Delete(resample)
            del resample
            Delete(threshold)
            del threshold
            gc.collect
    gc.collect

print(f">>> Checkpoint: 7, time={time.time()-t_init:.2f} s")
gc.collect
