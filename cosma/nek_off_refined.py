# pvbatch script for resampling NEK5000 data
from paraview.simple import *
import time

t_init=time.time()

print(f"Checkpoint: 1, time={time.time()-t_init:.2f} s")

# File path and output configuration

input_file = "../../../globus/csd3/run3_mhd_off_N7/pink.nek5000"
#output_file = input_file.replace("pink.nek5000", "pink_resampled_2850_700_4760.pvti")
output_file = input_file.replace("pink.nek5000", "pink_resampled_400_400_400.pvti")

print(f"Checkpoint: 2, time={time.time()-t_init:.2f} s")

# Load NEK5000 data
nek5000_data = Nek5000Reader(FileName=input_file)

print(f"Checkpoint: 3, time={time.time()-t_init:.2f} s")

# Update pipeline to get metadata (e.g., bounds)
nek5000_data.UpdatePipeline()

print(f"Checkpoint: 4, time={time.time()-t_init:.2f} s")

# Apply Resample To Image filter
resample = ResampleToImage(Input=nek5000_data)
resample.UseInputBounds = 1
#resample.SamplingDimensions = [2850, 700, 4760]
resample.SamplingDimensions = [400, 400, 400]

print(f"Checkpoint: 5, time={time.time()-t_init:.2f} s")

# Update pipeline for resampling
resample.UpdatePipeline()

print(f"Checkpoint: 6, time={time.time()-t_init:.2f} s")


# Save output as .vti (VTK ImageData format)
SaveData(output_file, proxy=resample, Writealltimestepsasfileseries=1, Filenamesuffix='_%.3d')

print(f"Checkpoint: 7, time={time.time()-t_init:.2f} s")

