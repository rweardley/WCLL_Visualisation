# pvbatch script for resampling NEK5000 data
from paraview.simple import *
import time
import gc

def nekrs_resample_to_image(input_file, output_file, full_sampling_dimensions, ts_idx, filter_domains, spectralIDs, domainNames, input_bounds=None):

    t_init = time.time()

    print(f"[{ts_idx}] >>> Checkpoint: 1, time={time.time()-t_init:.2f} s")

    # File path and output configuration

    print(input_file)
    print(output_file)
    print(full_sampling_dimensions)
    print(spectralIDs)
    print(domainNames)

    print(f"[{ts_idx}] >>> Checkpoint: 2, time={time.time()-t_init:.2f} s")

    # Load NekRS data
    nek5000_data = Nek5000Reader(FileName=input_file)
    if filter_domains:
        nek5000_data.AddSpectralElementIdsasCellData = 1

    print(f"[{ts_idx}] >>> Checkpoint: 3, time={time.time()-t_init:.2f} s")

    # Update pipeline to execute data loading for current timestep
    nek5000_data.UpdatePipeline(time=nek5000_data.TimestepValues[ts_idx])

    gc.collect()

    if input_bounds:
        full_domain_bounds = input_bounds
    else:
        full_domain_bounds = nek5000_data.GetDataInformation().GetBounds()
    print(f"[{ts_idx}] >>> full_domain_bounds={full_domain_bounds}")

    for domain in range(len(domainNames)):
        if filter_domains:
            if domain == 0:
                print(f"[{ts_idx}] >>> Commencing loop through domains")
            threshold = Threshold(registrationName=f"Threshold_{ts_idx}_{domain}", Input=nek5000_data)
            print(f"[{ts_idx}] >>> Domain {domain}: {domainNames[domain]}, time={time.time()-t_init:.2f} s")
            print(f"[{ts_idx}] >>> Filtering by Spectral Element ID, time={time.time()-t_init:.2f} s")
            threshold.Set(
                Scalars=["CELLS", "spectral element id"],
                LowerThreshold=spectralIDs[domain] + 1,
                UpperThreshold=spectralIDs[domain + 1],
            )
            threshold.UpdatePipeline()

            # Get domain SamplingBounds and SamplingDimensions
            domain_bounds = threshold.GetDataInformation().GetBounds()
            x_ratio = abs((domain_bounds[1] - domain_bounds[0]) / (
                full_domain_bounds[1] - full_domain_bounds[0]
            ))
            print(f"[{ts_idx}] >>> x_ratio={x_ratio}")
            y_ratio = abs((domain_bounds[3] - domain_bounds[2]) / (
                full_domain_bounds[3] - full_domain_bounds[2]
            ))
            print(f"[{ts_idx}] >>> y_ratio={y_ratio}")
            z_ratio = abs((domain_bounds[5] - domain_bounds[4]) / (
                full_domain_bounds[5] - full_domain_bounds[4]
            ))
            print(f"[{ts_idx}] >>> z_ratio={z_ratio}")
            domain_sampling_dimensions = [
                int(full_sampling_dimensions[0] * x_ratio),
                int(full_sampling_dimensions[1] * y_ratio),
                int(full_sampling_dimensions[2] * z_ratio),
            ]
            print(f"[{ts_idx}] >>> domain_sampling_dimensions={domain_sampling_dimensions}")
            print(f"[{ts_idx}] >>> domain_bounds={domain_bounds}")

            print(f"[{ts_idx}] >>> Checkpoint: 4 ({domain}), time={time.time()-t_init:.2f} s")
            print(f"[{ts_idx}] >>> Resample to Image")

            # Apply Resample To Image filter
            resample = ResampleToImage(registrationName=f"Resample_{ts_idx}_{domain}", Input=threshold)
            resample.Set(
                UseInputBounds=0,
                SamplingBounds=domain_bounds,
                SamplingDimensions=domain_sampling_dimensions,
            )
        else:
            if domain == 0:
                resample = ResampleToImage(Input=nek5000_data)
                resample.Set(
                    UseInputBounds=1,
                    SamplingDimensions=full_sampling_dimensions,
                )
            else:
                continue

        print(f"[{ts_idx}] >>> Checkpoint: 5 ({domain}), time={time.time()-t_init:.2f} s")

        # Update pipeline for resampling
        resample.UpdatePipeline()

        print(f"[{ts_idx}] >>> Checkpoint: 6 ({domain}), time={time.time()-t_init:.2f} s")

        # Save output as .vti (VTK ImageData format)
        output_filename = f"{output_file}_"
        if filter_domains:
            output_filename += f"{domainNames[domain]}_"
        output_filename += f'{ts_idx:03d}.pvti'
        SaveData(output_filename, proxy=resample)

        # Clean up
        Delete(resample)
        del resample
        if filter_domains:
            Delete(threshold)
            del threshold
        gc.collect()
        print(f"[{ts_idx}] >>> Checkpoint: 7 ({domain}), time={time.time()-t_init:.2f} s")

    print(f"[{ts_idx}] >>> Checkpoint: 8, time={time.time()-t_init:.2f} s")
    gc.collect()
