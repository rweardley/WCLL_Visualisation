run_number_to_checkpoint_interval = [
    5.0e-5,     # 00run
    5.0e-5,     # 01run
    5.0e-5,     # 02run
    5.0e-5,     # 03run
    5.0e-5,     # 04run
    5.0e-5,     # 05run
    1.6e-4,     # 06run
    1.7e-4,     # 07run
    2.5e-4,     # 08run
    4.2e-4,     # 09run
    4.2e-4,     # 10run
    4.2e-4,     # 11run
    4.2e-4,     # 12run
    4.2e-4,     # 13run
]

reference_frame_rate = 12
reference_run_number = 10

def stride_from_run(run):
    if run >= 6:
        return 2
    else:
        return 8

for run in range(len(run_number_to_checkpoint_interval)):
    stride = stride_from_run(run)
    reference_interval = run_number_to_checkpoint_interval[
        reference_run_number
    ] * stride_from_run(reference_run_number)
    interval = run_number_to_checkpoint_interval[run] * stride
    frame_rate = reference_frame_rate * (
        reference_interval / interval
    )
    print(f"{run:02}run: {frame_rate}")
