import sys, glob, re, os, time
from datetime import timedelta
from paraview.simple import *
from paraview import servermanager

state_file, data_dir, mesh_path, out_dir = sys.argv[1:5]

# --- rank detection (for when this runs under mpirun/srun) ---
pm = servermanager.vtkProcessModule.GetProcessModule()
rank = pm.GetPartitionId()

t_start = time.time()


def log(msg):
    if rank == 0:
        elapsed = timedelta(seconds=int(time.time() - t_start))
        print(f"[{elapsed}] {msg}", flush=True)


log(f"Starting: state={state_file} data_dir={data_dir} mesh={mesh_path}")

LoadState(state_file)
log("State loaded")


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


pvti_readers = []

for (name, sid), proxy in GetSources().items():
    cls = proxy.__class__.__name__

    if cls == "XMLPartitionedImageDataReader":
        old_file = list(proxy.FileName)[0]
        base = os.path.basename(old_file)

        m = re.match(r"^(.*?)(\d+)(\.pvti)$", base)
        if not m:
            raise RuntimeError(f"Can't infer series pattern from '{base}'")
        prefix, _, ext = m.groups()

        series = sorted(
            glob.glob(os.path.join(data_dir, f"{prefix}*{ext}")),
            key=natural_key,
        )
        if not series:
            raise RuntimeError(
                f"No files matched {prefix}*{ext} in {data_dir}"
            )

        proxy.FileName = series
        proxy.UpdatePipelineInformation()
        pvti_readers.append((name, proxy))
        log(
            f"Reader '{name}': found {len(series)} files matching '{prefix}*{ext}'"
        )

    elif cls == "STLReader":
        proxy.FileNames = [mesh_path]
        proxy.UpdatePipelineInformation()
        log(f"STL reader '{name}' set to {mesh_path}")

# sanity check: all three series should agree on frame count
counts = {name: len(p.TimestepValues) for name, p in pvti_readers}
if len(set(counts.values())) > 1:
    log(f"WARNING: mismatched timestep counts across readers: {counts}")
else:
    log(f"Timestep counts match across readers: {counts}")

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()

view = GetActiveView()

timesteps = list(scene.TimeKeeper.TimestepValues)
nframes = len(timesteps)
log(f"Animation set up: {nframes} frames to render")

os.makedirs(out_dir, exist_ok=True)

# view.ViewSize = [1920, 1080]

frame_start = time.time()
for i, t in enumerate(timesteps):
    f0 = time.time()

    scene.TimeKeeper.Time = t
    SaveScreenshot(
        os.path.join(out_dir, f"frame_{i:04d}.png"),
        view,
        ImageResolution=[1920, 1080],
    )

    frame_time = time.time() - f0
    avg = (time.time() - frame_start) / (i + 1)
    remaining = timedelta(seconds=int(avg * (nframes - i - 1)))
    log(
        f"Frame {i+1}/{nframes} done (t={t:.4g}, {frame_time:.1f}s, ETA {remaining})"
    )

log(f"All done. Total time: {timedelta(seconds=int(time.time() - t_start))}")
