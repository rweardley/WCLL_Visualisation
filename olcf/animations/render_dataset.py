import sys, glob, re, os
from paraview.simple import *

state_file, data_dir, mesh_path, out_dir = sys.argv[1:5]

LoadState(state_file)  # no data_directory - we'll set every reader explicitly


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

    elif cls == "STLReader":
        proxy.FileNames = [mesh_path]
        proxy.UpdatePipelineInformation()

# sanity check: all three series should agree on frame count
counts = {name: len(p.TimestepValues) for name, p in pvti_readers}
if len(set(counts.values())) > 1:
    print(f"WARNING: mismatched timestep counts across readers: {counts}")

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()

view = GetActiveView()
SaveAnimation(f"{out_dir}/frame_.png", view, ImageResolution=[1920, 1080])
