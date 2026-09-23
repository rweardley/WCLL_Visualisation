# render_dataset.py
import sys
from paraview.simple import *

state_file, data_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

LoadState(state_file, data_directory=data_path)

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()

view = GetActiveView()
SaveAnimation(f'{out_dir}/frame_.png', view,
               ImageResolution=[1920, 1080])
