# para_example.py:
from paraview.simple import *

# Add a polygonal sphere to the 3D scene
s = Sphere()
s.ThetaResolution = 128                        # Number of theta divisions (longitude lines)
s.PhiResolution = 128                          # Number of phi divisions (latitude lines)

# Convert Proc IDs to scalar values
p = ProcessIdScalars()                         # Apply the ProcessIdScalars filter to the sphere

display = Show(p)                              # Show data
curr_view = GetActiveView()                    # Retrieve current view

# Generate a colormap for Proc Id's
cmap = GetColorTransferFunction("ProcessId")   # Generate a function based on Proc ID
cmap.ApplyPreset('Viridis (matplotlib)')       # Apply the Viridis preset colors
#print(GetLookupTableNames())                  # Print a list of preset color schemes

# Set Colorbar Properties
display.SetScalarBarVisibility(curr_view,True) # Show bar
scalarBar = GetScalarBar(cmap, curr_view)      # Get bar's properties
scalarBar.WindowLocation = 'Any Location'       # Allows free movement
scalarBar.Orientation = 'Horizontal'           # Switch from Vertical to Horizontal
scalarBar.Position = [0.15,0.80]               # Bar Position in [x,y]
scalarBar.LabelFormat = '%.0f'                 # Format of tick labels
scalarBar.RangeLabelFormat = '%.0f'            # Format of min/max tick labels
scalarBar.ScalarBarLength = 0.7                # Set length of bar

# Render scene and save resulting image
Render()
SaveScreenshot('pvbatch-test.png',ImageResolution=[1080, 1080])