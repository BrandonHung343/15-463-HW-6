As usual, brief description of functions and what part/question they correspond to are in the assgn6.py file.
The write-up explains in detail what goes on under the hood, while this readme will explain some nuances. 

spatial_temporal_shadow_edge(path, pxSpeed, small=False) detects the spatial and temporal right shadow edges.
It saves them in the src directory as hlines.npy, vlines.npy, and tLines.npy. In this case, pass the included
directory '../data/frog/v1shad/' as path. Ignore pxSpeed, as it is no longer used.
   
load_calibration(path) returns the transformation vectors and rotation matrices in tH, RH, tV, RV order. 
Note that it assumes you run the calibration demo first, so I've included the calibration information. 
Make sure to set path = '../data/frog/v1/'

shadow_line_calib(dPath, extPath, hSplit, rows) performs shadow line calibration. It stores the 3D points
as vPoints3D.npy and hPoints3D.npy, in src. 
Make sure to set dPath='', extPath='../data/frog/v1/', hSplit=292, rows=768.

shadow_plane_calib(path) performs shadow plane calibration. It stores the planes as shadowPlanes.npy. 
Make sure to set path=''

reconstruction(dPath, cropX, cropY, extPath, tStart, refImage) performs and plots a reconstruction of the image.
For the frog, set dPath = '', cropX = np.arange(285, 750), cropY = np.arange(300, 630), extPath = '../data/frog/v1/',
tStart = 64, refImage). Uncomment the ref image line above. This one doesn't save anything.


