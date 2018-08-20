import numpy as np
inputFile="SJ5000XCalibration.npz"
npzfile = np.load(inputFile) #load camera parameters
#Run Options
opts={
"deltaMode":0,
"speedWindow":3,
"downScale":0.8,
"useCLAHE":1,
"track_len":100,
}

cam_prms = {
    "correction":.91,#correction factor experimentally observed
    "FOV":np.array([170,70]), 
     }
distortComp={
    'Cmtx':npzfile['Cmtx'],
    'distortion':npzfile['distortion'] #parameters computed using calibrate.py 
}
 