import numpy as np
#Run Options
opts={
"deltaMode":0,
"speedWindow":8,
"downScale":0.8,
"useCLAHE":1,
"track_len":100,
}

cam_prms = {
    "correction":0.91,#correction factor experimentally observed
    "FOV":np.array([170,70]), 
    #'Cmtx':npzfile['Cmtx'],
    #'distortion':npzfile['distortion'] #parameters computed using calibrate.py from the openCV samples
     }
 