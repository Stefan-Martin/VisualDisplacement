import cv2 as cv
import numpy as np
from glob2 import glob
from matplotlib import pyplot as plt
from tkinter import filedialog

img_mask = 'Demos/UI/headrest*.png'  # default
search_path=filedialog.askopenfilename(title = "Select search image",filetypes = (("Image Toast","*.png"),("all files","*.*")),initialdir="Demos/UI")
template = cv.imread(search_path) #do not compress template otherwise color space is skewed
chans=cv.split(template)
colors = ("b", "g", "r")
templateChans=[]

for (chan, color) in zip(chans, colors):
    hist = cv.calcHist([chan], [0], None, [256], [0, 256])/(chan.size)
    templateChans.append(hist)
	# plot the histogram
    #plt.plot(hist, color = color)
    #plt.xlim([0, 256])
#plt.show()
template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
#hist_Template = cv.calcHist([template], [0], None, [265], [0, 256])
#plt.figure()
img_names = glob(img_mask)

def checkImage(fn):
    colorMatch=True
    img = cv.imread(fn)
    #print("Processing...",fn)
    if img is None:
        print("Failed to load", fn)
        return None
    vis = img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    w, h = template.shape[::-1]

    # Apply template Matching
    res = cv.matchTemplate(img,template,cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    if (max_val>0.9):
        mask=np.zeros_like(img)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(mask,top_left, bottom_right, 255, -1)
        chans=cv.split(vis)
        plt.title("%s" % fn)
        diffs=[]
        for (chan, color,tempHist) in zip(chans, colors,templateChans):
            hist = cv.calcHist([chan], [0], mask, [256], [0, 256])/(cv.countNonZero(mask))
            diff=abs(hist-tempHist)
            plt.plot(diff, color = color)
            diffs.append(diff)
            match=not any(t>.1 for t in diff)
            plt.xlim([0, 256])
            colorMatch=match and colorMatch
        #plt.show()
        #print ("Colormatch:",colorMatch)
        cv.rectangle(vis,top_left, bottom_right, (0,255,0), 5)
        print("Found template in %s" % fn)
    else:
        print ("Template not found %s" % fn)
        cv.putText(vis, "Template not found", (30, 100), cv.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), lineType=cv.LINE_AA, thickness=6) 
    if not colorMatch:
        print ("Color Match Failed %s" % fn)
        cv.putText(vis, "Wrong Color", (30, 100), cv.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), lineType=cv.LINE_AA, thickness=6)
    return vis        

threadn = 1#cv.getNumberOfCPUs()
print("Run with %d threads..." % threadn)
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(threadn)
if threadn==1:
    detections=[checkImage(i) for i in img_names]
    cv.waitKey(1000)
#detections = pool.map(checkImage, img_names)
cv.namedWindow("Results")
for i in detections:
    i=cv.resize(i, (0,0), fx=0.5, fy=0.5) #resize the image
    cv.imshow("Results",i)
    cv.waitKey(0)