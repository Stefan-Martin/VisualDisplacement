'''
Using LK sparse optical flow to compute real-world displacements for tracked points.
User commands:
G rab locations of tracked points
P ause/ resume the video feed
    M (when paused) opens and closes the measurement frame
        When in measure mode, LMB adds a point to measure
        RMB clears measure point
eXit, saving video
E xport deltas to CSV

Options are set in Settings.cfg
'''

import numpy as np
import Tkinter, tkFileDialog
import cv2
import csv
from collections import deque
import ConfigParser
#from Utils import white_balance

root = Tkinter.Tk()
root.withdraw()
config = ConfigParser.SafeConfigParser()
config.read('Settings.cfg')
#inputFile="CameraCalibration.npz"
#npzfile = np.load(inputFile) #load camera parameters

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 1, #of pnts to track
                       qualityLevel = 0.05,
                       minDistance = 7,
                       blockSize = 7 )
cam_prms = {
    "distance":55, #Units: mm
    "FOV":np.array([170,70]),
    "RES":np.array([1920,1080]),
    'FPS':60,
    #'Cmtx':npzfile['Cmtx'],
    #'distortion':npzfile['distortion'] #parameters computed using calibrate.py from the openCV samples
     } 

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class App: 
    def __init__(self, video_src):
        self.cap = cv2.VideoCapture(video_src)
        self.tracks = []
        self.snapshots=[]
        self.fourcc=cv2.VideoWriter_fourcc(*'FMP4') #codec
        self.firstFrame=0
       

    def detectBestPoint(self,x,y,radius):
        msk=np.zeros_like(self.prev_gray)
        msk=cv2.circle(msk,(x,y),radius,(255,0,0),-1)
        points=cv2.goodFeaturesToTrack(self.prev_gray, mask = msk, **feature_params) #select best Shi-Tomasi point in radius around click
        #self.frame=cv2.circle(self.frame, (x,y), 2, (0,0, 255), -1) #blue is click point
        x,y=points[0,0,:2]
        #self.frame=cv2.circle(self.frame, (x,y), 2, (255,0, 0), -1) #red is best point
        #cv2.imshow("Click Ref",self.frame)
        #cv2.waitKey(1)
        return (x,y)

    def mouseClick(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.tracks.append([self.detectBestPoint(x, y,10)])
            Npts=len(self.tracks)
            self.snapshots.append([]) #add new point track to snapshots
            print "Added point"+ str(Npts-1)
    
    def CSVexport(self,output):
        with open('TrackData.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(output) #really bad formatting

    def displaceMap(self,deltaP,deltaT,distance,FOV,RES,FPS):
        FOV=np.deg2rad(FOV)
        bFactor=distance*np.sin(FOV)/np.cos(FOV/2.0)
        deltaReal=bFactor/RES*deltaP*0.91 #correctionfactor 0.91
        deltaReal=tuple(deltaReal)
        deltaDisplay=str((round(deltaReal[0], 1),round(deltaReal[1], 1))) #format for screen
        dist=np.linalg.norm(deltaReal) #compute magnitude of dispalcement vector
        speed=dist*FPS/deltaT
        return deltaReal,deltaDisplay,speed
    
    def grabPoints(self,mode):
        displacements=[]
        speeds=[]
        prev_point=(0,0)
        prev_framePos=0
        framePos=self.cap.get(cv2.CAP_PROP_POS_FRAMES) #get frame index
        if not self.firstFrame:
            self.firstFrame=framePos
        for tr,snp in zip(self.tracks,self.snapshots):
            snp.append((framePos,tr[-1])) #append latest video position and point to the snapshot for each track
            if len(snp)>100:
                del(snp[0])#remove old entries if they get too long, replace with deque?
            deltaRow=[]
            speedRow=[]
            firstPoint=next((x for i, (frame,x) in enumerate(snp) if x), None) #find first non-zero
            for frameIDX,point in snp:
                deltaP=0
                if not mode: #initial offset
                    self.mask=cv2.circle(self.mask,(firstPoint),6,(0,255,0),1)
                    self.mask=cv2.line(self.mask, (firstPoint),(point), (0,255,0), 2)
                    deltaP=np.subtract(point,firstPoint)
                    deltaT=frameIDX-self.firstFrame
                elif mode: #continous delta
                    self.mask=cv2.circle(self.mask,(prev_point),6,(0,255,0),1)
                    #self.mask = cv2.line(self.mask, (prev_point),(point), (0,255,0), 2)
                    deltaP=np.subtract(point,prev_point)
                    deltaT=frameIDX-prev_framePos
                if point!=firstPoint:
                    deltaReal,deltaDisplay,speed=self.displaceMap(deltaP,deltaT,**cam_prms)
                    displayPoint=(int(point[0]),int(point[1]))
                    displayPoint=tuple(np.add(displayPoint,(0,10)))
                    self.mask=cv2.putText(self.mask, "D"+deltaDisplay, displayPoint, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), lineType=cv2.LINE_AA) #display point readout
                    speedRow.append(speed)
                    deltaRow.append(deltaReal)
                prev_point=point
                prev_framePos=frameIDX
            displacements.append(deltaRow)
            speeds.append(speedRow)
        print "Displacements (mm)"
        print('\n'.join([''.join(['{:4}'.format(map(round,tup)) for tup in row]) for row in displacements]))
        print "Speeds (mm/s)"
        print('\n'.join([''.join(['{:5}'.format(round(tup,1)) for tup in row]) for row in speeds]))
        return displacements,speeds
    
    def measure(self,event, x, y, flags, param): 
        if event == cv2.EVENT_LBUTTONDOWN:
            self.MeasurePts.append((x, y))
            self.mask=np.zeros_like(self.frame)#clearOut previous
            for i in self.MeasurePts:
                    x,y=i
                    cv2.circle(self.mask, (x, y), 5, (255,0, 0), 2) #draw a circle to mark the point
            if len(self.MeasurePts)>1:
                loc=self.MeasurePts
                delta=np.subtract(loc[1],loc[0])
                _real,deltaDisplay,norm=self.displaceMap(delta,cam_prms["FPS"],**cam_prms)
                print deltaDisplay, norm
                cv2.putText(self.mask,"Dx,Dy:"+deltaDisplay+ " Norm:"+ str(round(norm,1)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
                cv2.arrowedLine(self.mask,loc[0],loc[1],(255,255,255)) #normLine
                Xpoint=(loc[1][0],loc[0][1])
                cv2.arrowedLine(self.mask,loc[0],Xpoint,(0,255,0)) #XLine
                cv2.arrowedLine(self.mask,Xpoint,loc[1],(0,0,255)) #YLine
        elif event== cv2.EVENT_RBUTTONDOWN:
            self.MeasurePts.clear()
            self.mask=np.zeros_like(self.frame)#clearOut previous
        visM=cv2.add(self.frame,self.mask)
        cv2.imshow("Measure",visM)

    def run(self):
        deltaMode=config.getboolean('RunOptions','deltaMode')
        speedWindow=config.getint('RunOptions','speedWindow')
        downScale=config.getfloat('RunOptions','downScale')
        useCLAHE=config.getboolean('RunOptions','useCLAHE')
        track_len=config.getint('RunOptions','track_len')

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #contrast-limited histogram equalization
        #Take first frame
        _ret, self.frame = self.cap.read()
        cam_prms['RES']=map(int,[self.cap.get(3)*downScale,self.cap.get(4)*downScale]) #set resolution based on input video and apply downscaling
        cam_prms['FPS']=self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter("Tracked.avi",self.fourcc, cam_prms['FPS'], tuple(cam_prms['RES'])) 
        self.frame=cv2.resize(self.frame, (0,0), fx=downScale, fy=downScale) #resize the image
        self.mask=np.zeros_like(self.frame)
        cam_prms['distance']=input("Enter camera distance (mm): ")
        print "Select points to track"
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self.mouseClick)
        #h, w = self.frame.shape[:2]
        #newcameramtx, roi = cv.getOptimalNewCameraMatrix(cam_prms.Cmtx, cam_prms.distortion, (w, h), 1, (w, h)) for distortion compensation
        fps=0
        deltas,speeds=0,0
        paused=False
        while True:
            while paused: 
                cv2.imshow('frame',vis)
                k = cv2.waitKey(0)
                if k==ord('p'):
                    paused=False 
                elif k==ord('m'):
                    cv2.namedWindow('Measure') #opens new window for measuring utility
                    cv2.setMouseCallback('Measure', self.measure)
                    cv2.putText(self.frame, "Select two points to measure, Rightclick to clear", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
                    cv2.imshow("Measure",self.frame)
                    cv2.waitKey(0)
                    cv2.destroyWindow("Measure")
            videoRead,self.frame = self.cap.read()
            if not videoRead:
                break
            #dst = cv.undistort(self.frame, camera_matrix, dist_coefs, None, newcameramtx)
            timer = cv2.getTickCount() #here for computing FPS
            self.frame=cv2.resize(self.frame, (0,0), fx=downScale, fy=downScale) #resize the image to make it more managable for processing
            #self.frame=white_balance(self.frame)
            frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            if useCLAHE:
                frame_track= clahe.apply(frame_gray) #histogram equalization to boost contrast
            else: frame_track=frame_gray
            vis = self.frame.copy()
            if len(self.tracks) > 0: #wait for us to have pnts before we try to track them
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_track, p0, None, **lk_params) #compute optical flow
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(frame_track, self.prev_gray, p1, None, **lk_params) #back-track to see if points are nice
                # Select good pnts
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                #p1 = p1[_st==1] not necessary to check status of points
                new_tracks = []
                for idx, (tr, (x, y),good_flag) in enumerate(zip(self.tracks, p1.reshape(-1, 2),good)):
                    if not good_flag: #skips over badly tracked points
                        print "Lost "+"P"+str(idx)
                        del self.snapshots[idx] #delete snapshots associated with these points to avoid breaking measurments due to reindexing
                        continue
                    tr.append((x, y)) #add good points to track
                    if len(tr) > track_len:
                        del tr[0] #maintains max track length, replace with deque?
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0,0, 255), -1) #mark points with filled red circle
                    cv2.putText(vis, "P"+str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA) #display point index
                    if len(tr)>speedWindow and (tr[-speedWindow]!=tr[-1]):
                        diff=np.subtract(tr[-1],tr[-speedWindow]) #compute the speed of the point over the last n frames, length determined by speedWindow
                        _thing,_curD,curS=self.displaceMap(diff,speedWindow,**cam_prms)
                        cv2.putText(vis, "S:"+str(round(curS,1)), (x, int(y+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), lineType=cv2.LINE_AA) #current speed
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (255,255, 255)) #draw a path behind the point
            else:
                vis=self.frame
            self.prev_gray = frame_track
            vis=cv2.rectangle(vis,(15,5),(100,30),(0,0,0),-1)
            vis=cv2.putText(vis, "FPS"+str(fps),(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA) #display FPS
            vis=cv2.add(vis,self.mask)
            cv2.imshow('frame',vis)
            self.out.write(vis) #output video
            k = cv2.waitKey(1) 
            fps = round(cv2.getTickFrequency() / (cv2.getTickCount() - timer),1)
            if k ==ord('x'):
                break #exit the program
            elif k==ord('g'):
                self.mask=np.zeros_like(self.frame)
                deltas,speeds=self.grabPoints(deltaMode) #Grabs points and computes displacements
            elif k==ord('p'):
                paused=True
            elif k==ord('e'):
                self.CSVexport(deltas) #export displacements to CSV
        self.out.release() #cleanup, very important
        self.cap.release()    

def main():
    try:
        video_src = tkFileDialog.askopenfilename(title = "Select video file",filetypes = (("Video Toast","*.mp4"),("all files","*.*")))
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
