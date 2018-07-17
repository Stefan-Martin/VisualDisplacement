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
from Utils import detectBestPoint
import Config

root = Tkinter.Tk() #File open dialog
root.withdraw()

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class VDT: 
    def __init__(self, video_src):
        self.cap = cv2.VideoCapture(video_src)
        self.tracks = []
        self.snapshots=[]
        self.MeasurePts=deque(maxlen=2)
       
    def mouseClick(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            newTrack=deque(maxlen=Config.opts['track_len'])
            self.snapshots.append(deque(maxlen=50)) #add new point track to snapshots
            newTrack.append(detectBestPoint(self.frame_gray,x, y,10)) #find a good point near where the user clicked
            self.tracks.append(newTrack)
            Npts=len(self.tracks)
            print "Added point"+ str(Npts-1)
    
    def CSVexport(self,output):
        with open('TrackData.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(output) 

    def displaceMap(self,deltaP,deltaT,correction,distance,FOV,RES,FPS):
        FOV=np.deg2rad(FOV)
        bFactor=distance*np.sin(FOV)/np.cos(FOV/2.0)
        deltaReal=bFactor/RES*deltaP*correction
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
        for tr,snp in zip(self.tracks,self.snapshots):
            snp.append((framePos,tr[-1])) #append latest video position and point to the snapshot for each track
            deltaRow=[]
            speedRow=[]
            (firstFrame,(firstPoint))=snp[0]
            for frameIDX,point in snp:
                deltaP=0
                if not mode: #initial offset
                    self.mask=cv2.circle(self.mask,(firstPoint),6,(0,255,0),1)
                    self.mask=cv2.line(self.mask, (firstPoint),(point), (0,255,0), 2)
                    deltaP=np.subtract(point,firstPoint)
                    deltaT=frameIDX-firstFrame
                elif mode: #continous delta
                    self.mask=cv2.circle(self.mask,(prev_point),6,(0,255,0),1)
                    self.mask = cv2.line(self.mask, (prev_point),(point), (0,255,100), 2)
                    deltaP=np.subtract(point,prev_point)
                    deltaT=frameIDX-prev_framePos
                if point!=firstPoint:
                    deltaReal,deltaDisplay,speed=self.displaceMap(deltaP,deltaT,**Config.cam_prms)
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
            overlay=np.zeros_like(self.frame)#create drawing mask
            self.MeasurePts.append((x, y))
            for i in self.MeasurePts:
                    x,y=i
                    cv2.circle(overlay, (x, y), 5, (255,0, 0), 2) #draw a circle to mark the point
            if len(self.MeasurePts)>1:
                loc=self.MeasurePts
                delta=np.subtract(loc[1],loc[0])
                _real,deltaDisplay,norm=self.displaceMap(delta,Config.cam_prms["FPS"],**Config.cam_prms)
                print deltaDisplay, norm
                cv2.putText(overlay,"Dx,Dy:"+deltaDisplay+ " Norm:"+ str(round(norm,1)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
                cv2.arrowedLine(overlay,loc[0],loc[1],(255,255,255)) #normLine
                Xpoint=(loc[1][0],loc[0][1])
                cv2.arrowedLine(overlay,loc[0],Xpoint,(0,255,0)) #XLine
                cv2.arrowedLine(overlay,Xpoint,loc[1],(0,0,255)) #YLine
                visM=cv2.add(self.frame,overlay)
                cv2.imshow("Measure",visM)
        elif event== cv2.EVENT_RBUTTONDOWN:
            self.MeasurePts.clear()
            cv2.imshow("Measure",self.frame)

    def run(self):
        paused=False
        fourcc=cv2.VideoWriter_fourcc(*'FMP4') #codec
        downScale=Config.opts["downScale"]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #contrast-limited histogram equalization
        
        #Take first frame
        _ret, self.frame = self.cap.read()
        Config.cam_prms['RES']=map(int,[self.cap.get(3)*downScale,self.cap.get(4)*downScale]) #set resolution based on input video and apply downscaling
        Config.cam_prms['FPS']=self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter("Tracked.avi",fourcc, Config.cam_prms['FPS'], tuple(Config.cam_prms['RES'])) 
        self.frame=cv2.resize(self.frame, (0,0), fx=downScale, fy=downScale) #resize the image
        self.mask=np.zeros_like(self.frame)

        Config.cam_prms['distance']=input("Enter camera distance (mm): ")
        print "Select points to track"
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self.mouseClick)
        while True:
            
            while paused: #paused subroutine halts video feed
                cv2.imshow('frame',vis) #show the last frame
                k = cv2.waitKey(0)
                if k==ord('p'):
                    paused=False 
                elif k==ord('m'):
                    cv2.destroyWindow('frame')
                    cv2.namedWindow('Measure') #opens new window for measuring utility
                    cv2.setMouseCallback('Measure', self.measure)
                    cv2.putText(self.frame, "Select two points to measure, Rightclick to clear", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
                    cv2.imshow("Measure",vis)
                    cv2.waitKey(0) #press any key to exit measure mode
                    cv2.destroyWindow("Measure")
                    cv2.namedWindow('frame')
                    cv2.setMouseCallback('frame', self.mouseClick)

            videoRead,self.frame = self.cap.read() #get next video frame
            if not videoRead: #detect end of video
                break
            timer = cv2.getTickCount() #for computing FPS
            self.frame=cv2.resize(self.frame, (0,0), fx=downScale, fy=downScale) #resize the image to make it more managable for processing
            self.frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            if Config.opts['useCLAHE']:
                frame_track= clahe.apply(self.frame_gray) #histogram equalization to boost contrast
            else: frame_track=self.frame_gray
            vis = self.frame.copy()
            
            if len(self.tracks) > 0: #wait for us to have points before we try to track them
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_track, p0, None, **lk_params) #compute optical flow
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(frame_track, self.prev_gray, p1, None, **lk_params) #back-track to make sure points are nice
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1 # Select good pnts
                
                for idx, (tr, (x, y),good_flag) in enumerate(zip(self.tracks, p1.reshape(-1, 2),good)):
                    if not good_flag: #skips over badly tracked points
                        print "Lost "+"P"+str(idx)
                        del self.snapshots[idx] #delete snapshots associated with these points to avoid breaking measurments due to reindexing
                        del self.tracks[idx] #delete track
                        continue
                    tr.append((x, y)) #add good points to track
                    cv2.circle(vis, (x, y), 2, (0,0, 255), -1) #mark points with filled red circle
                    cv2.putText(vis, "P"+str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA) #display point index
                    window=Config.opts['speedWindow']
                    if len(tr)>window and (tr[-window]!=tr[-1]):
                        diff=np.subtract(tr[-1],tr[-window]) #compute the speed of the point over the last n frames, length determined by window
                        _thing,_curD,curS=self.displaceMap(diff,window,**Config.cam_prms)
                        if curS>0.5:
                            cv2.putText(vis, "S:"+str(round(curS,1)), (x, int(y+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), lineType=cv2.LINE_AA) #print current speed
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (255,255, 255)) #draw a path behind the point
            self.prev_gray = frame_track

            vis=cv2.rectangle(vis,(15,5),(100,30),(0,0,0),-1)
            fps = round(cv2.getTickFrequency() / (cv2.getTickCount() - timer),1)
            vis=cv2.putText(vis, "FPS"+str(fps),(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA) #display FPS
            vis=cv2.add(vis,self.mask)
            cv2.imshow('frame',vis)
            self.out.write(vis) #output video
            k = cv2.waitKey(1) 
            if k ==ord('x'):
                break #exit the program
            elif k==ord('g'):
                self.mask=np.zeros_like(self.frame)
                deltas,speeds=self.grabPoints(Config.opts['deltaMode']) #Grabs points and computes displacements
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
    VDT(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
