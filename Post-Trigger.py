import numpy as np
import cv2
from collections import deque

class App: 
    def __init__(self, video_src): 
        self.cap = cv2.VideoCapture(video_src)
        self.fourcc=cv2.VideoWriter_fourcc(*'FMP4') #codec
        self.buffer=deque(maxlen=300) #frame buffer, don't set this above 15000
        self.RES=() #output resolution

    def mouseClick(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print "Click"
    def postCapture(self,buffer):
        self.cap.release()
        out = cv2.VideoWriter('output.avi',self.fourcc, 30, self.RES)
        for frame in buffer:
            out.write(frame)
        out.release
        cv2.destroyAllWindows()

    def run(self):
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self.mouseClick)
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        self.RES=(frame_width,frame_height)
        while True:
            _ret,self.frame = self.cap.read()
            cv2.imshow('frame',self.frame)
            k = cv2.waitKey(1)
            self.buffer.append(self.frame)
            if k ==ord('x'):
                self.postCapture(self.buffer)
                break
        self.cap.release()

def main():
    #import sys
    try:
        video_src = 0
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
