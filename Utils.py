import cv2 
import numpy as np
# params for ShiTomasi corner detection
feature_params = dict(qualityLevel = 0.05,
                       minDistance = 7,
                       blockSize = 7 )

def white_balance(img):
    result = cv2.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv.COLOR_LAB2BGR)
    return result

def detectBestPoint(img_gray,x,y,radius):
        msk=np.zeros_like(img_gray)
        msk=cv2.circle(msk,(x,y),radius,(255,0,0),-1)
        points=cv2.goodFeaturesToTrack(img_gray, mask = msk, maxCorners=1,**feature_params) #select best Shi-Tomasi point in radius around click
        #self.frame=cv2.circle(self.frame, (x,y), 2, (0,0, 255), -1) #blue is click point
        x,y=points[0,0,:2]
        #self.frame=cv2.circle(self.frame, (x,y), 2, (255,0, 0), -1) #red is best point
        #cv2.imshow("Click Ref",self.frame)
        #cv2.waitKey(1)
        return (x,y)