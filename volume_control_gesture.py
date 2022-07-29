import mediapipe as mp
import cv2
import hand_tracking_module as htm
import math
import time
import numpy as np

#volume controller in python ->pycaw module
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


#directly using these initializations for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

#web cam object
vid=cv2.VideoCapture(0)

#for setting width of our web cam
# vid.set(cv2.CAP_PROP_FRAME_WIDTH,220)
#or
# vid.set(3,600)

#for setting height
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT,400)
# or
# vid.set(4,400)

# vid.set(3,200)
# vid.set(4,300)


#object of our model
detector=htm.HandDetector()
dist=0
while True:
    success,img=vid.read()
    img=detector.find_hands(img)
    # if ha nd.multi_hand_landmark:
    #     for id,lm in hand.multi_hand_landmark:
    #         w,h,c=img.shape
    #         cx,cy=lm.x*w,lm.y*h
    #         cv2.putText(img,str(id),(cx,cy),2,3,(0,20,100),3)
    # for i in range(0,20):
    lmlist=detector.find_landmark(img,draw=False)
    if(len(lmlist)!=0):
        x1,y1=lmlist[4][1],lmlist[4][2]
        x2,y2=lmlist[8][1],lmlist[8][2]
        cv2.circle(img,(x1,y1),5,(0,0,0),3)
        cv2.circle(img,(x2,y2),5,(0,0,0),3)
        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
        dist=math.sqrt((x2-x1)**2+(y2-y1)**2)

        #hand range->20-250
        cv2.putText(img,str(int(dist)),(100,100),1,2,(0,100,150),5)

    # get current volume
    currentVolumeDb = volume.GetMasterVolumeLevel()


    print(currentVolumeDb)
    #vol has a range from -96.0 to 0.0
    vol=np.interp(dist,[20,250],[-96,0])
    volrange=np.interp(dist,[50,300],[400,150])
    # print(vol)
    volume.SetMasterVolumeLevel(vol,None)
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volrange)),(85,400),(0,0,0),cv2.FILLED)
    cv2.waitKey(1)
    cv2.imshow("image",img)
