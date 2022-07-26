import mediapipe as mp
import cv2
import time
#capturing video

vid=cv2.VideoCapture(0)

#hand detection
handsmp=mp.solutions.hands
hands=handsmp.Hands()

#for drawing 21 points
mpdraw=mp.solutions.drawing_utils

#frame rate
ctime=0
ptime=0
while True:
    signal,img=vid.read()
    RGBimg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hands.process(RGBimg)
    if(result.multi_hand_landmarks):
        for handlm in result.multi_hand_landmarks:
            #for pointing hand line
            mpdraw.draw_landmarks(img,handlm,handsmp.HAND_CONNECTIONS)

            #detecting individual points in hand
            for id,lm in enumerate(handlm.landmark):
                # print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                # print(id,cx,cy)
                if(id==0):
                    cv2.circle(img,(cx,cy),15,(200,100,50),cv2.FILLED)

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,str(int(fps)),(550,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
    cv2.waitKey(1)
    cv2.imshow("image",img)