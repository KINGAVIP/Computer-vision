import mediapipe as mp
import cv2
import time
#hand tracking input using open cv
vid=cv2.VideoCapture(0)

#hand detection class object present in media pipe

mphands=mp.solutions.hands
hands=mphands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)

#drawing points on hand detect
mpdraw=mp.solutions.drawing_utils

#finding fps
pt=0 #previous time
ct=0 #current time
while True:
    success,img=vid.read()
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(img,handlms,mphands.HAND_CONNECTIONS)
#calculating fps
    ct=time.time()
    fps=1/(ct-pt)
    pt=ct
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,222),3)
    # for hand_landmarks_all in result.multi_hand_landmarks:
    #     print(hand_landmarks_all)
    cv2.imshow('image',img)
    cv2.waitKey(1)


