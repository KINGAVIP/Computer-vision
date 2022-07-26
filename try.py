import time
import cv2
import mediapipe as mp
import hand_tracking_module as htm

ctime=0
ptime=0
vid = cv2.VideoCapture(0)
detector=htm.HandDetector()
while True:
    signal, img = vid.read()
    img=detector.find_hands(img)
    lmlist=detector.find_landmark(img)
    if(len(lmlist)!=0):
        print(lmlist[3])
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (550, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
    cv2.waitKey(1)
    cv2.imshow("image", img)
