import mediapipe as mp
import cv2
import time
import hand_tracking_module as htm

vid=cv2.VideoCapture(0)

detector=htm.HandDetector(det_conf=0.7)



while True:
    success, img = vid.read()
    img = detector.find_hands(img)
    lm_list = detector.find_landmark(img,draw=False)

    if(len(lm_list)!=0):
        #we need coordinates of four fingers->1,2,3,4

        #finger 1
        x1_up,y1_up=lm_list[8][1],lm_list[8][2]
        x1_middle,y1_middle=lm_list[7][1],lm_list[7][2]

        #finger 2
        x2_up,y2_up=lm_list[12][2],lm_list[12][2]
        x2_middle,y2_middle=lm_list[11][2],lm_list[11][2]

        #finger 3
        x3_up,y3_up=lm_list[16][2],lm_list[16][2]
        x3_middle,y3_middle=lm_list[15][2],lm_list[15][2]

        #finger 4
        x4_up,y4_up=lm_list[20][2],lm_list[20][2]
        x4_middle,y4_middle=lm_list[19][2],lm_list[19][2]

        if(y1_middle>y1_up and y2_middle>y2_up and y3_middle>y3_up and y4_middle>y4_up):
            print("paper")
            # time.sleep(5)
        elif (y1_middle<y2_up and y2_middle<y2_up and y3_middle<y3_up and y4_middle<y4_up):
            print("stone")
            # time.sleep(5)
        else:
            print("scissor")
    cv2.waitKey(1)
    cv2.imshow("image", img)
