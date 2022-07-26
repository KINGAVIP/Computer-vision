import cv2
import mediapipe as mp
import time
class HandDetector():
    def __init__(self,mode=False,max_hands=2,model_complexity=1,det_conf=0.5,track_conf=0.5):
        self.mode=mode
        self.model_complexity=model_complexity
        self.max_hands=max_hands
        self.det_conf=det_conf
        self.track_conf=track_conf

        self.handsmp = mp.solutions.hands
        self.hands =self.handsmp.Hands(
            self.mode,self.max_hands,self.model_complexity,self.det_conf,self.track_conf)
        self.mpdraw = mp.solutions.drawing_utils

        # class handDetector():
        #     def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        #         self.mode = mode
        #         self.maxHands = maxHands
        #         self.detectionCon = detectionCon
        #         self.trackCon = trackCon
        #         self.mpHands = mp.solutions.hands
        #         self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        #                                         self.detectionCon, self.trackCon)
        #         self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self,img,draw=True):
        RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(RGBimg)
        if (self.result.multi_hand_landmarks):
            for handlm in self.result.multi_hand_landmarks:
                # for pointing hand line
                if draw:
                    self.mpdraw.draw_landmarks(img, handlm, self.handsmp.HAND_CONNECTIONS)
        return img

    def find_landmark(self,img,handNo=0,draw=True):
        lmlist = []
        if self.result.multi_hand_landmarks:
            myhand=self.result.multi_hand_landmarks[handNo]

                # for pointing hand line
                # mpdraw.draw_landmarks(img, handlm, handsmp.HAND_CONNECTIONS)

                # detecting individual points in hand
            for id, lm in enumerate(myhand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id,cx,cy)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (200, 100, 50), cv2.FILLED)
        return lmlist
def main():
    ctime=0
    ptime=0
    vid = cv2.VideoCapture(0)
    detector=HandDetector()
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

if __name__ == '__main__':
    main()