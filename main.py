# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import time
import mediapipe as np
import os
import numpy as np
import handtrackingmodule as htm
import cv2


##########
brushThikness  = 10
eraserThikness = 50
xp, yp = 0, 0
########
folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[1]
drawcolor = (255, 0, 0)

cap = cv2.VideoCapture(0)

cap.set(3,1920)
cap.set(4,1080)
detector = htm.handDetector(detectionCon=0.85)
imgCanvas = np.zeros((1080,1920,3), np.uint8)
while True:
    #1.import Image

    success,img = cap.read()
    img = cv2.flip(img,1)

    #2.find out the land mark

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
        #print(lmList)

        # tios of index and middle finger
        x1,y1 = lmList[8][1:]   # the second and third elements are important so first omitted
        x2,y2 = lmList[12][1:]  # middle finger


        #3 check which fingers are up

        fingers = detector.fingersUp()
        print(fingers)
        #4. if selection mode-when two finger are up

        if fingers[1] and fingers[2]:
            print('selection mode')
            xp, yp = 0, 0

            if y1<125:
                if 350<x1<550 :
                    header = overlayList[4]
                    drawcolor = (0,0,255)

                elif 650<x1<800:
                    header = overlayList[1]
                    drawcolor = (255,0,0)

                elif 1000<x1<1150:
                    header = overlayList[2]
                    drawcolor = (0,255,0)
                elif 1300<x1<1450:
                    header = overlayList[3]
                    drawcolor = (0,0,0)

            cv2.rectangle(img, (x1, y1-15), (x2, y2+15),drawcolor, cv2.FILLED)



        #5. If Drawing mode - when index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1), 15, drawcolor, cv2.FILLED)
            #print("Drawing mode")
            if xp ==0 and yp == 0 :
                xp,yp = x1,y1

            if drawcolor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, eraserThikness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, eraserThikness)
            else:
                cv2.line(img, (xp,yp),(x1,y1), drawcolor, brushThikness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1), drawcolor, brushThikness)

            xp,yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGray, 50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    #set the header
    img[ 0:125 , 300:1580 ] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0.0)
    cv2.imshow("image",img)
    cv2.imshow("Canvas",imgCanvas)

    cv2.waitKey(1)





