import cv2 as cv
import numpy as np
import cv2
import random
import time

from funcs import *

time.sleep(3)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("cameraproblem")
    exit()

ret, frame = cap.read()

fpose=pose(frame)
nplayers=fpose.shape[0]
playerstatus=['alive']*nplayers



while 'alive' in playerstatus:

    cv2.imshow('main1',doll('back'))


    length=random.randint(1,4)
    playsound(length)
    cv2.imshow('main1',doll('front'))


    # time.sleep(0.5)
    ff=True


    while True:
        ret, frame = cap.read()



        if ff:
            pframe=frame
            pfpose=pose(pframe)
            pfpose=sortx(pfpose)
            ff=False
        else:

            fpose=pose(frame)
            fpose=sortx(fpose)
            delta=error(pfpose,fpose)
            threshold=10000000000000000
            for i in range(len(delta)):
                if delta[i]>threshold:
                    playerstatus[i]=='dead'
                    print(i,' died')


            cv2.imshow('main2',playerbar(playerstatus))


            cv2.imshow('main3',hlightstatus(playerstatus,frame,fpose))

            
           
            pframe=frame
            pfpose=fpose
            if cv.waitKey(1) == ord('q'):
                 break



cap.release()
cv.destroyAllWindows()    


