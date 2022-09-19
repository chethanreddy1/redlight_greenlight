import cv2 as cv
import numpy as np
import cv2
import random
import time

from funcs import *

img=np.zeros((1080,1920,3))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("cameraproblem")
    exit()
ret, frame = cap.read()

fpose=pose(frame)
nplayers=fpose.shape[0]
playerstatus=['alive']*nplayers

img[216:,384:,:]=hlightstatus(playerstatus,frame,fpose)
img[:216,384:,:]=playerbar(playerstatus)

while 'alive' in playerstatus:
    img[:,:384,:]=doll('back')
    cv2.imshow('main',img)

    length=random.randint(1,4)
    playsound(length)

    img[:,:384,:]=doll('front')
    cv2.imshow('main',img)

    # time.sleep(0.5)
    ff=True
    t=time.time()

    while t-time.time>5:
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
            threshold=0
            for i in len(delta):
                if delta[i]>threshold:
                    playerstatus[i]=='dead'
            img[:216,384:,:]=playerbar(playerstatus)
            img[216:,384:,:]=hlightstatus(playerstatus,frame,fpose)
            pframe=frame
            pfpose=fpose
            cv2.imshow('main',img)

cap.release()
cv.destroyAllWindows()    


