import cv2 as cv
import numpy as np
import cv2


pframe=None
ff=True
cap = cv.VideoCapture(0)

def hlight(pframe,frame):

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # blurring to remove noise from camera shaking
    frame = cv.GaussianBlur(frame, (21, 21), 0)
    #abs difference from previous frame
    frameDelta = cv.absdiff(pframe, frame)
    thresh = cv.threshold(frameDelta, 10, 255, cv.THRESH_BINARY)[1]

    #morphogical dilation to remove small holes
    thresh = cv.dilate(thresh, None, iterations=2)
    return frameDelta, thresh, frame

def hlightred(pframe,frame):

    frameDelta, thresh, frame2=hlight(pframe,frame)
    red=thresh==[255]
    red=np.uint8(np.stack((red,red,red),axis=-1))
    red=red*[0,0,255]

    framer=frame+red
    framer=np.clip(framer,0,255)

    print(red.shape)
    return np.uint8(framer)


if not cap.isOpened():
    print("cameraproblem")
    exit()

while True:

    ret, frame = cap.read()

    #if not first frame of stream save previous frame
    if ff:
        pframe=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ff=False
        pass

    red=hlightred(pframe,frame)
    frameDelta,thresh,frame=hlight(pframe,frame)
    

    cv.imshow('framedelta', frameDelta)
    cv.imshow('thresh',thresh)
    cv.imshow('frame',frame)
    cv.imshow('red',red)


    pframe=frame
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()    