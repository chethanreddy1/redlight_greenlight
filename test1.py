import numpy as np
import cv2 as cv
import imutils 
pframe=None
ff=True
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("lol")
    exit()
while True:

    ret, frame = cap.read()
    if ff:
        pframe=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ff=False
        pass

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frameDelta = cv.absdiff(pframe, frame)
    thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
    thresh = cv.dilate(thresh, None, iterations=2)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
		# if the contour is too small, ignore it
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
        print(c.shape)
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if not ret:
        print("lol2")
        break

    cv.imshow('frame', frame)
    pframe=frame
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()