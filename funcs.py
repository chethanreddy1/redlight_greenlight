import tensorflow as tf
import numpy as np
import cv2
import tensorflow_hub as hub

model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']


def playsound(length):

    #play a sound for 'length' seconds
    return None

def doll(facing):

    #return 1080x384 bgr image array of doll facing front if input facing is a string 'front' and back if 
    # facing is 'back'

    img=np.zeros((1080,384,3))
    return img

def pose(image):

    #return array of size (n,34) where n is number of humans and second axis is [y_0, x_0, y_1, x_1, …, y_16, x_16]
    #where y_i, x_i are the yx-coordinates of the i-th body part correspondingly

    ps=np.zeros((4,34))
    return ps

def sortx(pose):

    #input is 'pose' returned in function pose() return (n,34) array where 0th axis is sorted based on 
    # average x axis value of corresponding body part coordinates
    return pose


def error(pfpose,fpose):

    #input is 'pose' returned in pose() return 1d array of size n where nth element is average 
    # distance between body parts of nth human
    er=0
    return er

def playerbar(playerstatus):
    #return 216x1536 image showing status of players visually

    bar=np.zeros((216,1536))
    return bar

def hlightstatus(playerstatus,frame,fpose):
    #highlight players red if player status is dead and green if alive
    #using coordinates from fpose on frame

    return frame


