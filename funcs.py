from asyncio import start_server
from email.mime import image
from re import I
import tensorflow as tf
import numpy as np
import cv2
import tensorflow_hub as hub
from pydub import AudioSegment
from pydub.playback import play


model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']


def playsound(length):
    song = AudioSegment.from_mp3("rg/rg.mp3")
    cr=song[:length*1000]
    print('started')
    play(cr)
    print('playsound')




def doll(facing):

    #return 1080x384 bgr image array of doll facing front if input facing is a string 'front' and back if 
    # facing is 'back'
    if facing=='front':
        img=cv2.resize(cv2.imread('rg/dollfront.jpg'),(384,1080))
    else:
        img=cv2.resize(cv2.imread('rg/dollback.jpg'),(384,1080))
    
    return img




def pose(image):

    #return array of size (n,17,2) where n is number of humans and second and third axis is [[y_0, x_0], [y_1, x_1], …, [y_16, x_16]]
    #where y_i, x_i are the yx-coordinates of the i-th body part of nth human correspondingly
    im=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    frame=image.copy()
    im = tf.expand_dims(im, axis=0)
    im = tf.cast(tf.image.resize(im, (288, 512)), dtype=tf.int32)
    outputs = movenet(im)
    keypoints = outputs['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    coords=keypoints[:,:,:2]
    conf=keypoints[:,:,2]
    l=[]
    for i in range(len(conf)):
        if np.sum(conf[i])<1.7:
            l.append(i)
    coords=np.delete(coords,l,axis=0)
    coords=np.uint16(np.squeeze(np.multiply(coords, [image.shape[0],image.shape[1]])))

    return coords


def sortx(cords):
    
    #input is 'cords' returned in function pose() return (n,17,2) array where 0th axis is sorted based on 
    # average x axis value of body part coordinates
    srtdcrds=np.zeros(cords.shape)
    xaxis=cords[:,:,1]
    xaxis=np.sum(xaxis,axis=1)
    xl=list(xaxis)
    xl1=xl.copy()
    xl.sort()
    for i in range(len(xl)):
        j=xl1.index(xl[i])
        srtdcrds[i,:,:]=cords[j,:,:]
    return srtdcrds


def error(pfpose,fpose):

    #input is 'pose' returned in pose() return 1d array of size n where nth element is average 
    # distance between body parts of nth human
    n=fpose.shape[0]
    er=np.zeros(n)
    for i in range(n):
        c=fpose[i]
        p=pfpose[i]
        dist=np.power(c-p,2)
        dist=np.sum(dist,axis=1)
        dist=np.power(dist,0.5)
        dist=np.sum(dist)
        er[i]=dist
    return er


def playerbar(playerstatus):

    #return 216x1536 image showing status of players visually

    bar=np.zeros((216,1536,3))
    bar[:,:,2]+=255
    return bar


def hlightstatus(playerstatus,frame,fpose):


    #highlight players red if player status is dead and green if alive
    #using coordinates from fpose on frame. resize final image to 864x1536
    image=np.copy(frame)
    for i in range(fpose.shape[0]):
        for kp in fpose[i]:
            if playerstatus[i]=='alive':
                cv2.circle(image, (int(kp[1]), int(kp[0])), 6, (0,255,0), -1)
            else:
                cv2.circle(image, (int(kp[1]), int(kp[0])), 6, (0,0,255), -1)

    return cv2.resize(image,(1536,864))
 


