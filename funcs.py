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

    #return array of size (n,17,2) where n is number of humans and second and third axis is [[y_0, x_0], [y_1, x_1], …, [y_16, x_16]]
    #where y_i, x_i are the yx-coordinates of the i-th body part of nth human correspondingly

    im=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
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
    return np.uint8(np.squeeze(np.multiply(coords, [image.shape[0],image.shape[1]])))


def sortx(pose):

    #input is 'pose' returned in function pose() return (n,17,2) array where 0th axis is sorted based on 
    # average x axis value of corresponding body part coordinates
    return pose


def error(pfpose,fpose):

    #input is 'pose' returned in pose() return 1d array of size n where nth element is average 
    # distance between body parts of nth human
    er=0
    return er


def playerbar(playerstatus):
    #return 216x1536 image showing status of players visually

    bar=np.zeros((216,1536,3))
    return bar


def hlightstatus(playerstatus,frame,fpose):
    image=frame.copy()
    #highlight players red if player status is dead and green if alive
    #using coordinates from fpose on frame. resize final image to 864x1536

    return cv2.resize(image,(1536,864))





