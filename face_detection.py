import sys
import cv2
import re
import os
import dlib
import numpy as np
from skimage import io
from scipy import misc

face_path = './lfw_wiki.txt'

def GetFileNameAndExt(filename):
    (filepath,tempfilename) = os.path.split(filename);
    (shotname,extension) = os.path.splitext(tempfilename);
    return shotname

def videoCap(video):
    vc = cv2.vc = cv2.VideoCapture(video)
    timef = 1
    num_frame = 1
    filename = GetFileNameAndExt(video)
    print filename
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval and num_frame <= 3:
        rval, frame = vc.read()  
        if(timef%25 == 0):
            cv2.imwrite('./'+ filename +  '_' +str(num_frame)+'.jpg',frame)
            num_frame += 1
        timef += 1  
        cv2.waitKey(1)  
    vc.release()

def face_predictor(pic, pic_file): 
    predictor_path = sys.argv[1]
    faces_path = pic
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img = io.imread(faces_path)

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    if len(dets) == 0:
        pic_file.write(pic + '\n')
        return pic

    filename = GetFileNameAndExt(pic)
    #f=open(filename + '.txt','w')
    
    flag = 0;

    for k, d in enumerate(dets):   
        print("dets{}".format(d))
        #f.write('Face area: ' + str(d) + '\n')

        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))

        left, right, top, bottom = d.left()*0.9, d.right()*1.1, d.top()*0.6, d.bottom()*1.15
        print(d.left(), d.right(),d.top(), d.bottom())

        if(left < 0): 
            left = 0
        if(right > img.shape[1]): 
            right = img.shape[1]
        if(top < 0): 
            top = 0
        if(bottom > img.shape[0]): 
            bottom = img.shape[0]

        #cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()),(0,0,255),2,0)

        #f.write('Face spacial point: \n')
        #for i in range(len(dets)):
        #    facepoint = np.array([[p.x, p.y] for p in predictor(img, dets[i]).parts()])
        #    for j in range(68):
        #        cv2.circle(img,(facepoint[j][0],facepoint[j][1]) , 3, (255,0,0), -1)
        #        f.write(str((facepoint[j][0],facepoint[j][1]))+'\n') 

        if(flag == 0):
            img1 = img[top:bottom, left:right]
            print(img1.shape)

            crop_size = 224
            img1 = misc.imresize(img1, (crop_size, crop_size))
            print(img1.shape)

            io.imsave( './'+ faces_path , img1)
            flag = flag + 1

        
 

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'predictor.dat':
            dropout_pic = open('dropout_pic.txt', 'w')
            with open(face_path) as f:
                lines = f.readlines()
                for l in lines:
                    items = l.split()
                    print (items[0])
                    face_predictor(items[0], dropout_pic)
    else:
        for video in os.listdir(face_path):
            if os.path.splitext(video)[1] == ".ts":
                videoCap(video)