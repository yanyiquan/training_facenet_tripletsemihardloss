from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import keras
import os
import numpy as np
import cv2

detector= MTCNN()
def cropping_image(path):
    for file in os.listdir(path):
        dirs = './face/'+ file
        os.makedirs(dirs)
        a=0
        for file_img in os.listdir(os.path.join(path, file)):
            img = cv2.imread(os.path.join(path, file, file_img))
            result = detector.detect_faces(img)
            if result != []:
                for person in result:
                    box = person['box']
                    img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                    if img.size != 0:
                        img = cv2.resize(img, (224, 224))
                        cv2.imwrite(dirs+'/'+str(a)+'.jpg', img)
                        a+=1
def processing_data(path):
    X=[]
    Y=[]
    for file in os.listdir(path):
        dirs = './face/' + file
        for file_img in os.listdir(os.path.join(path, file)):
            img = cv2.imread(os.path.join(path, file, file_img))
            X.append(img)
            Y.append(int(file))
    return np.asarray(X), np.asarray(Y)
