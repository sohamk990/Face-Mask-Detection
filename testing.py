import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import gc

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()        
        
        self.ac = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
        self.conv1 = nn.Conv2d(1,100,3)
        self.conv2 = nn.Conv2d(100,200,3)
        self.drop = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(200*23*23, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.ac(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.ac(x)
        x = self.pool(x)
        
        x = x.view(-1,200*23*23)
        
        x = self.drop(x)
        
        x = self.fc1(x)
        
        x = self.fc2(x)
        return x

model = torch.load("../input/modellll/model.pth")
model.eval()

face_clsfr=cv.CascadeClassifier('../input/haaaaaaas/haas.xml')

source=cv.VideoCapture('../input/ameyaaaa/Ameya_With_Mask.MP4')

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

human=[]

while True:
    
    ret,image=source.read()
    if ret==False:
        break
    
    classes = None
    with open('../input/maskdetection/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]


    Width = image.shape[1]
    Height = image.shape[0]

    net = cv.dnn.readNet('../input/maskdetection/yolov3.weights', '../input/maskdetection/yolov3.cfg')

    net.setInput(cv.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)


    class_ids = []
    confidences = []
    boxes = []

    #create bounding box 
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

    for i in indices:
        i = i[0]
        box = boxes[i]
        if class_ids[i]==0:
            label = str(classes[class_id]) 
            cv.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (255, 0, 0), 8)
            #cv.putText(image, label, (round(box[0])-10,round(box[1])-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            img=image[round(box[1]):round(box[1]+box[3]),round(box[0]):round(box[0]+box[2])]
            image=cv.cvtColor(image, cv.COLOR_RGB2BGR)
            cv.imshow(image)
            human.append(img)
    
face=[]

cascPathface = os.path.dirname(
    cv.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv.CascadeClassifier(cascPathface)
eyeCascade = cv.CascadeClassifier(cascPatheyes)

for i in range(len(human)):
    frame = human[i]
    if(frame.shape[0]!=0 and frame.shape[1]!=0):
        plt.imshow(frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv.CASCADE_SCALE_IMAGE)
        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
            faceROI = frame[y:y+h,x:x+w]
            new_frame=frame[round(y):round(y+h),round(x):round(x+w)]
            face.append(new_frame)

size=(human[0].shape[1],human[0].shape[0])
out=cv.VideoWriter('final.mp4',cv.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(face)):

    img=face[i]
    if(img.shape[0]!=0 and img.shape[1]!=0):
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        faces=face_clsfr.detectMultiScale(gray,1.3,5)  
        img=cv.resize(img,(human[0].shape[1],human[0].shape[0]))
        
        for x,y,w,h in faces:
            face_img=gray[y:y+w,x:x+w]
            resized=cv.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))

            ten_reshaped = torch.from_numpy(reshaped.astype(np.float32))
            ten_reshaped = ten_reshaped.view(-1,1,100,100)
            result=model(ten_reshaped)

            result = result.detach().numpy()
            result = softmax(result)
            label=np.argmax(result,axis=1)[0]
            print(result)
            if(result[0][0]>=0.10):
                label=0
                print('MASK')
            else:
                label=1
                print('NO_MASK')
            cv.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
            cv.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
            cv.putText(img, labels_dict[label], (x, y-10),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            out.write(img)
            print(img.shape)


source.release()
