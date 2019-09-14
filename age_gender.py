# Import required modules
import cv2 as cv
import math
import time
import argparse
import torch
import imp
import numpy as np
from torchvision.transforms import ToTensor

from utils import crop_box

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    try:
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    except:
        print("Could not blob person")
        return frameOpencvDnn, []

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    if detections.shape[2]:
        confidence = detections[0, 0, 0, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, 0, 3] * frameWidth)
            y1 = int(detections[0, 0, 0, 4] * frameHeight)
            x2 = int(detections[0, 0, 0, 5] * frameWidth)
            y2 = int(detections[0, 0, 0, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

faceProto = "age_gender_models/opencv_face_detector.pbtxt"
faceModel = "age_gender_models/opencv_face_detector_uint8.pb"

age_weights_path = "age_net/age_net.pth"
age_model_path = "age_net/age_net.py"


gender_weights_path = "gender_net/gender_net.pth"
gender_model_path = "gender_net/gender_net.py"

# Load networks

MainModel = imp.load_source('MainModel', gender_model_path)
genderNet = torch.load(gender_weights_path)
genderNet.eval()
genderNet.cuda()

MainModel = imp.load_source('MainModel', age_model_path)
ageNet = torch.load(age_weights_path)
ageNet.eval()
ageNet.cuda()

faceNet = cv.dnn.readNet(faceModel, faceProto)

# Open a video file or an image file or a camera stream

def process_face(frame, face_threshold=0.5, gender_threshold=0.7):
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    padding = 20
    frameFace, bboxes = getFaceBox(faceNet, frame, face_threshold)
    age = None,
    gender = None
    age_conf = None
    gender_conf = None
    if not bboxes:
        return frameFace, (None,None), (None,None)

    for bbox in bboxes:
        # print(bbox)
        face = crop_box(frame,bbox,padding)
        try:
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        except:
            print("Could not blob face")
            continue
        tensor = torch.from_numpy(blob)
        tensor = tensor.cuda()
        genderPreds = genderNet.forward(tensor)
        max_id = genderPreds[0].argmax()
        gender = genderList[max_id]
        gender_conf = genderPreds[0][max_id]
        if gender_conf < gender_threshold:
            gender = None
            gender_conf = None
            continue
        # print("Gender Output : {}".format(genderPreds))
        # print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
        agePreds = ageNet.forward(tensor)
        age = ageList[agePreds[0].argmax()]
        age_conf =  agePreds[0][agePreds[0].argmax()]
        # print("Age Output : {}".format(agePreds))
        # print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)        
    return frameFace, (age, age_conf), (gender,gender_conf)
