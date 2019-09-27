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

def getFaceBox(net, frames, conf_threshold=0.7, from_one=False ):
    bboxes = []
    if from_one:
        frames = frames[:1]
    for frame in frames:
        try:
            blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        except:
            bboxes.append([])
            continue

        net.setInput(blob)
        detections = net.forward()
        done = False
        for i in range(len(frames)):
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            if detections.shape[1]:
                confidence = detections[0, 0, 0, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, 0, 3] * frameWidth)
                    y1 = int(detections[0, 0, 0, 4] * frameHeight)
                    x2 = int(detections[0, 0, 0, 5] * frameWidth)
                    y2 = int(detections[0, 0, 0, 6] * frameHeight)
                    bboxes.append([x1, y1, x2, y2])
                    done = True
                    if not from_one:
                        break
        if not done:
            bboxes.append([])

    return frames, bboxes

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

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = np.array(['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'])
genderList = np.array(['Male', 'Female'])
padding = 20
def process_face(frames, face_threshold=0.5, gender_threshold=0.7, from_one=False):

    frameFace, bboxes = getFaceBox(faceNet, frames, face_threshold, from_one=from_one)
    if not len(bboxes):
        return frameFace, None, None, None
    faces = []
    for i in range(len(frameFace)):
        if len(bboxes[i]):
            faces.append(crop_box(frameFace[i],bboxes[i],padding))
        else: #TODO: there should be a more efficient aproach to obtaining None Values
            faces.append(255*np.ones(shape=[227, 227, 3], dtype=np.float32))
    try:
        blob = cv.dnn.blobFromImages(faces, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    except:
        return frameFace, None, None, None
    tensor = torch.from_numpy(blob)
    tensor = tensor.cuda()

    genderPreds = genderNet.forward(tensor)
    gender_confs,max_ids = genderPreds.max(1)
    genders = genderList[max_ids]
    mask = gender_confs < gender_threshold

    agePreds = ageNet.forward(tensor)
    age_confs, age_ids = agePreds.max(1)
    ages = ageList[age_ids]
    dim = (gender_confs.size()[0],1)
    genders_out = np.hstack((genders.reshape(*dim),gender_confs.detach().cpu().numpy().reshape(*dim)))
    ages_out = np.hstack((ages.reshape(*dim),age_confs.detach().cpu().numpy().reshape(*dim)))

    mask = mask.detach().cpu().numpy()
    genders_out[mask==1] = np.array([None,None])
    ages_out[mask==1] = np.array([None,None])
    
    return frameFace, ages_out, genders_out, bboxes
