import numpy as np
import cv2

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


# initialize a list of colors 
def get_colors():
	np.random.seed(42)
	colors = np.random.randint(0, 255, size=(200, 3),
		dtype="uint8")
	return colors


# scale the bounding box coordinates back relative to
# the size of the image, keeping in mind that YOLO
# actually returns the center (x, y)-coordinates of
# the bounding box followed by the boxes' width and
# height
def translate_bbox(detection,W,H):
    box = detection[0:4] * np.array([W, H, W, H])
    (centerX, centerY, width, height) = box.astype("int")
    x = int(centerX - (width / 2))
    y = int(centerY - (height / 2))
    return x,y, width,height

# Non-Maxima supression
def nms(boxes,confidences,detection_treshold,supression_threshold):
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, detection_treshold, supression_threshold)
    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])
    return dets

