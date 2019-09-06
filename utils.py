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

#crops bbox from frame
def crop_box(frame,box,padding):
    return frame[int(max(0,box[1]-padding)):int(min(box[3]+padding,frame.shape[0]-1)),int(max(0,box[0]-padding)):int(min(box[2]+padding, frame.shape[1]-1))]

def generate_line(previous_box,box):
    (x, y) = (int(box[0]), int(box[1]))
    (w, h) = (int(box[2]), int(box[3]))
    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
    return p0,p1