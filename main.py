# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob

from age_gender import process_face
from sort import Sort
from utils import ccw, intersect, translate_bbox,nms,get_colors, crop_box, generate_line 

def set_args():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", help="path to input video")
	ap.add_argument("-o", "--output", required=True,
		help="path to output video")
	ap.add_argument("-y", "--yolo", required=True,
		help="base path to YOLO directory")
	ap.add_argument("-cc", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	ap.add_argument("-nmst", "--threshold", type=float, default=0.3,
		help="threshold when applying non-maxima suppression")
	ap.add_argument("-ft", "--face-threshold", type=float, default=0.4,
		help="threshold when applying non-maxima suppression")
	ap.add_argument("-gt", "--gender-threshold", type=float, default=0.7,
		help="threshold when applying non-maxima suppression")
	ap.add_argument("-ct", "--count", action='store_true',
		help="Count people")
	ap.add_argument("-ch", "--characteristics", action='store_true',
		help="Predict gender and age")
	return ap

def set_yolo(args):
	labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	labels = open(labelsPath).read().strip().split("\n")

	weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	# and determine only the *output* layer names that we need from YOLO
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	return labels, net,ln

if __name__ == '__main__':
	ap = set_args()
	args = vars(ap.parse_args())
	if not (args['characteristics'] or args['count']):
		raise ValueError("Usage: Need at least one of -ct or -ch")
	padding = 10 # padding for bbox cropping
	writer = None
	# try to determine the total number of frames in the video file
	vs = cv2.VideoCapture(args["input"] if args["input"] is not None else 0)
	if args["input"] is not None:
		try:
			prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
				else cv2.CAP_PROP_FRAME_COUNT
			total = int(vs.get(prop))
			print("[INFO] {} total frames in video".format(total))
		except:
			print("[INFO] could not determine # of frames in video")
			print("[INFO] no approx. completion time can be provided")
			total = -1

	if args['count']:
		tracker = Sort()
		memory = {}
		line = [(20, 43), (60, 900)] # go through line
		counter = 0 # counted objects
		frameIndex = 0
		LABELS, net,ln = set_yolo(args)
		COLORS = get_colors()
		face_chars = {} #hash to store obtained caracteristics	
		(W, H) = (None, None)
		# loop over frames from the video file stream
		while True:
			# read the next frame from the file
			(grabbed, frame) = vs.read()
			if not grabbed:
				break
			if W is None or H is None:
				(H, W) = frame.shape[:2]

			# construct a blob and perform a forward  pass of the YOLO object detector
			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			start = time.time()
			layerOutputs = net.forward(ln)
			

			# lists of detections for frame
			boxes = []
			confidences = []
			classIDs = []

			for output in layerOutputs:
				for detection in output:
					# extract the class ID and confidence (i.e., probability)
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]
					if confidence > args["confidence"] and classID == 0: # 0 is for person
						# change bbox output format
						x,y, width,height = translate_bbox(detection,W,H)
						# update our lists 
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)

			# apply non-maxima suppression to suppress overlapped bboxes
			dets = nms(boxes,confidences, args['confidence'], args['threshold'])

			np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
			dets = np.asarray(dets)
			tracks = tracker.update(dets)
			boxes = []
			indexIDs = []
			c = []
			previous = memory.copy()
			memory = {}
			for track in tracks:
				boxes.append([track[0], track[1], track[2], track[3]])
				indexIDs.append(int(track[4]))
				memory[indexIDs[-1]] = boxes[-1]

			if len(boxes) > 0:
				i = 0
				for box in boxes:
					(x, y) = (int(box[0]), int(box[1]))
					(w, h) = (int(box[2]), int(box[3]))
					person = crop_box(frame,box,padding)
					if args["characteristics"] and indexIDs[i] not in face_chars:
						face, age, gender = process_face(person, face_threshold=args["face_threshold"], gender_threshold=args["gender_threshold"])
						if age is not None:
							face_chars[indexIDs[i]] = {
								"age": age,
								"gender": gender
							}
					color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
					cv2.rectangle(frame, (x, y), (w, h), color, 2)
					#check intersection with target line
					if indexIDs[i] in previous:
						previous_box = previous[indexIDs[i]]
						p0,p1 = generate_line(previous_box,box)
						cv2.line(frame, p0, p1, color, 3)
						if intersect(p0, p1, line[0], line[1]):
							counter += 1
					if indexIDs[i] not in face_chars:
						text = "{}".format(indexIDs[i])
						cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
					else:
						text = "{},{}".format(face_chars[indexIDs[i]]["gender"], face_chars[indexIDs[i]]["age"])
						cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
					i += 1
			cv2.line(frame, line[0], line[1], (0, 255, 255), 5)
			cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
			end = time.time()
			if total > 0:
				if frameIndex == 0:
					elap = (end - start)
					print("[INFO] single frame took {:.4f} seconds".format(elap))
					print("[INFO] estimated total time to finish: {:.4f}".format(
						elap * total))
				print(f"[INFO] {frameIndex}/{total}\r", end="")
			if writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(args["output"], fourcc, 30,
					(frame.shape[1], frame.shape[0]), True)
			writer.write(frame)
			frameIndex += 1
		print("[INFO] cleaning up...")
		writer.release()
	elif args['characteristics']:
		frameIndex = 0
		while cv2.waitKey(1) < 0:
			(grabbed, frame) = vs.read()
			if not grabbed:
				break
			start = time.time()
			frameFace, age,gender = process_face(frame, face_threshold=args["face_threshold"], gender_threshold=args["gender_threshold"])
			#cv2.imshow("Read characteristics", frameFace)
			if writer is None:
				# initialize our video writer
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(args['output'], fourcc, 30,
					(frameFace.shape[1], frameFace.shape[0]), True)
			writer.write(frameFace)
			end = time.time()
			if total > 0:
				if frameIndex == 0:
					elap = (end - start)
					print("[INFO] single frame took {:.4f} seconds".format(elap))
					print("[INFO] estimated total time to finish: {:.4f}".format(
						elap * total))
				print(f"[INFO] {frameIndex}/{total}\r", end="")
			frameIndex += 1
	vs.release()