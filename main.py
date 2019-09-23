# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
import signal
import atexit
import sys

from sort import Sort
from utils import ccw, intersect_object, translate_bbox,nms,get_colors, crop_box 
from data import Storage
from datetime import datetime
from yolo.darknet import Darknet
from yolo.util import *
from yolo.preprocess import prep_image, inp_to_image

def set_args():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", help="path to input video")
	ap.add_argument("-o", "--output", required=True,
		help="path to output video (must have .mp4 ext)")
	ap.add_argument("-do", "--data-output", required=True,
		help="path to output data")
	ap.add_argument("-y", "--yolo", required=True,
		help="base path to YOLO directory")
	ap.add_argument("-cc", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	ap.add_argument("-nmst", "--threshold", type=float, default=0.3,
		help="threshold when applying non-maxima suppression")
	ap.add_argument("-ft", "--face-threshold", type=float, default=0.4,
		help="threshold when applying face detection")
	ap.add_argument("-gt", "--gender-threshold", type=float, default=0.7,
		help="threshold when applying gender classification")
	ap.add_argument("-ct", "--count", action='store_true',
		help="Count people")
	ap.add_argument("-ch", "--characteristics", action='store_true',
		help="Predict gender and age")
	ap.add_argument("-sh", "--show", action='store_true',
		help="Show frames while processing")
	return ap

def set_yolo(args):
	labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	labels = load_classes(labelsPath)

	weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	# and determine only the *output* layer names that we need from YOLO
	print("[INFO] loading YOLO from disk...")
	model = Darknet(configPath)
	model.load_weights(weightsPath)
	model.net_info["height"] = 320
	model.cuda()
	model.eval()
	return labels, model

def handle_exit(*func_args):
	print("[INFO] saving up...")
	vs.release()
	writer.release()
	if args['count']:
		for index,date in counts:
			if index in face_chars:
				storage.save(
					date,
					face_chars[index]['gender'],
					face_chars[index]['age']
					)
			else:
				storage.save(
					date,
					None,
					None
					)
		storage.store(args['data_output'])
	end = time.time()
	elap = end - start
	print("[INFO] Total time took {:.4f} seconds".format(elap))
	print("[INFO] FPS: {}".format(int(frameIndex/elap)))
	sys.exit(0)

if __name__ == '__main__':
	ap = set_args()
	args = vars(ap.parse_args())
	if not (args['characteristics'] or args['count']):
		raise ValueError("Usage: Need at least one of -ct or -ch")
	if args['characteristics']:
		from age_gender import process_face

	padding = 10 # padding for bbox cropping
	writer = None
	fps = None
	vs = cv2.VideoCapture(args["input"] if args["input"] is not None else 0)

	storage = Storage()

	atexit.register(handle_exit)
	signal.signal(signal.SIGTERM, handle_exit)
	signal.signal(signal.SIGINT, handle_exit)
	if args["input"] is not None:
		try:
			prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
				else cv2.CAP_PROP_FRAME_COUNT
			total = int(vs.get(prop))
			fps = int(vs.get(cv2.CAP_PROP_FPS))
			print("[INFO] {} total frames in video".format(total))
			print("[INFO] video fps {} ".format(fps))
		except:
			print("[INFO] could not determine # of frames in video")
			print("[INFO] no approx. completion time can be provided")
			total = -1
	else:
		total = 0
	if args['count']:
		tracker = Sort()
		memory = {}
		line = None # go through line
		counter = 0 # counted objects
		frameIndex = 0
		LABELS, net = set_yolo(args)
		COLORS = get_colors()
		face_chars = {} #hash to store obtained caracteristics
		counts = []	# array of counted intersections.
		counted_ids = set() #set of already counted ids
		(W, H) = (None, None)
		# loop over frames from the video file stream
		
		while cv2.waitKey(1) < 0:
			# read the next frame from the file
			(grabbed, frame) = vs.read()
			if not grabbed:
				break
			if W is None or H is None:
				(H, W) = frame.shape[:2]
			if line is None: #catch click input for line definition
				cv2.namedWindow("input")
				line = []
				def click_and_crop(event, x, y, flags, param):
					# if the left mouse button was clicked, record x,y
					if event == cv2.EVENT_LBUTTONDOWN:
						line.append((x,y))
				cv2.setMouseCallback("input", click_and_crop)
				while len(line) < 2:
					# display the image and wait for a keypress
					cv2.imshow("input", frame)
					key = cv2.waitKey(1) & 0xFF
				cv2.destroyAllWindows()
				start = time.time()
				
			# construct a blob and perform a forward  pass of the YOLO object detector
			
			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320),
				swapRB=True, crop=False)
			tensor = torch.from_numpy(blob)
			tensor = tensor.cuda()
			layerOutputs = net(tensor,True)
			dets = write_results(layerOutputs, args['confidence'], len(LABELS), nms = True, nms_conf = args['threshold']) #[im_id, x0,y0,x1,y1,?,conf?,class_id]
			if type(dets) != int:
				#bbox rescaling
				im_dim_list=torch.Tensor([[W,H]*2]*dets.size()[0]).cuda()
				dets[:,1:5] /= 320
				dets[:,1:5] *= im_dim_list

				# lists of detections for frame
				confidences = []
				classIDs = []
				to_track = []
				for i in range(dets.shape[0]):
					if int(dets[i,-1]) == 0:
						classIDs.append(int(dets[i,-1]))
						confidences.append(dets[i,-2])
						to_track.append(np.asarray(dets[i,[1,2,3,4,6]]))

				np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
				dets = np.asarray(to_track)
				tracks = tracker.update(dets)
				boxes = []
				indexIDs = []
				previous = memory.copy()
				memory = {}
				for track in tracks:
					boxes.append([track[0], track[1], track[2], track[3]])
					indexIDs.append(int(track[4]))
					memory[indexIDs[-1]] = boxes[-1]

				if len(boxes) > 0:
					i = 0
					people = []
					frame_ids = []
					for box in boxes:
						(x, y) = (int(box[0]), int(box[1]))
						(w, h) = (int(box[2]), int(box[3]))
						
						if args["characteristics"]: #TODO: this is slowing down yolo
							people.append(crop_box(frame,box,padding).astype(np.float32))
							frame_ids.append(indexIDs[i])
						
						color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
						cv2.rectangle(frame, (x, y), (w, h), color, 2)
						#check intersection with target line
						if indexIDs[i] in previous and indexIDs[i] not in counted_ids:
							previous_box = previous[indexIDs[i]]
							if intersect_object(previous_box,box,line): #TODO: this seems to be quite slow and not working properly
								counter += 1
								counted_ids.add(indexIDs[i])
								counts.append((indexIDs[i],datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
								
						if indexIDs[i] not in face_chars:
							text = "{}".format(indexIDs[i])
							cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
						else:
							text = "{},{}".format(face_chars[indexIDs[i]]["gender"][0], face_chars[indexIDs[i]]["age"][0])
							cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
						i += 1
					if args["characteristics"]:
						faces, ages, genders = process_face(people, face_threshold=args["face_threshold"], gender_threshold=args["gender_threshold"])
						if genders is not None and ages is not None:
							for i in range(len(genders)):
								if  frame_ids[i] not in face_chars and genders[i][0] != 'None':
										face_chars[frame_ids[i]] = {
											"age": ages[i],
											"gender": genders[i]
										}
								elif genders[i][0] != 'None':
									# check for a better quality prediction
										if genders[i][1] > face_chars[frame_ids[i]]['gender'][1]:
											face_chars[frame_ids[i]]['gender'] = genders[i]
										if ages[i][1] > face_chars[frame_ids[i]]['age'][1]:
											face_chars[frame_ids[i]]['age'] = ages[i]

			cv2.line(frame, line[0], line[1], (0, 255, 255), 5)
			cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
			if total > 0:
				if frameIndex == 0:
					end = time.time()
					elap = (end - start)
					print("[INFO] single frame took {:.4f} seconds".format(elap))
					print("[INFO] estimated total time to finish: {:.4f}".format(
						elap * total))
				print(f"[INFO] {frameIndex}/{total}\r", end="")
			if writer is None:
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				output_fps = fps if fps is not None else 20
				writer = cv2.VideoWriter(args["output"], fourcc, output_fps,
					(frame.shape[1], frame.shape[0]), True)
			if args["show"]: 
				cv2.imshow("Count people", frame)
			writer.write(frame)
			frameIndex += 1
	# FIXME: NOT RUNNING
	elif args['characteristics']:
		frameIndex = 0
		start = time.time()
		while cv2.waitKey(1) < 0:
			(grabbed, frame) = vs.read()
			if not grabbed:
				break
			frameFace, age,gender = process_face(frame, face_threshold=args["face_threshold"], gender_threshold=args["gender_threshold"])
			if args["show"]:  cv2.imshow("Read characteristics", frameFace)
			if writer is None:
				# initialize our video writer
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				output_fps = fps if fps is not None else 20
				writer = cv2.VideoWriter(args['output'], fourcc, output_fps,
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