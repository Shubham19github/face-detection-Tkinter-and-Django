from django.shortcuts import render
from django.conf import settings
from django.http import StreamingHttpResponse
import time
import threading
import cv2
import numpy as np
import os

# files used to detect faces
protoTypeFile = os.path.join(settings.BASE_DIR, 'deploy.prototxt.txt')
caffeModelFile = os.path.join(settings.BASE_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

cap = None

# Create your views here.

class LiveFeed(object):
	global protoTypeFile, caffeModelFile, cap

	def __init__(self):

		self.net = cv2.dnn.readNetFromCaffe(protoTypeFile, caffeModelFile)

		self.ret, self.frame = cap.read()
		threading.Thread(target=self.updateFrame, args=()).start()

	def updateFrame(self):
		while True:
			self.ret, self.frame = cap.read()
			self.frame = self.detectFace()
			# cv2.waitKey(25)	# delaying on purpose to see the face detected


	def readCamera(self):
		ret, jpeg = cv2.imencode('.jpg', self.frame)
		return jpeg.tobytes()

	def detectFace(self):

		# grab the frame dimensions and convert it to a blob
		(h, w) = self.frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))
		
		# pass the blob through the network and obtain the detections and
		# predictions
		self.net.setInput(blob)
		detections = self.net.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence < 0.5:
				continue

			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
		
			# draw the bounding box of the face
			
			cv2.rectangle(self.frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

		return self.frame


def get_live_feed(lf):
	# continuously get the frames from webcam
	while True:
		frame = lf.readCamera()
		yield(b'--frame\r\n'
			b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# function executes when start camera button is called
def live_feed_view(request):
	global cap

	cap = cv2.VideoCapture(0)
	
	lf = LiveFeed()

	responseData = get_live_feed(lf)
	live_feed =  StreamingHttpResponse(responseData , content_type="multipart/x-mixed-replace;boundary=frame")
	
	return live_feed

# releases or closes the webcam
def toDelete_lf():
	global cap
	cap.release()