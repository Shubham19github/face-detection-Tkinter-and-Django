import threading
import imutils
import cv2
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

class FaceDetect(object):
	
	def __init__(self, vs):

		self.thread = None
		self.stopEvent = None
		self.vs = vs
		self.frame = None
		self.panel = None

		# loading files to detect faces
		self.net = cv2.dnn.readNetFromCaffe('data/deploy.prototxt.txt', 'data/res10_300x300_ssd_iter_140000.caffemodel')

		self.root = tk.Tk()
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onWindowClose)	#callback to window close function
		self.root.wm_title("Face Detect")		#Title of window
		self.root.geometry("600x400+200+100")	#Dimension of thee App
		self.root.config(bg="#fff")

		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.detectFaces, args=())
		self.thread.start()


	def detectFaces(self):
		global text
		try:
			while not self.stopEvent.is_set():
				self.frame = self.vs.read()
				self.frame = imutils.resize(self.frame, width=600)

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

	
				image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
				image = Image.fromarray(image)
				image = ImageTk.PhotoImage(image)
		
				# if the panel is not None, we need to initialize it
				if self.panel is None:
					self.panel = tk.Label(image=image)
					self.panel.image = image
					self.panel.pack(side="left", padx=10, pady=10)
		
				# otherwise, simply update the panel
				else:
					self.panel.configure(image=image)
					self.panel.image = image
 
		except RuntimeError:
			print("Exiting App")

	# on app close
	def onWindowClose(self):

		self.stopEvent.set()
		self.vs.stop()
		self.root.destroy()
