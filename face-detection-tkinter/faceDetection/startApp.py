from App.app import FaceDetect
from imutils.video import VideoStream
import time

vs = VideoStream(0).start()
time.sleep(1)

sa = FaceDetect(vs)
sa.root.mainloop()