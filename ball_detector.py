# import essentials
import math
import numpy as np
import cv2
from cv2 import aruco

# over wifi (phone hotspot)
#url = "http://192.168.43.48:8000/stream.mjpg"
# over ethernet (local switch)
url = "udp://224.0.0.0:1234"

def detect_balls(fast, img):
    """Detect balls in image, return coordinates as list."""

    kps = fast.detect(img,None)
    kps = merge_keypoints(kps, 20)

    return [kp.pt for kp in kps]

def distance_between_points(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def has_close_points(point, others, radius):
    for other in others:
        if (distance_between_points(point.pt, other.pt) < radius):
            return True
    return False

def merge_keypoints(keypoints, radius):
    result = []
    for point in keypoints:
        if (not has_close_points(point, result, radius)):
            result += [point]

    return result

def maskFrame(frame):

    if (frame is None):
        print("frame is none")
        return

    """
    Perform basic preprocessing to create a mask that can be overlayed over the
    image. This will dramatically reduce the search space for more complicated
    operations further down the pipeline, speeding up computation.

    :param frame: The imput image
    :return: A binary mask corresponding to parts of the image that have
    similar hue, saturation, and value levels as the objects to be detected
    """
    # Blur the image to reduce high frequency noise
    #print(len(frame))
    frame = cv2.GaussianBlur(frame,(5,5),0)

    # Convert the colorspace to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # We want to look for bright, multicolored balls. That means we want to extract parts of the image with:
    # Any hue
    # High saturation
    # High value
    #original
    #return cv2.inRange(hsv,
                       #np.array([0,130,100]),
                       #np.array([255,255,255]))
    #pink
    return cv2.inRange(hsv,
                       np.array([150,130,100]),
                       np.array([200,255,255]))
    
    #yellow
    return cv2.inRange(hsv,
                       np.array([20,130,100]),
                       np.array([50,255,255]))


cap = cv2.VideoCapture(url)
#cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# get vcap property, float(v,h)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

parameters =  aruco.DetectorParameters_create()
fast = cv2.FastFeatureDetector_create()

# Use Aruco Dictionary for 4x4 markers (250)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)

ret, frame = cap.read()
frame = cv2.bitwise_and(frame,frame, mask= maskFrame(frame))
cv2.imwrite("balldetectionframe.jpg", frame)

import time

while(True):
   # Capture frame-by-frame


   ret, frame = cap.read()

   if (frame is None):
      print("frame could not be read.")

      cap.release()
      #cap.open(0)
      cap = cv2.VideoCapture(url)
      time.sleep(3)
      continue
   print(ret, frame)
   cv2.imshow("frame", frame)

   # ARUCO
   corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
   print(corners)

   # Floorballs
   frame = cv2.bitwise_and(frame,frame, mask= maskFrame(frame))
   balls = detect_balls(fast, frame)
   cv2.imshow("mask", frame)
   print(balls)

   if cv2.waitKey(1) & 0xFF == ord('q'):
       break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
