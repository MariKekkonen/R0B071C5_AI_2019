import numpy as np
import cv2
from cv2 import aruco 

# Install numpy and opencv including aruco with pip
# -> pip install numpy
# -> pip install opencv-contrib-python
# https://stackoverflow.com/questions/45972357/python-opencv-aruco-no-module-named-cv2-aruco

# To use webcam -> cv2.VideoCapture(0) 
cap = cv2.VideoCapture("udp://@224.0.0.0:1234") # video from udp stream
#cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Get the video dimensios
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(width, height)

# Default parameters are enough
parameters =  aruco.DetectorParameters_create()
# Using aruco 4x4 with ids 0 - 49
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# You can generate arucos from
# http://chev.me/arucogen/

# We will give your personal aruco markers later. 
# (The size of the marker will be 12cm 
# with 1.5 cm white borders. So 15cm x 15cm. 
# Note: The size is not finalized yet. 
# So the dimension might chance slightly.)

# Each aruco code represents a number (id) between 0 - 49.
# Each teams will receive 2 aruco markers, one for each bot.

# IDs of the teams will be from 10 to 30. 
# e.g. Your ids could be 14 & 15. 
# Then the ids of enemies will be 10-13 and 16-30 

# The video will be 60 fps. Sometimes it might be clever to
# skip some frames. Not needed though. Depending on the performance of
# the computer and algorithms.
skipCounter = 0

while(True):

    # Take the frame
    cap.grab()

    # Without "True or", every 2nd frame wouldn't be proccessed
    skipCounter += 1
    if True or skipCounter % 2 == 0:

        # Retrive the frame
        ret, frame = cap.read()

        # Run the aruco decection
        # Return lists of the detected arucos
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if ids is not None:
            print("Aruco(s) spotted", ids, corners)
        else:
            print("-")
        
         #Display the resulting frame
        cv2.imshow("frame", frame)
    
            
    # Quitting 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close everything
cap.release()
cv2.destroyAllWindows()
