import cv2
import numpy as np

# Read from the first camera device
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

w = 640 #1280#1920
h = 480 #720#1080

cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

# If video device was successfully opened
while(cap.isOpened()):
    # Read one frame
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display
    cv2.imshow("Camera", frame)

    # Wait 1ms for key entry and terminate if 'q' is entered
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
