# Intel_EdgeAI

Different repository from [Intel_Edge_AI_3rd](https://github.com/Yuriel849/Intel_Edge_AI_3rd) and holds code, projects, and work from my studies with Intel instructors.

## Web camera with Linux Ubuntu
Run in terminal "sudo apt install v4l-utils" to install necessary library
Run in terminal "v4l2-ctl --list-devices" to check camera devices

### * Getting video capture from the first camera device with OpenCV in Python
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
"CAP_DSHOW" is the parameter for Windows systems, use "CAP_V4L2" for Linux systems
