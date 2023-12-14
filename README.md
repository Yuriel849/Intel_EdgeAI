# Intel_EdgeAI

Different repository from [Intel_Edge_AI_3rd](https://github.com/Yuriel849/Intel_Edge_AI_3rd) and holds code, projects, and work from my studies with Intel instructors.

## Web camera with Linux Ubuntu
```shell
sudo apt install v4l-utils
```
to install necessary library<br>
```shell
v4l2-ctl --list-devices
```
to check camera devices

### * Getting video capture from the first camera device with OpenCV in Python
>cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)<br>
>> "CAP_DSHOW": parameter for Windows systems<br>
"CAP_V4L2": parameter for Linux systems

