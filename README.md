# wrapperPyKinect2
This is a wrapper of a wrapper, the original wrapper is [PyKinect2](https://github.com/Kinect/PyKinect2).
This code provides a fast and useful way to immediately use **Microsoft Kinect One** with **Python**.

Kinect device -> OS -> SDK Kinect -> PyKinect2 -> wrapperPyKinect2 

***

**Prerequisites:**
* PyKinect2 (and all its dependecies, of course) 
* Microsoft Kinect SDK 2.0 (for Kinect One)
* OpenCV (tested with 2.4.13)

***

**How to**
* `acquisitionKinect.py` needs a class like `frame.py` to save data
* `getframe(frame)`
* That's all!

***

**Data acquired**
* Frame number 
* Frame RGB 
* Frame Depth (16 bit) 
* Frame Depth (8 bit)
* TO DO Skeleton Body Joints

***

**Code sample**
```
...
kinect = AcquisitionKinect()
frame = Frame()
...
while True:
    try:
        kinect.get_frame(frame)
        if frame.frameDepth is None:
            continue
    except:
        ...
```
