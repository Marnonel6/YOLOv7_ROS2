# YOLOv7 ROS2 Package
This repository contains a ROS2 package for running YOLOv7.
## Dependencies
Run the following to install PyTorch:

    pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
## Extra Step
Move `yolov7-tiny.pt` into the workspace next to the src directory
## Launch File
Run `ros2 launch object_detection object_detection.launch.xml` to start the object_detection node
## Demo

[Alt-Text](https://user-images.githubusercontent.com/113070827/226159680-f46db3a1-692d-4c00-b32c-ffc60f6270b5.mp4)