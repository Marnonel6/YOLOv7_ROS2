# Brief description
YOLOv7 is a real-time object detection algorithm that is based on the You Only Look Once (YOLO) architecture and consists of convolutional neural networks (CNNs). A python ROS2 YOLOv7 package was developed with Rintaroh Shima for real-time object detection. Below is a sample video of the custom trained model. An Intel RealSense D435i was mounted on Go1 with its frame specifications at 620x480 and 30fps. The model was downsized and deployed on a Nvidia Jetson Nano on Go1.

# Extra steps
Move `guide_dog.pt` into workspace next to src directory

# Dependencies
     pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
