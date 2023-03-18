# Brief description
YOLOv7 is a real-time object detection algorithm that is based on the You Only Look Once (YOLO) architecture and consists of convolutional neural networks (CNNs). A python ROS2 YOLOv7 package was developed with Rintaroh Shima for real-time object detection. Below is a sample video of the custom trained model. An Intel RealSense D435i was mounted on Go1 with its frame specifications at 620x480 and 30fps. The model was downsized and deployed on a Nvidia Jetson Nano on Go1.

# Extra steps
Move `guide_dog.pt` into workspace next to src directory

# Dependencies
     pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
     
# Launch file for Demo
`ros2 launch object_detection object_detection.launch.xml use_realsense:=True use_YOLOv7:=True`

# Demo
https://user-images.githubusercontent.com/60977336/226078995-964fbce5-dd42-4553-b531-df5996f69850.mp4

A custom dataset was created (Link above) and hand labeled. In order to ensure that the model was as general as possible the photos where augmented in several different ways (Contrast, stretched, flipped, blurred, etc). Below is the labeled test data and the final models predictions.

![yolov7_test_data](https://user-images.githubusercontent.com/60977336/226084555-74ed7a13-bb35-461a-8069-ade6f4d9e4a5.jpg)

![yolov7_test_data_2](https://user-images.githubusercontent.com/60977336/226084557-815a6129-ed7d-4d68-bfa1-9e4a4e94a4f4.jpg)

The confusion matrix indecates the the model achieved a average true positive rate of 75% and a false negative rate of 25% over all the classes, indicating that the model is able to accurately detect and classify objects in images with a relatively high degree of precision and recall. Furthermore, the model achieved a false positive rate of 0% and a true negative rate of 100% on average, suggesting that it is highly accurate in distinguishing between positive and negative instances. These results are promising and indicate that the model is a reliable and accurate solution for object detection tasks within the classes and in the setting which the data set was taken.

![confusion_matrix](https://user-images.githubusercontent.com/60977336/226084551-653490a5-0fa7-47d6-ba16-141a5a0a84f8.png)

My custom object detection model achieved a mean Average Precision (mAP) score of 0.85 when the intersection over union (IoU) threshold was set at 0.5, indicating good performance in localizing objects. The model also achieved a mAP score of 0.58 across a range of IoU thresholds from 0.5 to 0.95, indicating reasonable performance in detecting objects with varying levels of overlap. The model achieved a precision of 0.88 and a recall of 0.82, indicating that the model is able to accurately identify a high percentage of relevant objects while minimizing the number of false positives. These metrics are important indicators of the overall performance of an object detection model and suggest that the model is well-suited for a variety of applications that require accurate detection and localization of objects.

![results](https://user-images.githubusercontent.com/60977336/226084553-42ea7893-e17f-4351-ab5f-288e6e5c72d3.png)

The model achieved an average F1 score of 0.77 across a confidence range of 0.1 to 0.8. This suggests that the model performs well at a range of confidence levels and is able to accurately detect and classify objects with a high degree of precision and recall.

![yolov7_guide_dog](https://user-images.githubusercontent.com/60977336/226084554-6955c8d6-c14c-4048-9177-05a4b6df1fac.jpg)

