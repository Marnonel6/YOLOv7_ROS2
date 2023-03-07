import rclpy
from rclpy.node import Node
# from std_msgs.msg import Int16MultiArray
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Point

# import argparse
# from pathlib import Path

import cv2
import torch
import numpy as np
import pyrealsense2 as rs
# import requests

# import csv
# import copy
# import itertools
# from collections import Counter
# from collections import deque

# from std_msgs.msg import Int32
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized,\
    TracedModel


class ObjectDetection(Node):
    def __init__(self):
        super().__init__("ObjectDetection")
        # True initial variables - these only get set once

        self.declare_parameter("weights", "guide_dog.pt", ParameterDescriptor(description="Weights file"))
        self.declare_parameter("conf_thres", 0.25, ParameterDescriptor(description="Confidence threshold"))
        self.declare_parameter("iou_thres", 0.45, ParameterDescriptor(description="IOU threshold"))
        self.declare_parameter("device", "cpu", ParameterDescriptor(description="Name of the device"))
        self.declare_parameter("img_size", 640, ParameterDescriptor(description="Image size"))

        self.weights = self.get_parameter("weights").get_parameter_value().string_value
        self.conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        self.iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.img_size = self.get_parameter("img_size").get_parameter_value().integer_value

        self.frequency = 30  # Hz
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

        # Publishers for Classes
        self.pub_person = self.create_publisher(Point, "/person", 10)
        self.person = Point()
        self.pub_door = self.create_publisher(Point, "/door", 10)
        self.door = Point()
        self.pub_stairs = self.create_publisher(Point, "/stairs", 10)
        self.stairs = Point()

        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device) # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1


        # Realsense package
        self.bridge = CvBridge()
        # # RealSense image
        # self.rs_sub = self.create_subscription(Image, '/camera/color/image_raw', self.rs_callback, 10)
        # self.align_depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.align_depth_callback, 10)
        self.rs_sub = self.create_subscription(CompressedImage, '/camera/color/image_raw/compressed', self.rs_callback, 10)
        # self.align_depth_sub = self.create_subscription(CompressedImage, '/camera/aligned_depth_to_color/image_raw/compressed', self.align_depth_callback, 10)


        self.image = None
        self.depth = None
        self.depth_color_map = None
        self.rgb_image = None

        self.camera_RGB = False
        self.camera_depth = False

    def align_depth_callback(self, data):
        self.depth  = self.bridge.imgmsg_to_cv2(data)

        # cv2.waitKey(1)

        # key = cv2.waitKey(10)
        # if key == 27:  # ESC
        #     self.cap.release()
        #     cv2.destroyAllWindows()

        # Get color depth image
        self.depth_color_map = cv2.applyColorMap(cv2.convertScaleAbs(self.depth, alpha=0.08), cv2.COLORMAP_JET)

        self.camera_depth = True

    def rs_callback(self, data):
        # self.image = self.bridge.imgmsg_to_cv2(data)
        self.image = self.bridge.compressed_imgmsg_to_cv2(data)
        self.rgb_image = self.image

        # cv2.waitKey(1)

        # key = cv2.waitKey(10)
        # if key == 27:  # ESC
        #     self.cap.release()
        #     cv2.destroyAllWindows()

        # Get color depth image
        # self.rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # cv2.imshow("YOLOv7 Object detection result RGB", self.image)

        self.camera_RGB = True


    def detect(self):
        # Define image size, format, and frame rate
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Inscrease size, but decrease FPS
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
        # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)

        # Start the pipeline streaming according to the configuration
        pipeline = rs.pipeline()
        profile = pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # Perform per pixel geometric transformation on the data provided
        # Allign depth frame to color
        align_to = rs.stream.color
        align = rs.align(align_to)

        while True:
            # Wait for available frames
            # Frame set includes time synchronized frames of each enabled stream in the pipeline
            frames = pipeline.wait_for_frames()

            # Get aligned frames from RGB-D camera
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            # Convert frames to numpy arrays
            img = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            # Flip image for dog realsense mounted upside down
            # img = cv2.flip(cv2.flip(np.asanyarray(color_frame.get_data()),0),1)
            # depth_image = cv2.flip(cv2.flip(np.asanyarray(depth_frame.get_data()),0),1)

            # Get color depth image
            depth_color_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
            im0 = img.copy()
            img = img[np.newaxis, :, :, :]
            img = np.stack(img, 0)
            img = img[..., ::-1].transpose((0, 3, 1, 2))
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'

                        # self.get_logger().info(f"{xyxy}")

                        if conf > 0.8: # Limit confidence threshold to 80% for all classes
                            # Draw a boundary box around each object
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)
                            plot_one_box(xyxy, depth_color_map, label=label, color=self.colors[int(cls)], line_thickness=2)

                            label_name = f'{self.names[int(cls)]}'

                            # Get box top left & bottom right coordinates
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            x = int((c2[0]+c1[0])/2)
                            y = int((c2[1]+c1[1])/2)

                            # Limit location and distance of object to 480x480 and 5meters away
                            if x < 480 and y < 640 and depth_image[x][y] < 5000:
                                # get depth using x,y coordinates value in the depth matrix
                                profile_stre = profile.get_stream(rs.stream.color)
                                intr = profile_stre.as_video_stream_profile().get_intrinsics()
                                depth_coords = rs.rs2_deproject_pixel_to_point(intr, [x,y], depth_image[x][y])

                                if depth_coords != [0.0,0.0,0.0]:
                                    # Choose label for publishing position Relative to camera frame
                                    if label_name == 'person':
                                        self.person.x = depth_coords[0]*depth_scale
                                        self.person.y = depth_coords[1]*depth_scale
                                        self.person.z = depth_coords[2]*depth_scale # Depth
                                        self.pub_person.publish(self.person)
                                    if label_name == 'door':
                                        self.door.x = depth_coords[0]*depth_scale
                                        self.door.y = depth_coords[1]*depth_scale
                                        self.door.z = depth_coords[2]*depth_scale # Depth
                                        self.pub_door.publish(self.door)
                                    if label_name == 'stairs':
                                        self.stairs.x = depth_coords[0]*depth_scale
                                        self.stairs.y = depth_coords[1]*depth_scale
                                        self.stairs.z = depth_coords[2]*depth_scale # Depth
                                        self.pub_stairs.publish(self.stairs)
                                    self.get_logger().info(f"depth_coord = {depth_coords[0]*depth_scale}  {depth_coords[1]*depth_scale}  {depth_coords[2]*depth_scale}")


# Using cv2.flip() method
# Use Flip code 0 to flip vertically
# image = cv2.flip(src, 0)
                # cv2.imshow("Detection result", cv2.flip(cv2.flip(im0,0),1))
                # cv2.imshow("Detection result depth", cv2.flip(cv2.flip(depth_color_map,0),1))
                cv2.imshow("Detection result", im0)
                cv2.imshow("Detection result depth", depth_color_map)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def YOLOv7_detect(self):

        img = self.rgb_image
        im0 = img.copy()
        img = img[np.newaxis, :, :, :]
        img = np.stack(img, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        t3 = time_synchronized()

        # Process detections   
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'

                    if conf > 0.8: # Limit confidence threshold to 80% for all classes
                        # Draw a boundary box around each object
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)
                        # plot_one_box(xyxy, self.depth_color_map, label=label, color=self.colors[int(cls)], line_thickness=2)

                        # label_name = f'{self.names[int(cls)]}'

                        # # Get box top left & bottom right coordinates
                        # c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        # x = int((c2[0]+c1[0])/2)
                        # y = int((c2[1]+c1[1])/2)

                        # # Limit location and distance of object to 480x480 and 5meters away
                        # if x < 480 and y < 640 and self.depth[x][y] < 5000:
                        #     # get depth using x,y coordinates value in the depth matrix
                        #     profile_stre = profile.get_stream(rs.stream.color)
                        #     intr = profile_stre.as_video_stream_profile().get_intrinsics()
                        #     depth_coords = rs.rs2_deproject_pixel_to_point(intr, [x,y], self[x][y])

                        #     if depth_coords != [0.0,0.0,0.0]:
                        #         # Choose label for publishing position Relative to camera frame
                        #         if label_name == 'person':
                        #             self.person.x = depth_coords[0]*depth_scale
                        #             self.person.y = depth_coords[1]*depth_scale
                        #             self.person.z = depth_coords[2]*depth_scale # Depth
                        #             self.pub_person.publish(self.person)
                        #         if label_name == 'door':
                        #             self.door.x = depth_coords[0]*depth_scale
                        #             self.door.y = depth_coords[1]*depth_scale
                        #             self.door.z = depth_coords[2]*depth_scale # Depth
                        #             self.pub_door.publish(self.door)
                        # #         if label_name == 'stairs':
                        # #             self.stairs.x = depth_coords[0]*depth_scale
                        # #             self.stairs.y = depth_coords[1]*depth_scale
                        # #             self.stairs.z = depth_coords[2]*depth_scale # Depth
                        # #             self.pub_stairs.publish(self.stairs)
                        #         # self.get_logger().info(f"depth_coord = {depth_coords[0]*depth_scale}  {depth_coords[1]*depth_scale}  {depth_coords[2]*depth_scale}")

            cv2.imshow("YOLOv7 Object detection result RGB", im0)
            # cv2.imshow("YOLOv7 Object detection result Depth", self.depth_color_map)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def timer_callback(self):
        self.detect()

        # if self.camera_RGB == True: #and self.camera_depth == True:
        #     self.YOLOv7_detect()

def main(args=None):
    """Run the main function."""
    rclpy.init(args=args)
    with torch.no_grad():
        node = ObjectDetection()
        rclpy.spin(node)
        rclpy.shutdown()

if __name__ == '__main__':
    main()
