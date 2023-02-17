import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray

import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
import pyrealsense2 as rs

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized,\
    TracedModel


class YoloV7():

    def __init__(self, weights, conf_thres, iou_thres, device, img_size):
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres


        # self.get_logger().info("HALLOOOOOO ????")

        # source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

        # # Directories
        # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device) # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(img_size, s=stride)  # check img_size

        # if trace:
        #     self.model = TracedModel(self.model, device, img_size)

        if self.half:
            self.model.half()  # to FP16

        # # Second-stage classifier
        # classify = False
        # if classify:
        #     modelc = load_classifier(name='resnet101', n=2)  # initialize
        #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

    def detect(self):
        # Define image size, format, and frame rate
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

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

        # while True:
        # Wait for available frames
        # Frame set includes time synchronized frames of each enabled stream in the pipeline
        frames = pipeline.wait_for_frames()
        # Get aligned frames from RGB-D camera
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        # if not depth_frame or not color_frame:
            # continue
        # Convert frames to numpy arrays
        img = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
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
        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
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
                    # Draw a boundary box around each object
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)
                    plot_one_box(xyxy, depth_color_map, label=label, color=self.colors[int(cls)], line_thickness=2)
                    # Get box top left & bottom right coordinates
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                x = int((c2[0]+c1[0])/2)
                y = int((c2[1]+c1[1])/2)
                # print(f"c1 = {c1}, c2 = {c2}")
                print(f"x = {x}, y = {y}")
                if x < 480 and y < 480: #and depth_image[x][y] < 1000:
                    # get depth using x,y coordinates value in the depth matrix
                    profile_stre = profile.get_stream(rs.stream.color)
                    intr = profile_stre.as_video_stream_profile().get_intrinsics()
                    depth_coords = rs.rs2_deproject_pixel_to_point(intr, [x,y], depth_image[x][y])
                    if depth_coords != [0.0,0.0,0.0]:
                        print(f"depth_coord = {depth_coords[0]*depth_scale}  {depth_coords[1]*depth_scale}  {depth_coords[2]*depth_scale}")
                        # self.get_logger().info(f"depth_coord = {depth_coords[0]*depth_scale}  {depth_coords[1]*depth_scale}  {depth_coords[2]*depth_scale}")
            cv2.imshow("Detection result", im0)
            cv2.imshow("Detection result depth", depth_color_map)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

class ObjectDetection(Node):
    def __init__(self):
        # while True:
        super().__init__("ObjectDetection")
        # True initial variables - these only get set once
        self.frequency = 1000  # Hz
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detection = YoloV7('yolov7.pt', 0.25, 0.45, self.device, 640)
        self.get_logger().info(f"depth_coord")

    def timer_callback(self):
        self.detection.detect()

def main(args=None):
    """Run the main function."""
    rclpy.init(args=args)
    # with torch.no_grad():
        # obj_detected = YoloV7('yolov7.pt', 0.25, 0.45, device, 640)
    node = ObjectDetection()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
#     opt, unknown = parser.parse_known_args()
#     print(opt)
#     #check_requirements(exclude=('pycocotools', 'thop'))

#     with torch.no_grad():
#         if opt.update:  # update all models (to fix SourceChangeWarning)
#             for opt.weights in ['yolov7.pt']:
#                 main()
#                 strip_optimizer(opt.weights)
#         else:
#             main()