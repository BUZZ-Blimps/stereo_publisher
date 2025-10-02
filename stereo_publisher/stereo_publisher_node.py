#!/usr/bin/env python3

import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

import yaml

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

cv2.setUseOptimized(True)
# cv2.setNumThreads(2)

class StereoPublisher(Node):
    def __init__(self):
        super().__init__('stereo_publisher')

        self.get_logger().info('Started stereo publisher node')

        self.running = True

        self._declare_parameters()
        self._get_parameters()

        # Initialize capture device.
        self.cap = cv2.VideoCapture(self.device_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera at {self.device_path}')
            return

        self._setup_camera()
        self._load_calibration()
        self._setup_stereorectification()

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, "stereo/image_raw", 15)

        self.left_pub = self.create_publisher(Image, "left/image_raw", 15)
        self.right_pub = self.create_publisher(Image, "right/image_raw", 15)

        # self.left_rect_pub = self.create_publisher(Image, "left/image_rect", 15)
        # self.right_rect_pub = self.create_publisher(Image, "right/image_rect", 15)

        # Create a camera timer callback at 15 Hz
        self.timer = self.create_timer(1 / 15.0, self.camera_callback)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def _declare_parameters(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_number', 1),
                ('device_path', '/dev/video2'),
                ('calibration_path', 'package://stereo_publisher/calibration/'),
                ('verbose_mode', False),
                ('save_frames', False),
                ('save_location', 'frames/'),
                ('undistort_camera', True)
            ]
        )

    def _get_parameters(self):
        self.camera_number = self.get_parameter('camera_number').value
        self.device_path = self.get_parameter('device_path').value
        self.calibration_path = self.get_parameter('calibration_path').value
        self.verbose_mode = self.get_parameter('verbose_mode').value
        self.save_frames = self.get_parameter('save_frames').value
        self.save_location = self.get_parameter('save_location').value
        self.undistort_camera = self.get_parameter('undistort_camera').value

    def _setup_camera(self):
        """Configure the camera properties."""
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _load_calibration(self):
        """Load camera calibration parameters for left and right cameras using the new approach."""
        try:
            # Left camera calibration.
            left_file = f'camera{self.camera_number}_left.yaml'
            left_path = os.path.join(self.calibration_path, left_file)
            with open(left_path, 'r') as file:
                left_data = yaml.safe_load(file)
                self.left_camera_matrix = np.array(left_data['camera_matrix']['data']).reshape(
                    left_data['camera_matrix']['rows'], left_data['camera_matrix']['cols'])
                self.left_distortion_coefficients = np.array(left_data['distortion_coefficients']['data'])
                self.left_rotation_matrix = np.array(left_data['rotation_matrix']['data']).reshape(
                    left_data['rotation_matrix']['rows'], left_data['rotation_matrix']['cols'])
                self.left_translation_vector = np.array(left_data['translation_vector']['data']).reshape(
                    left_data['translation_vector']['rows'], 1)
                self.left_image_size = tuple(left_data['image_size'])

            # Right camera calibration.
            right_file = f'camera{self.camera_number}_right.yaml'
            right_path = os.path.join(self.calibration_path, right_file)
            with open(right_path, 'r') as file:
                right_data = yaml.safe_load(file)
                self.right_camera_matrix = np.array(right_data['camera_matrix']['data']).reshape(
                    right_data['camera_matrix']['rows'], right_data['camera_matrix']['cols'])
                self.right_distortion_coefficients = np.array(right_data['distortion_coefficients']['data'])
                self.right_rotation_matrix = np.array(right_data['rotation_matrix']['data']).reshape(
                    right_data['rotation_matrix']['rows'], right_data['rotation_matrix']['cols'])
                self.right_translation_vector = np.array(right_data['translation_vector']['data']).reshape(
                    right_data['translation_vector']['rows'], 1)
                self.right_image_size = tuple(right_data['image_size'])

            # Use the vertical focal length (the [1,1] element) from the calibration file.
            self.goal_vertical_focal = self.left_camera_matrix[1, 1]

            self.h_fov = 2 * np.arctan(((self.left_image_size[1] / 2) / self.left_camera_matrix[0][0]))
            self.v_fov = 2 * np.arctan(((self.left_image_size[0] / 2) / self.left_camera_matrix[1][1]))

            self.fx = self.left_camera_matrix[0][0]
            self.fy = self.left_camera_matrix[1][1]

            self.cx = self.left_camera_matrix[0][2]
            self.cy = self.left_camera_matrix[1][2]

            self.get_logger().info('Successfully loaded camera calibration files')
        except Exception as e:
            self.get_logger().error(f'Failed to load calibration files: {e}')

    def _setup_stereorectification(self):
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.left_camera_matrix,
            self.left_distortion_coefficients,
            self.right_camera_matrix,
            self.right_distortion_coefficients,
            (640, 480),
            self.right_rotation_matrix,
            self.right_translation_vector,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha = 0
        )

        # Compute rectification maps if undistortion is not desired.
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix,
            self.left_distortion_coefficients,
            R1,
            P1,
            (640, 480),
            cv2.CV_32FC1)
            
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix,
            self.right_distortion_coefficients,
            R2,
            P2,
            (640, 480),
            cv2.CV_32FC1)

    def capture_frame(self):
        """Capture a frame from the camera."""
        return self.cap.read()

    def camera_callback(self):

        """Main callback: capture, process, stream frames, and publish metrics and grid distances."""
        timing = {}
        t_total = time.time()
        t_preprocess_start = time.time()

        ret, frame = self.capture_frame()
        if not ret:
            self.get_logger().error('Failed to capture frame')
            return

        # Split frame into left and right images.
        frame_width = frame.shape[1] // 2
        left_frame = frame[:, :frame_width]
        right_frame = frame[:, frame_width:]

        left_rect = cv2.remap(left_frame, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_frame, self.right_map1, self.right_map2, cv2.INTER_LINEAR)

        image_msg = self.bridge.cv2_to_imgmsg(left_frame, "bgr8")

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        self.left_pub.publish(image_msg)
        self.right_pub.publish(self.bridge.cv2_to_imgmsg(right_frame, "bgr8"))

        # self.left_rect_pub.publish(self.bridge.cv2_to_imgmsg(left_rect, "bgr8"))
        # self.right_rect_pub.publish(self.bridge.cv2_to_imgmsg(right_rect, "bgr8"))

        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

        # cv2.imshow('left', left_frame)
        # cv2.imshow('left rect', left_rect)
        # cv2.imshow('right', left_frame)
        # cv2.imshow('right rect', left_rect)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            self.running = False

def main(args=None):
    rclpy.init(args=args)
    stereo_publisher = StereoPublisher()
    while stereo_publisher.running:
        rclpy.spin_once(stereo_publisher)
    # rclpy.spin(stereo_publisher)
    stereo_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
