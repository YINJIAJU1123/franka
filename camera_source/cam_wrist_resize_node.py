#!/usr/bin/python3
"""
Resize node for the wrist RealSense camera (cam_wrist).  # by Jiaju 3.19

This node subscribes to the native image_rect_raw topic and resizes to 640x480,
and republishes as /camera/cam_wrist/color/image_raw
so that both cameras have consistent 640x480 image_raw topics.

The software resize keeps the wrist output at the same 640x480 size expected
by downstream consumers, regardless of the camera's native color profile.
"""

import copy

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import cv2


class CamWristResizeNode(Node):  # by Jiaju 3.19
    def __init__(self):
        super().__init__('cam_wrist_resize')
        self.bridge = CvBridge()
        self.target_width = 640
        self.target_height = 480
        self.latest_resized_info = None

        self.pub = self.create_publisher(
            Image,
            '/camera/cam_wrist/color/image_raw',
            10,
        )
        self.info_pub = self.create_publisher(
            CameraInfo,
            '/camera/cam_wrist/color/camera_info_resized',
            10,
        )
        self.wrist_info_pub = self.create_publisher(
            CameraInfo,
            '/wrist/camera_info',
            10,
        )
        self.sub = self.create_subscription(
            Image,
            '/camera/cam_wrist/color/image_rect_raw',
            self._callback,
            10,
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/cam_wrist/color/camera_info',
            self._info_callback,
            10,
        )
        self.get_logger().info(
            'cam_wrist_resize: image_rect_raw -> 640x480 image_raw; '
            'publishing resized CameraInfo to /camera/cam_wrist/color/camera_info_resized '
            'and /wrist/camera_info'
        )

    def _scale_camera_info(self, msg: CameraInfo) -> CameraInfo:
        out = CameraInfo()
        out.header = msg.header
        out.height = self.target_height
        out.width = self.target_width
        out.distortion_model = msg.distortion_model
        out.d = list(msg.d)
        out.r = list(msg.r)
        out.binning_x = msg.binning_x
        out.binning_y = msg.binning_y

        sx = self.target_width / float(msg.width) if msg.width else 1.0
        sy = self.target_height / float(msg.height) if msg.height else 1.0
        out.roi.x_offset = int(round(msg.roi.x_offset * sx))
        out.roi.y_offset = int(round(msg.roi.y_offset * sy))
        out.roi.width = int(round(msg.roi.width * sx))
        out.roi.height = int(round(msg.roi.height * sy))
        out.roi.do_rectify = msg.roi.do_rectify

        out.k = list(msg.k)
        out.k[0] *= sx
        out.k[2] *= sx
        out.k[4] *= sy
        out.k[5] *= sy

        out.p = list(msg.p)
        out.p[0] *= sx
        out.p[2] *= sx
        out.p[5] *= sy
        out.p[6] *= sy
        return out

    def _info_callback(self, msg: CameraInfo):
        self.latest_resized_info = self._scale_camera_info(msg)
        self.info_pub.publish(self.latest_resized_info)
        self.wrist_info_pub.publish(self.latest_resized_info)

    def _callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            resized = cv2.resize(
                cv_img,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            out_msg = self.bridge.cv2_to_imgmsg(resized, encoding=msg.encoding)
            out_msg.header = msg.header
            self.pub.publish(out_msg)
            if self.latest_resized_info is not None:
                info_msg = copy.deepcopy(self.latest_resized_info)
                info_msg.header = msg.header
                self.info_pub.publish(info_msg)
                self.wrist_info_pub.publish(info_msg)
        except Exception as e:
            self.get_logger().error(f'resize error: {e}')


def main():
    rclpy.init()
    node = CamWristResizeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node.handle:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
