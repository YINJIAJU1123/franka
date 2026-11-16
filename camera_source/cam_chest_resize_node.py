#!/usr/bin/python3
"""
Resize node for D405 (cam_chest).

D405 natively publishes image_rect_raw at 848x480.
This node subscribes to that topic, resizes to 640x480,
and republishes as /camera/cam_chest/color/image_raw
so that both cameras have consistent 640x480 image_raw topics.

Why a separate node instead of setting depth_module.color_profile:
  Explicitly setting depth_module.color_profile for D405 on a shared USB 2.0
  hub triggers a USB bus reset that crashes both cameras. Software resize is
  the stable workaround.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CamChestResizeNode(Node):
    def __init__(self):
        super().__init__('cam_chest_resize')
        self.bridge = CvBridge()

        self.pub = self.create_publisher(
            Image,
            '/camera/cam_chest/color/image_raw',
            10,
        )
        self.sub = self.create_subscription(
            Image,
            '/camera/cam_chest/color/image_rect_raw',
            self._callback,
            10,
        )
        self.get_logger().info(
            'cam_chest_resize: 848x480 image_rect_raw -> 640x480 image_raw'
        )

    def _callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            resized = cv2.resize(cv_img, (640, 480), interpolation=cv2.INTER_LINEAR)
            out_msg = self.bridge.cv2_to_imgmsg(resized, encoding=msg.encoding)
            out_msg.header = msg.header
            self.pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'resize error: {e}')


def main():
    rclpy.init()
    node = CamChestResizeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
