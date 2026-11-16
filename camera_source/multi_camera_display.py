#!/usr/bin/python3
"""
2x2 live display for the four RealSense cameras.

Shows one OpenCV window with per-camera title and measured receive FPS.
Use /usr/bin/python3 with ROS Humble to avoid conda Python ABI issues.
"""

import argparse
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


CAMERAS = [
    ("cam_wrist", "wrist", "/camera/cam_wrist"),
    ("cam_head", "head", "/camera/cam_head"),
    ("cam_left", "left", "/camera/cam_left"),
    ("cam_right", "right", "/camera/cam_right"),
]


class MultiCameraDisplay(Node):
    def __init__(self, mode: str, cell_width: int, cell_height: int):
        super().__init__(f"multi_camera_display_{mode}")
        self.mode = mode
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.bridge = CvBridge()
        self.frames = {}
        self.last_stamp = {}
        self.fps = {}
        self.window_name = f"RealSense 4-camera {mode}"

        for camera_name, label, base_topic in CAMERAS:
            topic = self._topic_for(base_topic)
            self.create_subscription(
                Image,
                topic,
                lambda msg, key=camera_name: self._image_callback(key, msg),
                qos_profile_sensor_data,
            )
            self.frames[camera_name] = None
            self.last_stamp[camera_name] = None
            self.fps[camera_name] = None
            self.get_logger().info(f"subscribe {label}: {topic}")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.cell_width * 2, self.cell_height * 2)
        self.create_timer(1.0 / 30.0, self._draw)

    def _topic_for(self, base_topic: str) -> str:
        if self.mode == "depth":
            return f"{base_topic}/depth/image_rect_raw"
        return f"{base_topic}/color/image_raw"

    def _image_callback(self, camera_name: str, msg: Image):
        now = time.monotonic()
        last = self.last_stamp[camera_name]
        if last is not None:
            dt = now - last
            if dt > 0:
                inst_fps = 1.0 / dt
                old_fps = self.fps[camera_name]
                self.fps[camera_name] = inst_fps if old_fps is None else old_fps * 0.9 + inst_fps * 0.1
        self.last_stamp[camera_name] = now

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.frames[camera_name] = self._to_bgr(image, msg.encoding)
        except Exception as exc:
            self.get_logger().error(f"{camera_name} convert error: {exc}")

    def _to_bgr(self, image: np.ndarray, encoding: str) -> np.ndarray:
        encoding = (encoding or "").lower()
        if self.mode == "depth" or encoding in ("16uc1", "mono16"):
            depth = image.astype(np.float32)
            valid = depth[np.isfinite(depth) & (depth > 0)]
            if valid.size:
                vmax = np.percentile(valid, 95)
                vmax = max(float(vmax), 1.0)
            else:
                vmax = 1.0
            normalized = np.clip(depth * (255.0 / vmax), 0, 255).astype(np.uint8)
            return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def _draw(self):
        cells = []
        for camera_name, label, _ in CAMERAS:
            frame = self.frames.get(camera_name)
            if frame is None:
                cell = np.full((self.cell_height, self.cell_width, 3), 45, dtype=np.uint8)
                cv2.putText(
                    cell,
                    "waiting...",
                    (24, self.cell_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (210, 210, 210),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cell = cv2.resize(frame, (self.cell_width, self.cell_height), interpolation=cv2.INTER_AREA)

            self._overlay(cell, camera_name, label)
            cells.append(cell)

        top = np.hstack([cells[0], cells[1]])
        bottom = np.hstack([cells[2], cells[3]])
        canvas = np.vstack([top, bottom])
        cv2.imshow(self.window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.get_logger().info("display window requested shutdown")
            rclpy.shutdown()

    def _overlay(self, cell: np.ndarray, camera_name: str, label: str):
        cv2.rectangle(cell, (0, 0), (self.cell_width, 42), (0, 0, 0), thickness=-1)
        fps = self.fps.get(camera_name)
        fps_text = "--.- FPS" if fps is None else f"{fps:4.1f} FPS"
        title = f"{label}  {camera_name}"
        cv2.putText(
            cell,
            title,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            cell,
            fps_text,
            (self.cell_width - 150, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 180),
            2,
            cv2.LINE_AA,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("color", "depth"), default="color")
    parser.add_argument("--cell-width", type=int, default=480)
    parser.add_argument("--cell-height", type=int, default=360)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = MultiCameraDisplay(args.mode, args.cell_width, args.cell_height)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        if node.handle:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
