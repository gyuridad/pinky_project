#!/usr/bin/env python3
"""
aruco_map_register_once.py

- Input:
  - AMCL/TF: map -> base_frame  (e.g., base_footprint)
  - TF:       base_frame -> camera_frame  (extrinsic)
  - Topic:    /aruco_pose (PoseStamped), frame_id == camera_frame
- Output:
  - Prints ONE-LINE command for:
      ros2 run tf2_ros static_transform_publisher x y z qx qy qz qw map aruco_600

Stability policy (simple & practical):
- Require N consecutive ticks where map->base pose changes are small (xy/yaw thresholds),
  and /aruco_pose is fresh.
"""

import math
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

import tf2_ros
from tf2_ros import TransformException

from geometry_msgs.msg import PoseStamped


def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)],
    ], dtype=np.float64)


def rot_to_quat(R: np.ndarray):
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return (float(qx), float(qy), float(qz), float(qw))


def T_from_tfmsg(tfmsg) -> np.ndarray:
    t = tfmsg.transform.translation
    q = tfmsg.transform.rotation
    R = quat_to_rot(q.x, q.y, q.z, q.w)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float64)
    return T


def T_from_posemsg(msg: PoseStamped) -> np.ndarray:
    p = msg.pose.position
    q = msg.pose.orientation
    R = quat_to_rot(q.x, q.y, q.z, q.w)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([p.x, p.y, p.z], dtype=np.float64)
    return T


def yaw_from_R(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class ArucoMapRegisterOnce(Node):
    def __init__(self):
        super().__init__("aruco_map_register_once")

        # ---- Params ----
        self.declare_parameter("aruco_id", 600)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")         # or base_link
        self.declare_parameter("camera_frame", "front_camera_link")

        self.declare_parameter("aruco_pose_topic", "/aruco_pose")      # PoseStamped from vision node
        self.declare_parameter("aruco_frame_prefix", "aruco_")         # output frame name = aruco_600

        # stability gating
        self.declare_parameter("control_hz", 5.0)
        self.declare_parameter("need_stable_ticks", 10)                # e.g., 10 ticks @5Hz => 2s stable
        self.declare_parameter("stable_xy_m", 0.02)                    # map->base change threshold
        self.declare_parameter("stable_yaw_deg", 2.0)                  # map->base yaw change threshold
        self.declare_parameter("pose_fresh_sec", 0.5)                  # /aruco_pose freshness

        self.declare_parameter("print_precision", 6)                   # decimals in command

        # ---- Read ----
        self.aruco_id = int(self.get_parameter("aruco_id").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)

        self.aruco_pose_topic = str(self.get_parameter("aruco_pose_topic").value)
        self.aruco_prefix = str(self.get_parameter("aruco_frame_prefix").value)
        self.aruco_frame = f"{self.aruco_prefix}{self.aruco_id}"

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.need_stable_ticks = int(self.get_parameter("need_stable_ticks").value)
        self.stable_xy_m = float(self.get_parameter("stable_xy_m").value)
        self.stable_yaw_deg = float(self.get_parameter("stable_yaw_deg").value)
        self.pose_fresh_sec = float(self.get_parameter("pose_fresh_sec").value)

        self.print_precision = int(self.get_parameter("print_precision").value)

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Subscribe pose ----
        self._latest_pose: Optional[PoseStamped] = None
        self._latest_pose_time = 0.0
        self.pose_sub = self.create_subscription(
            PoseStamped, self.aruco_pose_topic, self._on_pose, 10
        )

        # ---- Stability tracking ----
        self._prev_map_base = None
        self._stable_count = 0
        self._done = False

        dt = 1.0 / max(1.0, self.control_hz)
        self.timer = self.create_timer(dt, self._tick)

        self.get_logger().info(
            f"[ArucoMapRegisterOnce] waiting stable AMCL + marker...\n"
            f"  map_frame={self.map_frame}, base_frame={self.base_frame}, camera_frame={self.camera_frame}\n"
            f"  aruco_id={self.aruco_id}, aruco_frame={self.aruco_frame}, pose_topic={self.aruco_pose_topic}\n"
            f"  stable: need={self.need_stable_ticks} ticks, xy<{self.stable_xy_m}m, yaw<{self.stable_yaw_deg}deg"
        )

    def _on_pose(self, msg: PoseStamped):
        if msg.header.frame_id != self.camera_frame:
            self.get_logger().warn(
                f"/aruco_pose frame_id mismatch: got '{msg.header.frame_id}', expected '{self.camera_frame}'"
            )
            return
        self._latest_pose = msg
        self._latest_pose_time = time.time()

    def _lookup_T_map_base(self) -> np.ndarray:
        tf_map_base = self.tf_buffer.lookup_transform(
            self.map_frame, self.base_frame, rclpy.time.Time()
        )
        return T_from_tfmsg(tf_map_base)

    def _lookup_T_base_cam(self) -> np.ndarray:
        tf_base_cam = self.tf_buffer.lookup_transform(
            self.base_frame, self.camera_frame, rclpy.time.Time()
        )
        return T_from_tfmsg(tf_base_cam)

    def _is_pose_fresh(self) -> bool:
        if self._latest_pose is None:
            return False
        return (time.time() - self._latest_pose_time) <= self.pose_fresh_sec

    def _tick(self):
        if self._done:
            return

        # 1) Need fresh marker observation
        if not self._is_pose_fresh():
            self._stable_count = 0
            return

        # 2) Need map->base + base->cam available (AMCL + extrinsic)
        try:
            T_map_base = self._lookup_T_map_base()
            _ = self._lookup_T_base_cam()
        except TransformException as e:
            self._stable_count = 0
            self.get_logger().warn(f"TF not ready: {e}")
            return

        # 3) Stability check on map->base (AMCL 안정)
        if self._prev_map_base is None:
            self._prev_map_base = T_map_base
            self._stable_count = 0
            return

        dx = float(T_map_base[0, 3] - self._prev_map_base[0, 3])
        dy = float(T_map_base[1, 3] - self._prev_map_base[1, 3])
        dxy = math.hypot(dx, dy)

        yaw_now = yaw_from_R(T_map_base[:3, :3])
        yaw_prev = yaw_from_R(self._prev_map_base[:3, :3])
        dyaw = abs(wrap_pi(yaw_now - yaw_prev))

        if (dxy <= self.stable_xy_m) and (math.degrees(dyaw) <= self.stable_yaw_deg):
            self._stable_count += 1
        else:
            self._stable_count = 0

        self._prev_map_base = T_map_base

        if self._stable_count < self.need_stable_ticks:
            return

        # 4) Compute map->aruco once
        try:
            T_base_cam = self._lookup_T_base_cam()
        except TransformException as e:
            self.get_logger().warn(f"TF base->cam failed: {e}")
            self._stable_count = 0
            return

        T_cam_aruco = T_from_posemsg(self._latest_pose)

        # T_map_aruco = T_map_base * T_base_cam * T_cam_aruco
        T_map_aruco = T_map_base @ T_base_cam @ T_cam_aruco

        x, y, z = float(T_map_aruco[0, 3]), float(T_map_aruco[1, 3]), float(T_map_aruco[2, 3])
        qx, qy, qz, qw = rot_to_quat(T_map_aruco[:3, :3])

        p = self.print_precision
        cmd = (
            "ros2 run tf2_ros static_transform_publisher "
            f"{x:.{p}f} {y:.{p}f} {z:.{p}f} "
            f"{qx:.{p}f} {qy:.{p}f} {qz:.{p}f} {qw:.{p}f} "
            f"{self.map_frame} {self.aruco_frame}"
        )

        self.get_logger().warn("========== REGISTER COMMAND (copy & run) ==========")
        print(cmd)
        self.get_logger().warn("===================================================")
        self._done = True


def main():
    rclpy.init()
    node = ArucoMapRegisterOnce()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


### 실행 예시:
# ros2 run <your_pkg> aruco_map_register_once --ros-args \
#   -p aruco_id:=600 \
#   -p base_frame:=base_footprint \
#   -p camera_frame:=front_camera_link \
#   -p aruco_pose_topic:=/aruco_pose

### 마커가 보이면, 터미널에 이런 한 줄이 뜸:
# ros2 run tf2_ros static_transform_publisher x y z qx qy qz qw map aruco_600

### 그대로 복사해서 실행(또는 launch에 넣기)하면 map->aruco_600 등록 완료