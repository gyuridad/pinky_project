#!/usr/bin/env python3
"""
marker_based_amcl_corrector.py

- 가정:
  1) map -> aruco_600  (static TF로 "마커 절대좌표" 등록 완료)
  2) front_camera_link -> aruco_600  (관측 TF 발행 중)
  3) base_footprint -> front_camera_link (고정 extrinsic TF 존재)

- 동작:
  - 위 TF 3개로 수식:
      T_map_base = T_map_aruco * inv(T_cam_aruco) * inv(T_base_cam)
    로 마커 기반 로봇 포즈(map->base_footprint)를 계산
  - 조건(연속 N프레임, 재보정 쿨다운 등) 만족 시 /initialpose 로 publish하여 AMCL 보정

실행:
  python3 marker_based_amcl_corrector.py

확인:
  ros2 topic echo /initialpose
  (그리고 nav2/amcl이 map->odom을 재정렬하는지 확인)
"""

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformException

from geometry_msgs.msg import PoseWithCovarianceStamped
from tf_transformations import quaternion_from_euler

def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    """Quaternion(x,y,z,w) -> 3x3 rotation matrix"""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )

def tfmsg_to_T(tfmsg) -> np.ndarray:
    """geometry_msgs/TransformStamped -> 4x4 homogeneous matrix"""
    t = tfmsg.transform.translation
    q = tfmsg.transform.rotation
    R = quat_to_rot(q.x, q.y, q.z, q.w)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float64)
    return T

def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of rigid transform"""
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti

def yaw_from_R(R: np.ndarray) -> float:
    """Yaw from rotation matrix (yaw around +Z)"""
    return math.atan2(R[1, 0], R[0, 0])

def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

class MarkerAMCLCorrector(Node):
    def __init__(self):
        super().__init__("marker_amcl_corrector")

        # ==============================
        # ROS2 Parameter (단일 마커)
        # ==============================
        self.declare_parameter("aruco_ids", [600, 601])
        self.aruco_ids = list(self.get_parameter("aruco_ids").value)
        self.aruco_frames = [f"aruco_{int(mid)}" for mid in self.aruco_ids]
        self.get_logger().info(f"Using ArUco markers: {self.aruco_frames}")

        # ==============================
        # Frames 
        # ==============================
        self.map_frame = "map"
        self.base_frame = "base_footprint"
        self.cam_frame = "front_camera_link"

        # ==============================
        # Publisher (/initialpose)
        # ==============================
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )

        # ==============================
        # TF
        # ==============================
        self.use_latest_time = True
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=2.0)
        )
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ==============================
        # Control parameters
        # ==============================
        self.control_hz = 5.0
        self.min_consecutive = 3
        self.cooldown_sec = 5.0

        self.max_jump_xy = 0.5        # m
        self.max_jump_yaw_deg = 45.0  # deg

        self.var_xy = 0.02
        self.var_yaw = 0.003

        self._consec_ok = 0
        self._last_publish_time = 0.0

        self.timer = self.create_timer(
            1.0 / self.control_hz, self.on_timer
        )

    # --------------------------------
    # TF lookup helpers
    # --------------------------------
    # “보이는 마커 선택” 함수 추가 (가장 가까운 z 기준)
    def select_visible_marker(self):
        """
        cam_frame -> aruco_frame TF가 가능한 마커 중
        z(거리)가 가장 작은(가장 가까운) 마커를 선택
        return: (aruco_frame, z) or (None, None)
        """
        t = self._tf_time()
        best = None
        best_z = None

        for af in self.aruco_frames:
            try:
                tf_cam_aruco_msg = self.tf_buffer.lookup_transform(self.cam_frame, af, t)
            except TransformException:
                continue

            T_cam_aruco = tfmsg_to_T(tf_cam_aruco_msg)
            z = float(T_cam_aruco[2, 3])

            # z가 음수/비정상일 때 걸러도 됨 (선택)
            if z <= 0.01:
                continue

            if (best_z is None) or (z < best_z):
                best = af
                best_z = z

        return best, best_z

    def _tf_time(self) -> rclpy.time.Time:
        # 최신값을 원하면 Time(seconds=0) 가 안정적인 편 (static + dynamic 혼합시)
        if self.use_latest_time:
            return rclpy.time.Time(seconds=0)
        return rclpy.time.Time()

    def lookup_all_T(self, aruco_frame: str):
        t = self._tf_time()
        tf_map_aruco_msg = self.tf_buffer.lookup_transform(
            self.map_frame, aruco_frame, t
        )
        tf_cam_aruco_msg = self.tf_buffer.lookup_transform(
            self.cam_frame, aruco_frame, t
        )
        tf_base_cam_msg = self.tf_buffer.lookup_transform(
            self.base_frame, self.cam_frame, t
        )
        T_map_aruco = tfmsg_to_T(tf_map_aruco_msg)
        T_cam_aruco = tfmsg_to_T(tf_cam_aruco_msg)
        T_base_cam = tfmsg_to_T(tf_base_cam_msg)
        return T_map_aruco, T_cam_aruco, T_base_cam
    
    def compute_marker_based_base_pose(self, aruco_frame: str):
        """
        T_map_base = T_map_aruco * inv(T_cam_aruco) * inv(T_base_cam)
        """
        T_map_aruco, T_cam_aruco, T_base_cam = self.lookup_all_T(aruco_frame)
        T_map_base = T_map_aruco @ inv_T(T_cam_aruco) @ inv_T(T_base_cam)
        return T_map_base
    
    def get_amcl_base_pose(self):
        """
        AMCL(및 nav2)에서 현재 추정하는 map->base_footprint를 읽어서 비교용으로 사용
        (AMCL이 map->odom을 내고, odom->base가 이어지면 이 값이 "AMCL 추정"에 해당)
        """
        tf_map_base = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
        return tfmsg_to_T(tf_map_base)
    
    def publish_initialpose(self, x: float, y: float, yaw: float):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = float(qx)
        msg.pose.pose.orientation.y = float(qy)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)

        cov = [0.0] * 36
        cov[0] = float(self.var_xy)   # x
        cov[7] = float(self.var_xy)   # y
        cov[35] = float(self.var_yaw) # yaw
        msg.pose.covariance = cov

        # 1~3회 반복 발행 (AMCL이 확실히 받도록)
        for _ in range(3):
            self.initialpose_pub.publish(msg)

        self._last_publish_time = time.time()

    def on_timer(self):
        # 쿨다운 체크
        now = time.time()
        if now - self._last_publish_time < self.cooldown_sec:
            return

        # ✅ 보이는 마커 선택
        aruco_frame, z = self.select_visible_marker()
        if aruco_frame is None:
            self._consec_ok = 0
            return

        try:
            T_marker = self.compute_marker_based_base_pose(aruco_frame)
            # 비교용 AMCL 포즈
            T_amcl = self.get_amcl_base_pose()
        except TransformException as e:
            self._consec_ok = 0
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        mx, my, mz = T_marker[0, 3], T_marker[1, 3], T_marker[2, 3]
        myaw = yaw_from_R(T_marker[:3, :3])

        ax, ay, az = T_amcl[0, 3], T_amcl[1, 3], T_amcl[2, 3]
        ayaw = yaw_from_R(T_amcl[:3, :3])

        # jump sanity check (너무 튀면 안전상 보정 안 함)
        dxy = math.hypot(mx - ax, my - ay)
        dyaw = abs(wrap_pi(myaw - ayaw))

        ok_jump = (dxy <= self.max_jump_xy) and (math.degrees(dyaw) <= self.max_jump_yaw_deg)

        # 연속성 체크
        if ok_jump:
            self._consec_ok += 1
        else:
            self._consec_ok = 0

        # 간단 로그
        if self.get_clock().now().nanoseconds % 1_000_000_000 < 30_000_000:
            self.get_logger().info(
                f"marker_pose: x={mx:.3f} y={my:.3f} yaw={math.degrees(myaw):.1f}deg | "
                f"amcl_pose: x={ax:.3f} y={ay:.3f} yaw={math.degrees(ayaw):.1f}deg | "
                f"diff: dxy={dxy:.3f} dyaw={math.degrees(dyaw):.1f}deg | "
                f"consec={self._consec_ok}/{self.min_consecutive}"
            )

        if self._consec_ok >= self.min_consecutive:
            # 보정 실행
            self.get_logger().warn(
                f"Publishing /initialpose from marker-based pose: x={mx:.3f} y={my:.3f} yaw={math.degrees(myaw):.2f}deg "
                f"(dxy={dxy:.3f}, dyaw={math.degrees(dyaw):.1f}deg)"
            )
            self.publish_initialpose(mx, my, myaw)
            self._consec_ok = 0


def main():
    rclpy.init()
    node = MarkerAMCLCorrector()
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
# ros2 run actions marker_amcl_corrector --ros-args -p aruco_ids:="[600,601,602]"


