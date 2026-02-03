#!/usr/bin/env python3
import math
import time
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Header
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2

import tf2_ros
from tf2_ros import TransformException



def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _norm_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

class LidarObstacleDetector(Node):
    """
    - 전방 섹터(±front_half_angle) 안에서
    - enter_dist 이내의 포인트만 후보로 수집
    - map에서 occupied인 점은 정적으로 보고 제외 (TF/Map 준비되면)
    - 남은 동적 후보의 최소거리로 obstacle True/False 결정
    - obstacle=True일 때만 /obstacle_points 퍼블리시(동적 후보만)
    """

    def __init__(self):
        super().__init__("lidar_obstacle_detector")

        # Topics
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("obstacle_topic", "/obstacle_detected")

        # point cloud
        self.declare_parameter("cloud_topic", "/obstacle_points")
        self.declare_parameter("cloud_frame", "rplidar_link")   # 라이다 프레임 추천(없으면 base_link)
        self.declare_parameter("cloud_z", 0.0)             # 포인트 클라우드 생성시 z 고정값

        # front sector 설정
        self.declare_parameter("front_half_angle_deg", 15.0)  # degrees

        # ✅ front 방향 보정 오프셋 (라이다 0rad가 뒤를 볼 때: pi)
        self.declare_parameter("front_yaw_offset_rad", 3.141592)

        # Distance thresholds (m)
        self.declare_parameter("enter_dist_m", 0.30) # 장애물 감지 진입 거리
        self.declare_parameter("exit_dist_m", 0.40)  # 장애물 감지 해제 거리

        # Hold time (sec): obstacle이 한번 켜지면 최소 이 시간 동안 유지
        self.declare_parameter("hold_time_sec", 0.4)

        # Minimum valid range
        # 라이다 range 값 중 “너무 가까운 값”을 장애물 후보에서 아예 제외하기 위한 필터 파라미터
        self.declare_parameter("min_valid_range_m", 0.25)

        # ✅ map 기반 정적 장애물 제외 필터 파라미터
        self.declare_parameter("use_map_filter", True)
        self.declare_parameter("map_topic", "/local_costmap/costmap")    # "/local_costmap/costmap" or "/map"
        self.declare_parameter("map_frame", "odom")   # odom or map

        # occupancy threshold
        # nav_msgs/OccupancyGrid data: [-1 unknown, 0 free, 100 occupied] 가 일반적
        self.declare_parameter("occ_threshold", 40)  # >=50 이면 occupied로 간주

        # unknown(-1) 처리 정책
        # map 값이 -1(unknown)일 때 occupied로 볼지(제거) / free로 볼지(유지) 정책
        self.declare_parameter("treat_unknown_as_occupied", False)

        # TF lookup timeout
        self.declare_parameter("tf_timeout_sec", 0.05)

        # Debug
        self.declare_parameter("debug_log", True)
        self.debug_log = bool(self.get_parameter("debug_log").value)

        # debug - _sector_indices, range 관련
        self._last_front_log_t = 0.0
        self.declare_parameter("debug_front_log_sec", 1.0)  # 1초에 1번
        self.debug_front_log_sec = float(self.get_parameter("debug_front_log_sec").value)

        # debug - map origin x,y 관련
        self._map: Optional[OccupancyGrid] = None
        self._map_meta_logged = False
        self._last_map_stamp = None


        # load params
        self.scan_topic = self.get_parameter("scan_topic").value
        self.obstacle_topic = self.get_parameter("obstacle_topic").value

        self.cloud_topic = self.get_parameter("cloud_topic").value
        self.cloud_frame = self.get_parameter("cloud_frame").value
        self.cloud_z = float(self.get_parameter("cloud_z").value)

        self.front_half_angle = math.radians(float(self.get_parameter("front_half_angle_deg").value))
        self.front_yaw_offset = float(self.get_parameter("front_yaw_offset_rad").value)
        self.enter_dist = float(self.get_parameter("enter_dist_m").value)
        self.exit_dist = float(self.get_parameter("exit_dist_m").value)
        self.hold_time = float(self.get_parameter("hold_time_sec").value)
        self.min_valid_range = float(self.get_parameter("min_valid_range_m").value)

        self.use_map_filter = bool(self.get_parameter("use_map_filter").value)
        self.map_topic = self.get_parameter("map_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.occ_threshold = int(self.get_parameter("occ_threshold").value)
        self.treat_unknown_as_occupied = bool(self.get_parameter("treat_unknown_as_occupied").value)
        self.tf_timeout = float(self.get_parameter("tf_timeout_sec").value)

        # publishers / subscribers
        self.pub = self.create_publisher(Bool, self.obstacle_topic, 10)
        self.cloud_pub = self.create_publisher(PointCloud2, self.cloud_topic, 10)
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self._on_scan, 10)

        # map subscriber (optional)
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # map cache
        self._map: Optional[OccupancyGrid] = None

        # state
        self._obstacle = False
        self._last_true_time: Optional[float] = None

        self.get_logger().info(
            f"[LidarObstacleDetector] scan={self.scan_topic} pub={self.obstacle_topic} "
            f"sector=±{math.degrees(self.front_half_angle):.1f}deg "
            f"front_yaw_offset={self.front_yaw_offset:.6f}rad "
            f"sector=±{math.degrees(self.front_half_angle):.1f}deg "
            f"enter={self.enter_dist:.2f}m exit={self.exit_dist:.2f}m hold={self.hold_time:.2f}s "
            f"min_valid={self.min_valid_range:.2f}m "
            f"use_map_filter={self.use_map_filter} map={self.map_topic} map_frame={self.map_frame} "
            f"occ_thr={self.occ_threshold} unknown_as_occ={self.treat_unknown_as_occupied}"
        )

    # map callback
    def _on_map(self, msg: OccupancyGrid):
        self._map = msg

        # 1) 최초 1회만 출력
        if not self._map_meta_logged:
            info = msg.info
            o = info.origin.position
            self.get_logger().info(
                f"[map meta] res={info.resolution:.3f} "
                f"origin=({o.x:.3f},{o.y:.3f}) w={info.width} h={info.height}"
            )
            self._map_meta_logged = True

    # ------------------------
    # publish helpers
    def _publish(self, val: bool):
        msg = Bool()
        msg.data = val
        self.pub.publish(msg)

    def _publish_obstacle_cloud(self, points_xyz):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.cloud_frame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        cloud = pc2.create_cloud(header, fields, points_xyz)
        self.cloud_pub.publish(cloud) 

    # ------------------------
    # scan processing
    def _sector_indices(self, scan: LaserScan) -> Optional[Tuple[int, int]]:
        a_min = scan.angle_min
        a_inc = scan.angle_increment
        if a_inc == 0.0 or len(scan.ranges) == 0:
            return None

        # clamp to scan min/max
        ang0 = _clamp(-self.front_half_angle, scan.angle_min, scan.angle_max)
        ang1 = _clamp(+self.front_half_angle, scan.angle_min, scan.angle_max)

        i0 = int((ang0 - a_min) / a_inc)
        i1 = int((ang1 - a_min) / a_inc) 

        i0 = max(0, min(len(scan.ranges) - 1, i0))
        i1 = max(0, min(len(scan.ranges) - 1, i1))
        if i1 < i0:
            i0, i1 = i1, i0

        return (i0, i1)

    def _min_range_in_front_sector(self, scan: LaserScan) -> Optional[float]:
        a_min = scan.angle_min
        a_inc = scan.angle_increment
        if a_inc == 0.0 or len(scan.ranges) == 0:
            return None

        best = None
        for i, r in enumerate(scan.ranges):
            if r is None or math.isinf(r) or math.isnan(r):
                continue
            if r < self.min_valid_range:
                continue
            if r < scan.range_min or r > scan.range_max:
                continue

            ang = _norm_angle(a_min + i * a_inc + self.front_yaw_offset)

            # ✅ "보정된 각도" 기준으로 전방(±half_angle) 판정
            if abs(ang) > self.front_half_angle:
                continue

            if best is None or r < best:
                best = r

        return best
    
    def _collect_obstacle_points_xy(self, scan: LaserScan, r_max: float) -> List[Tuple[float, float, float]]:
        a_min = scan.angle_min
        a_inc = scan.angle_increment
        if a_inc == 0.0 or len(scan.ranges) == 0:
            return []

        rmin = scan.range_min
        rmax = min(float(r_max), float(scan.range_max))

        # ✅ 디버그(전방 각도 범위 확인)
        now = time.time()
        if (now - self._last_front_log_t) >= self.debug_front_log_sec:
            self._last_front_log_t = now
            # 전방 경계각(보정 후)
            left = _norm_angle(-self.front_half_angle)   # -half
            right = _norm_angle(+self.front_half_angle)  # +half
            self.get_logger().info(
                f"[front] offset={self.front_yaw_offset:.4f} rad, half={self.front_half_angle:.4f} rad, "
                f"rmin={rmin:.3f} rmax={rmax:.3f} n={len(scan.ranges)}"
            )

        pts: List[Tuple[float, float, float]] = []
        for i, r in enumerate(scan.ranges):
            if r is None or math.isinf(r) or math.isnan(r):
                continue
            if r < self.min_valid_range:
                continue
            if r < rmin or r > rmax:
                continue

            ang = _norm_angle(a_min + i * a_inc + self.front_yaw_offset)

            # ✅ 전방 판정도 보정된 각도로
            if abs(ang) > self.front_half_angle:
                continue

            # ✅ enter_dist 이내만 “장애물 후보 포인트”로 채택
            if r <= self.enter_dist:
                x = float(r * math.cos(ang))
                y = float(r * math.sin(ang))
                pts.append((x, y, float(self.cloud_z)))

        return pts
    
    # ------------------------

    def _world_to_map_index(self, map_msg: OccupancyGrid, wx: float, wy: float) -> Optional[int]:
        """
        map 좌표계(wx,wy)를 OccupancyGrid index로 변환.
        return: data[] 인덱스 또는 None(범위 밖)
        """
        info = map_msg.info
        origin = info.origin.position
        res = info.resolution
        if res <= 0.0:
            return None
        
        mx = int((wx - origin.x) / res)
        my = int((wy - origin.y) / res)

        if mx < 0 or my < 0 or mx >=info.width or my >= info.height:
            return None
        return my * info.width +mx    # OccupancyGrid.data[]의 인덱스

    # “라이다로 잡힌 점(월드좌표 wx, wy)이
    #  지도(/map)에서 원래부터 있는 벽/고정물(정적 장애물) 위에 찍힌 점인가?”를 판정하는 함수
    def _is_static_occupied_in_map(self, wx: float, wy: float) -> bool:
        """
        use_map_filter가 꺼져 있으면 → 지도 필터 자체를 안 쓰니까 False
        아직 map을 못 받았으면(self._map is None) → 판단 불가니까 False
        (wx, wy)가 map의 몇 번째 칸인지 idx로 바꿈
        그 칸의 점유값 occ = map_msg.data[idx]를 읽음
        occ가 -1(unknown)이면 → 옵션(treat_unknown_as_occupied)에 따라 True/False
        그 외(0~100)이면 → occ_threshold 이상이면 occupied로 보고 True
        """
        if not self.use_map_filter:
            return False
        
        if self._map is None:
            return False
        
        idx = self._world_to_map_index(self._map, wx, wy)
        if idx is None:
            return False    # map 밖이면 정적이라고 보기 어려우니 keep

        occ = int(self._map.data[idx])    # -1 unknown, 0 free, 100 occupied (일반)
        if occ < 0:
            return bool(self.treat_unknown_as_occupied)
        
        return occ >= self.occ_threshold

    def _transform_points_to_map_and_filter(
        self, points_cloud: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """
        cloud_frame 기준 포인트들을 map으로 변환해서,
        map에서 occupied면 "정적"으로 보고 제거.
        반환은 여전히 cloud_frame 기준 포인트 리스트(동적 후보)로 유지.
        """
        if not self.use_map_filter or self._map is None or len(points_cloud) == 0:
            return points_cloud    # "장애물 후보 포인트" 그대로 반환
        
        # cloud_frame -> map 변환 TF 얻기
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.cloud_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout),
            )
        except TransformException as e:
            # TF 실패면 필터 못하니 원본 유지
            self.get_logger().warn(f"[map_filter] TF lookup failed: {e}")
            return points_cloud
        
        # transform 구성요소
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        tz = tf.transform.translation.z
        qx = tf.transform.rotation.x
        qy = tf.transform.rotation.y
        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w

        # quaternion -> yaw(2D) 로 단순화 (라이다 2D 필터 목적)
        # (map 필터용이라 roll/pitch 무시)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        filtered: List[Tuple[float, float, float]] = []
        for (x, y, z) in points_cloud:
            wx = (cy * x - sy * y) + tx
            wy = (sy * x + cy * y) + ty

            # map에서 occupied면 정적 => 제거
            if self._is_static_occupied_in_map(wx, wy):
                continue

            filtered.append((x, y, z))
        return filtered
    
    def _min_range_from_points(self, points_xyz: List[Tuple[float, float, float]]) -> Optional[float]:
        """
        points_xyz (cloud_frame)에서 sqrt(x^2+y^2)의 최소를 반환.
        """
        if not points_xyz:
            return None
        best = None
        for (x, y, z) in points_xyz:
            r = math.hypot(x, y)
            if best is None or r < best:
                best = r
        return best
    
    def _on_scan(self, scan: LaserScan):
        now = time.time()

        # 0) 전방 섹터에서 "항상" 최소거리 계산 (enter_dist 이하로 제한하지 않음)
        min_r = self._min_range_in_front_sector(scan)  # <-- 아래에 새로 추가할 함수

        # 1) 기존 방식대로 "장애물 후보 포인트"를 수집 
        # r_max 는 포인트를 수집할 최대 거리
        pts = self._collect_obstacle_points_xy(scan, r_max=self.enter_dist)

        # 2) ✅ map 기반 정적 제외 필터 적용 (pts_dynamic만 남김)
        pts_dynamic = self._transform_points_to_map_and_filter(pts)

        # 3) 판정: min_r가 없으면(유효 빔이 없음) -> False로 떨어뜨리는 게 안전
        #    (현재 테스트는 "이동 중 장애물 감지"라서, 불확실하면 멈추는 방향도 가능하지만
        #     지금 문제는 False->True 래치가 더 큰 문제라서 False가 안정적)
        if min_r is None:
            self._obstacle = False
            self._last_true_time = None

            self._publish(self._obstacle)
            self._publish_obstacle_cloud([])

            if self.debug_log:
                self.get_logger().info("[dbg] obstacle=False min_r=None pts=0 pts_after_map=0")
            return

        # 4) hysteresis + hold_time
        if not self._obstacle:
            if min_r <= self.enter_dist:
                self._obstacle = True
                self._last_true_time = now
        else:
            # hold_time 동안은 무조건 유지
            if self._last_true_time is not None and (now - self._last_true_time) < self.hold_time:
                pass
            else:
                # exit_dist 이상이면 해제
                if min_r >= self.exit_dist:
                    self._obstacle = False
                    self._last_true_time = None

        # 5) publish
        self._publish(self._obstacle)
        self._publish_obstacle_cloud(pts_dynamic if self._obstacle else [])

        if self.debug_log:
            self.get_logger().info(
                f"[dbg] obstacle={self._obstacle} min_r={min_r:.3f} "
                f"pts={len(pts)} pts_after_map={len(pts_dynamic)}"
            )

        

def main():
    rclpy.init()
    node = LidarObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


### 실행예시:
# ros2 run pinky_camera lidar_publisher --ros-args   -p front_yaw_offset_rad:=3.141592   -p front_half_angle_deg:=12.0   -p min_valid_range_m:=0.25   -p enter_dist_m:=0.3   -p exit_dist_m:=0.4



