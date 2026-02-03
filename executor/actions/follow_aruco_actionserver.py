#!/usr/bin/env python3
"""
참고 로직 버전 (PrecisionController 스타일):

- SEARCH:
  마커가 보일 때까지 회전하며 탐색
  (마커 유실 타이머 포함)

- LOCK:
  포즈 샘플을 N개 수집한 뒤,
  중앙값(median)을 사용해 기준 포즈(lock_* 값) 확정

- PLAN:
  동작 단계(step) 리스트 생성
  (선택 사항: 좌우 x 보정 → 재 LOCK → z 방향 이동 → 최종 yaw 정렬)

- EXEC:
  ODOM 기준 제어로 각 step 실행
  (yaw 제어 / 전진 거리 제어)

- MJPEG 스트리밍:
  http://<ip>:5000/        (웹 페이지)
  http://<ip>:5000/stream (영상 스트림)

참고:
- 이 코드는 의도적으로 네 레퍼런스 코드와 "동일한 로직"을 사용함
  (샘플 수집 → 중앙값 LOCK → 계획 수립 → 실행 → 재 LOCK)
- 기본값에서는 좌우 보정(lateral correction)을 비활성화함
  (LATERAL_TOL_M = 0.0)
- 좌우 이동 기반 보정을 사용하려면
  LATERAL_TOL_M 값을 0보다 크게 설정하면 됨

주의:
- 기존 follow_aruco.py(PrecisionController)의 구조를 ActionServer로 감쌌다.
- Action goal 동안만 run loop를 수행하고 성공/실패를 반환한다.
"""

import math
import threading
import time
from typing import Optional, List, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from pinky_interfaces.action import FollowAruco  # ✅ 새 액션

from flask import Flask, Response, render_template_string
from libcamera import Transform
from picamera2 import Picamera2


# ---------------- utils ----------------
def yaw_from_quat(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def normalize_angle(rad: float) -> float:
    return math.atan2(math.sin(rad), math.cos(rad))


def clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else (lo if x < lo else x)


# ---------------- ArUco detector (same pattern) ----------------
class ArucoDetector:
    def __init__(self, dict_id=aruco.DICT_4X4_1000, marker_length_m=0.02):
        self.dict_id = dict_id
        self.marker_length_m = marker_length_m

    def get_dictionary(self):
        if hasattr(aruco, "Dictionary_get"):
            return aruco.Dictionary_get(self.dict_id)
        return aruco.getPredefinedDictionary(self.dict_id)

    def get_detector_parameters(self):
        if hasattr(aruco, "DetectorParameters_create"):
            params = aruco.DetectorParameters_create()
        else:
            params = aruco.DetectorParameters()

        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 31
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 5
        params.minMarkerPerimeterRate = 0.015
        params.maxMarkerPerimeterRate = 5.0
        params.polygonalApproxAccuracyRate = 0.025
        params.minCornerDistanceRate = 0.02
        params.minDistanceToBorder = 1
        params.cornerRefinementWinSize = 7
        params.cornerRefinementMaxIterations = 70
        params.cornerRefinementMinAccuracy = 0.005
        params.errorCorrectionRate = 0.8
        return params

    def estimate_pose_single_marker(self, corners, K, dist):
        if hasattr(aruco, "estimatePoseSingleMarkers"):
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length_m, K, dist
            )
            return rvecs, tvecs

        half = self.marker_length_m / 2.0
        objp = np.array(
            [
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )
        imgp = corners.reshape(-1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            objp, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None, None
        return np.array([rvec], dtype=np.float32), np.array([tvec], dtype=np.float32)

    def detect_pose(self, frame_rgb, aruco_dict, params, K, dist, target_id: Optional[int] = None):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        if hasattr(aruco, "detectMarkers"):
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)
        else:
            detector = aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return None

        if target_id is not None:
            flat = ids.flatten().astype(int).tolist()
            if target_id not in flat:
                return None
            idx = flat.index(target_id)
            use_corners = corners[idx]
            use_id = int(flat[idx])
        else:
            use_corners = corners[0]
            use_id = int(ids.flatten().astype(int)[0])

        rvecs, tvecs = self.estimate_pose_single_marker(use_corners, K, dist)
        if rvecs is None or tvecs is None:
            return None

        rvec = np.array(rvecs).reshape(-1)
        tvec = np.array(tvecs).reshape(-1)
        if rvec.size != 3 or tvec.size != 3:
            return None

        # yaw_to_x/yaw_to_z, cam_pos (same as reference)
        R, _ = cv2.Rodrigues(rvec.reshape(3))
        x_axis = R[:, 0]
        z_axis = R[:, 2]
        yaw_to_x = math.atan2(float(x_axis[0]), float(x_axis[2]))
        yaw_to_z = math.atan2(float(z_axis[0]), float(z_axis[2]))

        cam_pos = -R.T @ tvec.reshape(3, 1)
        x_m = float(cam_pos[0, 0])
        z_m = float(cam_pos[2, 0])

        return {
            "id": use_id,
            "corners": use_corners,
            "rvec": rvec.reshape(3),
            "tvec": tvec.reshape(3),
            "yaw_to_x": yaw_to_x,
            "yaw_to_z": yaw_to_z,
            "x_m": x_m,
            "z_m": z_m,
            "R": R,
        }


# ---------------- Flask stream (same idea) ----------------
class ArucoStreamServer:
    def __init__(self, host="0.0.0.0", port=5000):  
        self.host = host
        self.port = port
        self._frame_lock = threading.Lock()
        self._latest_bgr = None
        self._app = Flask(__name__)
        self._register_routes()

    def _register_routes(self):
        @self._app.route("/")
        def index():
            return render_template_string(
                """<!doctype html>
<html><head><meta charset="utf-8"><title>Aruco Stream</title></head>
<body style="background:#111;color:#eee;text-align:center;font-family:Arial">
<h2>Aruco Stream</h2>
<img src="/stream" style="max-width:95vw; border:2px solid #444; border-radius:10px"/>
</body></html>"""
            )

        @self._app.route("/stream")
        def stream():
            return Response(
                self._stream_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def set_latest_bgr(self, frame_bgr):
        with self._frame_lock:
            self._latest_bgr = frame_bgr

    def _get_latest_bgr(self):
        with self._frame_lock:
            if self._latest_bgr is None:
                return None
            return self._latest_bgr.copy()

    def _stream_frames(self):
        while True:
            frame = self._get_latest_bgr()
            if frame is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                time.sleep(0.05)
                continue
            jpg = buf.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            time.sleep(0.03)

    def start(self):
        th = threading.Thread(
            target=self._app.run,
            kwargs=dict(
                host=self.host,
                port=self.port,
                debug=False,
                threaded=True,
                use_reloader=False,
            ),
            daemon=True,
        )
        th.start()
        return th


# ---------------- ActionServer Node ----------------
class PrecisionController(Node):
    def __init__(self):
        super().__init__("follow_aruco_action_server")

        self.cb_group = ReentrantCallbackGroup()

        # Topics
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")

        # ✅ action name can be overridden
        self.declare_parameter("action_name", "pinky1/actions/follow_aruco")

        # ---- Params (Defaults) ----
        self.declare_parameter("marker_length_m", 0.02)
        self.declare_parameter("target_z_m", 0.1)
        self.declare_parameter("tol_z_m", 0.01)       # distance tolerance (m)
        self.declare_parameter("angle_tol_rad", 0.08)

        self.declare_parameter("lateral_tol_m", 0.03)
        self.declare_parameter("timeout_sec", 120.0)
        
        self.declare_parameter("samples_for_lock", 1)
        self.declare_parameter("search_yaw_speed", -0.20)
        self.declare_parameter("search_lost_sec", 3.0)

        self.declare_parameter("kp_yaw", 1.5)
        self.declare_parameter("max_yaw", 0.6)
        self.declare_parameter("turn_min_speed", 0.1)

        self.declare_parameter("kp_lin", 1.0)
        self.declare_parameter("max_lin", 0.10)
        self.declare_parameter("min_lin", 0.05)

        # Flask
        self.declare_parameter("flask_enable", True)
        self.declare_parameter("flask_host", "0.0.0.0")
        self.declare_parameter("flask_port", 5000)

        # ---- Read params ----
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        self.action_name = self.get_parameter("action_name").value

        self.marker_length_m = float(self.get_parameter("marker_length_m").value)
        self.TARGET_Z_M = float(self.get_parameter("target_z_m").value)
        self.TOL_Z_M = float(self.get_parameter("tol_z_m").value)
        self.ANGLE_TOL_RAD = float(self.get_parameter("angle_tol_rad").value)

        self.SAMPLES_FOR_LOCK = int(self.get_parameter("samples_for_lock").value)
        self.SEARCH_YAW_SPEED = float(self.get_parameter("search_yaw_speed").value)
        self.SEARCH_LOST_SEC = float(self.get_parameter("search_lost_sec").value)

        self.KP_YAW = float(self.get_parameter("kp_yaw").value)
        self.MAX_YAW = float(self.get_parameter("max_yaw").value)
        self.TURN_MIN_SPEED = float(self.get_parameter("turn_min_speed").value)

        self.KP_LIN = float(self.get_parameter("kp_lin").value)
        self.MAX_LIN = float(self.get_parameter("max_lin").value)
        self.MIN_LIN = float(self.get_parameter("min_lin").value)

        self.LATERAL_TOL_M = float(self.get_parameter("lateral_tol_m").value)
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)

        self.flask_enable = bool(self.get_parameter("flask_enable").value)
        self.flask_host = str(self.get_parameter("flask_host").value)
        self.flask_port = int(self.get_parameter("flask_port").value)

        # ---- Camera intrinsics (your fixed values) ----
        self.K = np.array(
            [
                [598.5252422042978, 0.0, 321.35841069961424],
                [0.0, 596.80057563617913, 249.25531392912907],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.dist = np.array(
            [
                0.16302535321241943,
                -0.50962206048835923,
                -0.0002584267984002369,
                0.0027091992261737566,
                0.56901681347085686,
            ],
            dtype=np.float64,
        )

        # ---- ROS I/O ----
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._on_odom,
            10,
            callback_group=self.cb_group,
        )

        # Action server
        self._as = ActionServer(
            self,
            FollowAruco,
            self.action_name,  # ✅ System1에서 호출하는 이름과 일치시켜라 (원하면 파라미터화 가능)
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self.cb_group,
        )

        # ---- detector + stream ----
        self.detector = ArucoDetector(marker_length_m=self.marker_length_m)
        self.aruco_dict = self.detector.get_dictionary()
        self.aruco_params = self.detector.get_detector_parameters()

        self.stream = ArucoStreamServer(self.flask_host, self.flask_port) if self.flask_enable else None
        if self.stream:
            self.stream.start()
            self.get_logger().info(f"[Flask] http://<raspi_ip>:{self.flask_port}/  (stream=/stream)")

        # ---- Picamera2 ----
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)},
            transform=Transform(vflip=1),
        )
        self.picam2.configure(config)
        self.picam2.start()

        # ---- Odom state ----
        self.current_yaw = None
        self.current_x = None
        self.current_y = None

        self.get_logger().info(
            "[FollowAruco] ActionServer ready. "
            f"cmd_vel={self.cmd_vel_topic} odom={self.odom_topic} "
            f"(defaults: target_z={self.TARGET_Z_M:.3f}, tol_z={self.TOL_Z_M:.3f}, "
            f"angle_tol={self.ANGLE_TOL_RAD:.3f}, lateral_tol={self.LATERAL_TOL_M:.3f}, "
            f"timeout={self.timeout_sec:.1f}s)"
        )

    # ---------------- ROS callbacks ----------------
    def _on_odom(self, msg: Odometry):
        self.current_yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def _goal_cb(self, goal_request: FollowAruco.Goal):
        # ✅ marker_id만 검사
        if int(goal_request.marker_id) <= 0:
            self.get_logger().warn("Reject FollowAruco goal: marker_id <= 0")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        self.get_logger().warn("FollowAruco cancel requested -> stop")
        self.cmd_pub.publish(Twist())
        return CancelResponse.ACCEPT


    # ---------------- helpers ----------------

    def _build_plan_from_lock(
        self,
        *,
        locked_err_z: float,
        locked_yaw_z: float,
        locked_dx_m: float,
        current_yaw: float,
        await_relock: bool,
    ) -> Tuple[List[Tuple[str, str, float]], bool]:
        """
        Returns: (plan, new_await_relock)
        Plan entries: (step_name, step_type, step_value)
          - ("move_z", "odom", meters)
          - ("align_z", "yaw", abs_yaw_goal)
          - ("turn_90", "yaw", abs_yaw_goal)
          - ("move_x", "odom", meters)
          - ("turn_back", "yaw", abs_yaw_goal)
          - ("relock", "relock", 0.0)
        """
        yaw_z_target = normalize_angle(locked_yaw_z + math.pi)

        # yaw step에서 쓸 “절대 목표 yaw” 만들기
        yaw_align_goal = normalize_angle(current_yaw + yaw_z_target)
        yaw_back = normalize_angle(current_yaw)

        # 이동량(거리) 계산: z와 x
        z_move = float(locked_err_z)  # err_z = (dz - target_z)

        corrected_dx_m = float(locked_dx_m)
        x_move = abs(corrected_dx_m)

        # plan 초기화
        plan = []

        # (중요) await_relock 분기: “옆으로 비켜간 후 두 번째 플랜”
        if await_relock:
            # after side-step, we only do z then yaw
            if abs(z_move) > self.TOL_Z_M:
                plan.append(("move_z", "odom", z_move))
            if abs(yaw_z_target) > self.ANGLE_TOL_RAD:
                plan.append(("align_z", "yaw", yaw_align_goal))
            return plan, False
 
        if self.LATERAL_TOL_M > 0.0 and x_move > self.LATERAL_TOL_M:
            side_sign = math.copysign(1.0, corrected_dx_m)  # dx<0 -> -90, dx>0 -> +90
            yaw_90 = normalize_angle(current_yaw + side_sign * (math.pi / 2.0))
            plan.append(("turn_90", "yaw", yaw_90))
            plan.append(("move_x", "odom", x_move))
            plan.append(("turn_back", "yaw", yaw_back))
            plan.append(("relock", "relock", 0.0))
            return plan, True

        if abs(z_move) > self.TOL_Z_M:
            plan.append(("move_z", "odom", z_move))
        if abs(yaw_z_target) > self.ANGLE_TOL_RAD:
            plan.append(("align_z", "yaw", yaw_align_goal))
        return plan, False

    def _overlay(self, frame_bgr, has_lock: bool, current_step, plan_len: int, det: Optional[dict]):
        cv2.putText(
            frame_bgr,
            f"lock={has_lock} step={(current_step[0] if current_step else '-')} plan={plan_len}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if det is None:
            return
        tvec = det["tvec"]
        cv2.putText(
            frame_bgr,
            f"ID={det['id']} dx={tvec[0]:+.3f} dy={tvec[1]:+.3f} dz={tvec[2]:.3f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # ---------------- Action execute ----------------
    async def _execute_cb(self, goal_handle):
        req = goal_handle.request

        # ✅ Design 1: goal에서 marker_id만 읽는다
        marker_id = int(req.marker_id)

        # ✅ goal에 timeout이 있으면 그걸 우선, 없으면 파라미터 timeout_sec 사용
        goal_timeout = float(getattr(req, "timeout_sec", 0.0))
        timeout_sec = goal_timeout if goal_timeout > 0.0 else float(self.timeout_sec)

        def stop_robot():
            self.cmd_pub.publish(Twist())


        # ---------------- [ADD] det 기반 완료 판정 함수 ----------------
        def dock_done_by_det(det: Optional[dict]) -> bool:
            """
            det 기반으로 '이미 충분히 도킹되었다'를 판정.
            조건:
            abs(err_z) <= TOL_Z_M  AND  abs(yaw_z_target) <= ANGLE_TOL_RAD
            여기서
            err_z = dz - TARGET_Z_M
            yaw_z_target = normalize_angle(yaw_to_z + pi)
            """
            if det is None:
                return False
            try:
                dz = float(det["tvec"][2])
                err_z = dz - float(self.TARGET_Z_M)

                yaw_to_z = float(det["yaw_to_z"])
                yaw_z_target = normalize_angle(yaw_to_z + math.pi)

                return (abs(err_z) <= float(self.TOL_Z_M)) and (abs(yaw_z_target) <= float(self.ANGLE_TOL_RAD))
            except Exception:
                return False

        # Goal-local state (CLEAN)
        has_lock = False
        await_relock = False
        last_seen_time = 0.0

        err_z_samples: List[float] = []
        dx_samples: List[float] = []
        yaw_z_samples: List[float] = []

        plan: List[Tuple[str, str, float]] = []
        current_step: Optional[Tuple[str, str, float]] = None
        step_pause_until = 0.0

        move_start_x: Optional[float] = None
        move_start_y: Optional[float] = None
        move_start_yaw: Optional[float] = None

        def reset_sampling():
            err_z_samples.clear()
            dx_samples.clear()
            yaw_z_samples.clear()

        t_start = time.time()

        self.get_logger().info(
            f"[FollowAruco] start marker_id={marker_id} timeout={timeout_sec:.1f}s "
            f"(defaults: target_z={self.TARGET_Z_M:.3f}, lateral_tol={self.LATERAL_TOL_M:.3f}, "
        )

        try:
            while rclpy.ok():
                # cancel?
                if goal_handle.is_cancel_requested:
                    stop_robot()
                    goal_handle.canceled()
                    res = FollowAruco.Result()
                    res.success = False
                    res.message = "canceled"
                    return res

                # timeout?
                if (time.time() - t_start) > self.timeout_sec:
                    stop_robot()
                    goal_handle.abort()
                    res = FollowAruco.Result()
                    res.success = False
                    res.message = "timeout"
                    return res

                # if odom not ready, stay still
                if self.current_yaw is None or self.current_x is None or self.current_y is None:
                    stop_robot()
                    time.sleep(0.01)
                    continue

                # 3) 카메라 프레임 획득 & ArUco 검출
                frame_rgb = self.picam2.capture_array()
                det = self.detector.detect_pose(
                    frame_rgb,
                    self.aruco_dict,
                    self.aruco_params,
                    self.K,
                    self.dist,
                    target_id = marker_id,
                )

                # Build BGR for stream/overlay
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # 마커가 보이고, 이미 거리/각도 조건이 만족되면 어떤 상태든 즉시 종료
                if dock_done_by_det(det):
                    stop_robot()
                    self.get_logger().info("[DONE] docking_done_by_det: |err_z|<=TOL_Z and |yaw_z_target|<=ANGLE_TOL")
                    goal_handle.succeed()
                    res = FollowAruco.Result()
                    res.success = True
                    res.message = "docking_done_by_det"
                    return res

                # ---------------- SEARCH / LOCK sampling ----------------
                # 4) “아직 LOCK이 없으면” SEARCH/샘플링 모드
                if not has_lock:
                    # 5) 마커가 안 보일 때(det is None) → SEARCH 회전
                    if det is None:
                        # 5-1) 화면 오버레이/스트림 업데이트
                        self._overlay(frame_bgr, has_lock, current_step, len(plan), None)
                        if self.stream:
                            self.stream.set_latest_bgr(frame_bgr)

                        msg = Twist()
                        # 5-2) 샘플링 도중에 갑자기 잃어버렸으면 잠깐 정지
                        if len(err_z_samples) > 0:
                            self.cmd_pub.publish(msg)
                            time.sleep(0.01)
                            continue

                        # 5-3) 너무 오래 못 봤으면 회전 시작
                        if (time.time() - last_seen_time) > self.SEARCH_LOST_SEC:
                            msg.angular.z = self.SEARCH_YAW_SPEED
                        self.cmd_pub.publish(msg)
                        time.sleep(0.01)
                        continue

                    # 6) 마커가 보일 때(det != None) → LOCK용 샘플 축적
                    # 마커가 보이면 이 블록으로 들어옴.
                    # 6-1) last_seen_time 갱신
                    last_seen_time = time.time()

                    # 6-2) 검출 결과를 꺼내고 화면에 그리기
                    corners = det["corners"]
                    rvec = det["rvec"]
                    tvec = det["tvec"]

                    # draw markers/axes like reference
                    try:
                        aruco.drawDetectedMarkers(frame_bgr, [corners])
                        cv2.drawFrameAxes(frame_bgr, self.K, self.dist, rvec.reshape(1, 3), tvec.reshape(1, 3), 0.05)
                    except Exception:
                        pass

                    # 6-3) 핵심 측정값 계산 & 샘플 리스트에 추가
                    dx, dy, dz = float(tvec[0]), float(tvec[1]), float(tvec[2])
                    err_z = (dz - self.TARGET_Z_M)

                    err_z_samples.append(err_z)
                    dx_samples.append(dx)
                    yaw_z_samples.append(float(det["yaw_to_z"]))

                    # 6-4) 샘플 개수 표시 + 스트림 업데이트
                    cv2.putText(
                        frame_bgr,
                        f"samples={len(err_z_samples)}/{self.SAMPLES_FOR_LOCK}",
                        (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    self._overlay(frame_bgr, has_lock, current_step, len(plan), det)
                    if self.stream:
                        self.stream.set_latest_bgr(frame_bgr)

                    # 6-5) 샘플이 충분하면 LOCK(중앙값) 확정 + 플랜 생성
                    if len(err_z_samples) >= self.SAMPLES_FOR_LOCK:
                        locked_err_z = float(np.median(err_z_samples))
                        locked_yaw_z = float(np.median(yaw_z_samples))
                        locked_dx_m = float(np.median(dx_samples))

                        plan, await_relock = self._build_plan_from_lock(
                            locked_err_z=locked_err_z,
                            locked_yaw_z=locked_yaw_z,
                            locked_dx_m=locked_dx_m,
                            current_yaw=float(self.current_yaw),
                            await_relock=await_relock,
                        )

                        current_step = None
                        step_pause_until = 0.0
                        move_start_x = None
                        move_start_y = None
                        move_start_yaw = None

                        has_lock = True
                        self.get_logger().info(
                            f"[LOCK] err_z={locked_err_z:+.3f} dx={locked_dx_m:+.3f} "
                            f"yaw_z={math.degrees(locked_yaw_z):+.1f}deg plan={len(plan)} "
                            f"(await_relock={await_relock})"
                        )

                    # 6-6) 샘플링 중에는 로봇은 항상 정지
                    stop_robot()
                    time.sleep(0.01)
                    continue

                # ---------------- EXEC plan ----------------
                # 7) 이제 LOCK이 있으니 플랜 실행 모드
                # 7-1) 오버레이/스트림은 계속 갱신
                self._overlay(frame_bgr, has_lock, current_step, len(plan), det)
                if self.stream:
                    self.stream.set_latest_bgr(frame_bgr)

                # ---------------- [ADD] EXEC 구간에서도 한번 더 안전하게 체크 ----------------
                # (특히 align_z 직후, plan이 비기 전에 조건 만족해도 종료되게)
                if dock_done_by_det(det):
                    stop_robot()
                    self.get_logger().info("[DONE] docking_done_by_det (exec stage)")
                    goal_handle.succeed()
                    res = FollowAruco.Result()
                    res.success = True
                    res.message = "docking_done_by_det"
                    return res

                # 8) Step 스케줄링: current_step이 없으면 plan에서 하나 꺼내기
                if current_step is None:
                    # 8-1) step 끝나고 “휴지 시간”이면 정지 유지
                    if time.time() < step_pause_until:
                        stop_robot()
                        time.sleep(0.01)
                        continue

                    # ✅ plan이 비었으면 "성공"이 아니라, 다시 샘플링으로 돌리는 게 더 안전
                    # (MoveToPID는 reached_flag가 명확함. 여기서는 det가 없으면 확정 불가)
                    if not plan:
                        self.get_logger().info("[FollowAruco] plan empty -> relock (no det-done yet)")
                        has_lock = False
                        await_relock = False
                        reset_sampling()
                        stop_robot()
                        time.sleep(0.01)
                        continue
                    
                    current_step = plan.pop(0)
                    
                    step_name, step_type, step_value = current_step
                    self.get_logger().info(f"[STEP] start {step_name} type={step_type} value={step_value}")

                # 9) Step 실행: step_type 별로 cmd_vel 생성
                step_name, step_type, step_value = current_step
                msg = Twist()   

                # 9-A) step_type == "yaw" : 목표 yaw로 회전
                if step_type == "yaw":
                    # 오차 계산
                    err = normalize_angle(float(step_value) - float(self.current_yaw))
                    # 오차가 ANGLE_TOL_RAD 이내면 step 완료:
                    if abs(err) <= self.ANGLE_TOL_RAD:
                        self.get_logger().info(f"[STEP] done {step_name} yaw_err={math.degrees(err):.2f}deg")
                        current_step = None
                        move_start_x = None
                        move_start_y = None
                        move_start_yaw = None
                        step_pause_until = time.time() + 1.0
                    else:
                        speed = clamp(self.KP_YAW * err, -self.MAX_YAW, self.MAX_YAW)
                        if abs(speed) < self.TURN_MIN_SPEED:
                            speed = math.copysign(self.TURN_MIN_SPEED, err)
                        msg.angular.z = float(speed)

                # 9-B) step_type == "odom" : odom으로 “전진거리” 맞추기
                elif step_type == "odom":
                    # step 시작 시점이면 start pose 저장
                    if move_start_x is None:
                        move_start_x = float(self.current_x)
                        move_start_y = float(self.current_y)
                        move_start_yaw = float(self.current_yaw)

                    # traveled 계산: “시작 yaw 방향으로 얼마나 전진했나”
                    dxw = float(self.current_x) - float(move_start_x)
                    dyw = float(self.current_y) - float(move_start_y)
                    forward_x = math.cos(float(move_start_yaw))
                    forward_y = math.sin(float(move_start_yaw))
                    traveled = dxw * forward_x + dyw * forward_y

                    err = float(step_value) - float(traveled)
                    # err가 TOL_Z_M 이내면 step 완료
                    if abs(err) <= self.TOL_Z_M:
                        self.get_logger().info(f"[STEP] done {step_name} dist_err={err:+.3f}m")
                        current_step = None
                        move_start_x = move_start_y = move_start_yaw = None
                        step_pause_until = time.time() + 1.0
                    # 아니면 P제어로 선속도 
                    else:
                        speed = clamp(self.KP_LIN * err, -self.MAX_LIN, self.MAX_LIN)
                        if abs(speed) < self.MIN_LIN:
                            speed = math.copysign(self.MIN_LIN, err)
                        msg.linear.x = float(speed)

                # 9-C) step_type == "relock" : 다시 LOCK부터 하도록 상태 리셋
                elif step_type == "relock":
                    self.get_logger().info(f"[STEP] done {step_name} -> relock sampling")
                    current_step = None
                    move_start_x = move_start_y = move_start_yaw = None
                    step_pause_until = time.time() + 1.0
                    has_lock = False
                    reset_sampling()

                else:
                    self.get_logger().warn(f"Unknown step_type={step_type} -> abort")
                    stop_robot()
                    goal_handle.abort()
                    res = FollowAruco.Result()
                    res.success = False
                    res.message = f"unknown_step_type:{step_type}"
                    return res

                self.cmd_pub.publish(msg)
                time.sleep(0.01)

            # rclpy not ok -> abort
            stop_robot()
            goal_handle.abort()
            res = FollowAruco.Result()
            res.success = False
            res.message = "rclpy_not_ok"
            return res

        except Exception as e:
            stop_robot()
            self.get_logger().error(f"[FollowAruco] exception: {e}")
            goal_handle.abort()
            res = FollowAruco.Result()
            res.success = False
            res.message = f"exception:{e}"
            return res


def main():
    rclpy.init()
    node = PrecisionController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()   # ✅ ActionServer가 요청 받을 때까지 계속 살아있음
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


### 실행 예시:
# ros2 run actions follow_aruco_action 

### 상위 제어에서 해야할 일을 간단 버전으로 테스트 할 수 있는 실행문
# ros2 action send_goal \
#   /pinky1/actions/follow_aruco \
#   pinky_interfaces/action/FollowAruco \
#   "{marker_id: 600}" \
#   --feedback

### flask 스트리밍 서버 
# http://127.0.0.1:5000 

