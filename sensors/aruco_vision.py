import math
import threading
import time
from typing import Optional

import cv2
import cv2.aruco as aruco
import numpy as np
 
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Bool

from flask import Flask, Response, render_template_string

import tf2_ros
from tf2_ros import TransformBroadcaster

from libcamera import Transform
from picamera2 import Picamera2


# ---------------- utils (your style) ----------------
def quat_from_rvec(rvec: np.ndarray):
    """
    Convert OpenCV Rodrigues rvec to quaternion (x,y,z,w).
    OpenCV rvec is rotation of marker w.r.t camera (R_cam_aruco).
    """
    R, _ = cv2.Rodrigues(rvec.reshape(3))
    # rotation matrix to quaternion
    # robust standard conversion:
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


# ---------------- ArUco detector (YOUR CODE, minimal touch) ----------------
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
    
# ---------------- Flask stream (YOUR CODE, minimal touch) ----------------
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
                """<!doctype html><html><head><meta charset="utf-8">
                <title>Aruco Stream</title></head>
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
    
# ---------------- Vision Node ----------------
class ArucoVisionNode(Node):
    def __init__(self):
        super().__init__("aruco_vision_node")

        # ---- Params ----
        # self.declare_parameter("marker_id", 600)  # target id (single)
        self.declare_parameter("watch_marker_ids", [])
        param = self.get_parameter_or("watch_marker_ids", Parameter("watch_marker_ids", value=[]))
        raw = param.value
        # value가 tuple/list/단일값/문자열로 올 수도 있어서 방어
        if raw is None:
            self.watch_marker_ids = []
        elif isinstance(raw, (list, tuple)):
            self.watch_marker_ids = [int(x) for x in raw]
        else:
            # 혹시 "[]" 같은 문자열로 들어오는 케이스 방어
            s = str(raw).strip()
            if s in ("", "[]"):
                self.watch_marker_ids = []
            else:
                # "1,2,3" 형태도 방어
                s = s.strip("[]")
                self.watch_marker_ids = [int(x.strip()) for x in s.split(",") if x.strip()]

        self.watch_set = set(self.watch_marker_ids)


        self.declare_parameter("marker_length_m", 0.02)
        self.declare_parameter("dict_id", int(aruco.DICT_4X4_1000))

        self.declare_parameter("camera_frame", "front_camera_link")
        self.declare_parameter("aruco_frame_prefix", "aruco_")  # -> aruco_600

        self.declare_parameter("publish_tf", True)
        self.declare_parameter("pose_topic", "/aruco_pose")
        self.declare_parameter("detected_topic", "/aruco_detected")

        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("lost_timeout_sec", 0.5)  # if no detection, stop publishing TF/pose after this

        # Flask
        self.declare_parameter("flask_enable", True)
        self.declare_parameter("flask_host", "0.0.0.0")
        self.declare_parameter("flask_port", 5000)

        # Picamera2
        self.declare_parameter("image_w", 640)
        self.declare_parameter("image_h", 480)
        self.declare_parameter("vflip", True)

        # Camera intrinsics (your fixed values default)
        self.declare_parameter("K", [
            598.5252422042978, 0.0, 321.35841069961424,
            0.0, 596.80057563617913, 249.25531392912907,
            0.0, 0.0, 1.0
        ])
        self.declare_parameter("dist", [
            0.16302535321241943,
            -0.50962206048835923,
            -0.0002584267984002369,
            0.0027091992261737566,
            0.56901681347085686,
        ])

        # ---- Read ----
        # self.marker_id = int(self.get_parameter("marker_id").value)
        self.marker_length_m = float(self.get_parameter("marker_length_m").value)
        self.dict_id = int(self.get_parameter("dict_id").value)

        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.aruco_prefix = str(self.get_parameter("aruco_frame_prefix").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.detected_topic = str(self.get_parameter("detected_topic").value)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.lost_timeout_sec = float(self.get_parameter("lost_timeout_sec").value)

        self.flask_enable = bool(self.get_parameter("flask_enable").value)
        self.flask_host = str(self.get_parameter("flask_host").value)
        self.flask_port = int(self.get_parameter("flask_port").value)

        self.image_w = int(self.get_parameter("image_w").value)
        self.image_h = int(self.get_parameter("image_h").value)
        self.vflip = bool(self.get_parameter("vflip").value)

        K_list = list(self.get_parameter("K").value)
        dist_list = list(self.get_parameter("dist").value)
        self.K = np.array(K_list, dtype=np.float64).reshape(3, 3)
        self.dist = np.array(dist_list, dtype=np.float64)

        # ---- Detector ----
        self.detector = ArucoDetector(dict_id=self.dict_id, marker_length_m=self.marker_length_m)
        self.aruco_dict = self.detector.get_dictionary()
        self.aruco_params = self.detector.get_detector_parameters()

        # ---- TF publisher ----
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_seen_time = 0.0

        # ---- Topics ----
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.detected_pub = self.create_publisher(Bool, self.detected_topic, 10)

        # ---- Flask stream ----
        self.stream = ArucoStreamServer(self.flask_host, self.flask_port) if self.flask_enable else None
        if self.stream:
            self.stream.start()
            self.get_logger().info(f"[Flask] http://<raspi_ip>:{self.flask_port}/  (stream=/stream)")

        # ---- Picamera2 ----
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (self.image_w, self.image_h)},
            transform=Transform(vflip=1 if self.vflip else 0),
        )
        self.picam2.configure(config)
        self.picam2.start()

        # ---- Timer ----
        dt = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(dt, self._tick)

        self.get_logger().info(
            f"[ArucoVisionNode] watch_marker_ids={self.watch_marker_ids}, camera_frame={self.camera_frame}, "
        )

    def _publish_detected(self, ok: bool):
        msg = Bool()
        msg.data = bool(ok)
        self.detected_pub.publish(msg)

    def _publish_pose_and_tf(self, det: dict):
        now = self.get_clock().now().to_msg()
        aruco_frame = f"{self.aruco_prefix}{int(det['id'])}"

        # tvec: marker position in camera frame (meters)
        tvec = det["tvec"]
        rvec = det["rvec"]

        qx, qy, qz, qw = quat_from_rvec(np.array(rvec, dtype=np.float64))

        # PoseStamped
        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = self.camera_frame
        ps.pose.position.x = float(tvec[0])
        ps.pose.position.y = float(tvec[1])
        ps.pose.position.z = float(tvec[2])
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pose_pub.publish(ps)

        # TF
        if self.publish_tf:
            ts = TransformStamped()
            ts.header.stamp = now
            ts.header.frame_id = self.camera_frame
            ts.child_frame_id = aruco_frame
            ts.transform.translation.x = float(tvec[0])
            ts.transform.translation.y = float(tvec[1])
            ts.transform.translation.z = float(tvec[2])
            ts.transform.rotation.x = qx
            ts.transform.rotation.y = qy
            ts.transform.rotation.z = qz
            ts.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(ts)

    def _overlay(self, frame_bgr, det: Optional[dict]):
        cv2.putText(
            frame_bgr,
            f"Vision TF parent={self.camera_frame}  allow={self.watch_marker_ids if self.watch_set else 'ANY'}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if det is None:
            cv2.putText(
                frame_bgr,
                "det: NONE",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return

        tvec = det["tvec"]
        cv2.putText(
            frame_bgr,
            f"ID={det['id']}  tvec=[{tvec[0]:+.3f},{tvec[1]:+.3f},{tvec[2]:.3f}]",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        try:
            corners = det["corners"]
            rvec = det["rvec"]
            tvec = det["tvec"]
            aruco.drawDetectedMarkers(frame_bgr, [corners])
            cv2.drawFrameAxes(frame_bgr, self.K, self.dist, rvec.reshape(1, 3), tvec.reshape(1, 3), 0.05)
        except Exception:
            pass


    def _tick(self):
        frame_rgb = self.picam2.capture_array()
        det = None
        if self.watch_set:
            # watch_set 중 보이는 것 하나를 고르기(우선순위: 리스트 순서)
            for mid in self.watch_marker_ids:
                det = self.detector.detect_pose(
                    frame_rgb,
                    self.aruco_dict,
                    self.aruco_params,
                    self.K,
                    self.dist,
                    target_id=int(mid),
                )
                if det is not None:
                    break
        else:
            # 아무거나 허용: 첫 번째 검출 마커 사용
            det = self.detector.detect_pose(
                frame_rgb,
                self.aruco_dict,
                self.aruco_params,
                self.K,
                self.dist,
                target_id=None,
            )

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if det is None:
            self._publish_detected(False)

            # Optional: if you want to "stop publishing" after timeout, do nothing.
            # (We already do nothing here.)
            self._overlay(frame_bgr, None)
            if self.stream:
                self.stream.set_latest_bgr(frame_bgr)
            return

        self.last_seen_time = time.time()
        self._publish_detected(True)
        self._publish_pose_and_tf(det)

        self._overlay(frame_bgr, det)
        if self.stream:
            self.stream.set_latest_bgr(frame_bgr)

    def destroy_node(self):
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = ArucoVisionNode()
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
# (A) 특정 마커만 (14번)
# ros2 run pinky_camera aruco_vision --ros-args -p watch_marker_ids:="[14]"

# (B) 여러 개 중 하나면 OK
# ros2 run pinky_camera aruco_vision --ros-args -p watch_marker_ids:="[600,601,602]"

# (C) 모름 → 아무거나
# ros2 run pinky_camera aruco_vision --ros-args -p watch_marker_ids:="[]"

