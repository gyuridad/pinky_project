#!/usr/bin/env python3
import json
import time
import math
import heapq
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
    # ROS 2(rclpy)에서 “콜백을 동시에(재진입 가능하게) 실행해도 된다” 는 성격의 Callback Group 타입을 가져오는 임포트
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32

import tf2_ros
from tf2_ros import TransformListener

# 인터페이스에 파일 생성해야함 !!!
from pinky_interfaces.msg import RobotState
from pinky_interfaces.action import ExecuteMission, MoveToPID, FollowAruco


def yaw_from_quat(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


@dataclass
class WP:
    x: float
    y: float
    yaw: float = 0.0


class PinkySystem1(Node):
    def __init__(self):
        super().__init__("pinky1_system1")

        self.cb_group = ReentrantCallbackGroup()

        # params
        self.declare_parameter("robot_name", "pinky1")
        self.declare_parameter("state_topic", "/pinky1/robot_state")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        self.declare_parameter("move_to_pid_action_name", "/pinky1/actions/move_to_pid")
        ##### follow #####
        self.declare_parameter("follow_aruco_action_name", "/pinky1/actions/follow_aruco")
        ##################
        ##### rtb #####
        self.declare_parameter("battery_voltage_topic", "/pinky1/battery_voltage")
        self.declare_parameter("battery_low_voltage_threshold", 7.5)
        self._low_v_latched = False   # ✅ 저전압 RTB 재발동 방지 래치
        ###############

        # ---- Waypoint params ----
        # 6개 예시 (네 맵 좌표로 수정!)
        default_wps = {
            "A": {"x": 0.0, "y": 0.0, "yaw": 0.0},    # 초기 위치
            "B": {"x": 0.04797892433180982, "y": 0.39166427058612435, "yaw": 0.06821276629599156},  # 상하차장소
            "C": {"x": 0.45056694884253223 , "y": 0.42498785253238847, "yaw": 1.6022886971368853},  # 검수대
            "D": {"x": 0.5491419897425645 , "y": -0.4895551603413119, "yaw": -1.6009117175028427},  # 조립대
            "M": {"x": 0.2692192294524333, "y": -1.1414312884666327, "yaw": 3.0821446616985146},   # 모듈창고
            # "P": {"x": 1.5, "y": 0.3, "yaw": 0.0},   # 미정
        }
        default_edges = {
            "A": ["B", "D", "M"],
            "B": ["A", "C"],
            "C": ["B", "D"],
            "D": ["A", "C", "M"],
            "M": ["A", "D"],
            # "P": ["M"],
        }
        self.declare_parameter("waypoints_json", json.dumps(default_wps))
        self.declare_parameter("waypoint_edges_json", json.dumps(default_edges))
        self.declare_parameter("waypoint_snap_max_dist", 0.5)

        self.declare_parameter("wp_timeout_sec", 60.0)       # waypoint 1개당 timeout
        self.declare_parameter("final_timeout_sec", 120.0)    # 최종 목표 timeout

        ##### rtb #####
        self._rtb_in_progress = False          # RTB 중복 방지
        self._low_v_since = None               # low 전압 지속 시간(선택)
        # self.is_charging = False             # ✅ 충전 상태 추가(추천)
        self.declare_parameter("battery_watch_period_sec", 1.0)
        self.declare_parameter("battery_low_hold_sec", 2.0)  # 2초 연속 low면 발동(추천)

        self.battery_watch_period = float(self.get_parameter("battery_watch_period_sec").value)
        self.battery_low_hold_sec = float(self.get_parameter("battery_low_hold_sec").value)
        ###############
    
        self.robot_name = self.get_parameter("robot_name").value
        self.state_topic = self.get_parameter("state_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.move_to_pid_action_name = self.get_parameter("move_to_pid_action_name").value

        ##### follow & rtb #####
        self.follow_aruco_action_name = self.get_parameter("follow_aruco_action_name").value
        self.battery_voltage_topic = self.get_parameter("battery_voltage_topic").value
        self.battery_low_voltage_threshold = float(self.get_parameter("battery_low_voltage_threshold").value)
        self.battery_voltage = 8.82
        self.battery_soc = 100.0 
        #######################
        
        self._load_waypoints()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- robot state internal ----
        self.system_state = "IDLE"     # 예: "IDLE", "RUNNING", "ERROR"
        self.queue_status = "idle"     # 예: "idle", "running", "done", "error", "canceled"
        self.current_index = -1        # 미션 plan의 steps 리스트에서 지금 몇 번째 작업 중인지 표시
        self.mission_id = ""
        self.plan_json = ""            # 받은 미션 계획 전체 JSON 문자열 원본
        self.history = []              # 각 step(또는 move_to_pid 호출) 결과를 쌓아두는 실행 기록
        self.last_violation = ""       # 가장 최근 규칙 위반/안전 위반 같은 내용을 한 줄로 저장하는 칸
        self.events = []               # 중요 이벤트 로그(INFO/WARN/ERROR 같은 이벤트를 시간과 함께 기록)

        self.latest_twist = Twist()
        self.primary_id = -1
        self.lost_sec = 999.0
        self.vision_json = "{}"

        # ---- RTB Future-chain runtime state ----  !!수정 필요!!
        self._rtb_goal_queue: List[Tuple[float, float, float, float, str]] = []

        # pub/sub
        self.state_pub = self.create_publisher(RobotState, self.state_topic, 10)

        ##### rtb #####
        self.battery_sub = self.create_subscription(
            Float32,  # 토픽 타입이 Float64면 Float64로 바꿔
            self.battery_voltage_topic,
            self._on_battery_voltage,
            10,
            callback_group=self.cb_group,
        )
        ###############

        # move_to_pid action client (단위액션)
        self.move_client = ActionClient(self, MoveToPID, self.move_to_pid_action_name, callback_group=self.cb_group)

        ##### follow #####
        self.follow_client = ActionClient(
            self, FollowAruco, self.follow_aruco_action_name, callback_group=self.cb_group
        )
        ##################

        # execute_mission action server (System1)
        self.exec_as = ActionServer(
            self,
            ExecuteMission,                       # 액션 타입(인터페이스) 이름
                # “Goal/Result/Feedback이 어떤 필드들로 구성되는지”가 ExecuteMission.action 파일로 정의돼있어야 함!!!
            "/pinky1/actions/execute_mission",    # 이 액션 서버의 이름(네임스페이스/주소)
                # System2(또는 다른 노드): ActionClient로 /pinky1/actions/execute_mission에 goal 전송
                # System1: ActionServer가 goal 수신 → _execute_mission_cb 실행 → 결과(Result) 반환
            execute_callback=self._execute_mission_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self.cb_group,
        )

        # ---------------- Timers ----------------
        self.create_timer(0.2, self.publish_state, callback_group=self.cb_group)
        ##### rtb #####
        self.create_timer(self.battery_watch_period, self._battery_watchdog, callback_group=self.cb_group)
        ###############

        self.get_logger().info(
            f"Pinky System1 ready. move_action={self.move_to_pid_action_name}, "
            f"follow_action={self.follow_aruco_action_name}"
        )

    ########## rtb ##########
    def _battery_is_low(self) -> bool:
        return float(self.battery_voltage) <= float(self.battery_low_voltage_threshold)

    def _battery_is_recovered(self) -> bool:
        # ✅ 같은 threshold를 쓰되, 회복은 더 높은 값에서만 인정(히스테리시스)
        margin = 0.5  # 예: 0.5V
        return float(self.battery_voltage) >= float(self.battery_low_voltage_threshold + margin)

    def _battery_watchdog(self):
        # IDLE에서만 자동 RTB (미션 중이면 execute 쪽이 처리)
        if self.system_state != "IDLE":
            self._low_v_since = None
                # 배터리 저전압 상태가 “얼마나 오래 지속됐는지”를 측정하기 위한 타임스탬프 변수야.
                # 즉, 순간적인 전압 드롭(노이즈) 때문에
                # RTB(Return-To-Base)가 바로 트리거되지 않도록 하는 디바운스 / 홀드 타이머 역할을 해
            return

        if self._rtb_in_progress:
            return

        # ✅ 저전압 래치가 걸려있으면, 회복될 때까지 RTB 재발동 금지
        if self._low_v_latched:
            if self._battery_is_recovered():      # ✅ recovered일 때만 unlatch!
                self._low_v_latched = False
                self._low_v_since = None
                self.events.append({"t": time.time(), "type": "INFO",
                                    "msg": f"battery recovered -> unlatch (V={self.battery_voltage:.2f}V)"})
            return

        
        if self._battery_is_low():
            now = time.time()
            # 전압이 낮아졌지만, 처음 감지됐을 때
            if self._low_v_since is None:
                self._low_v_since = now   # 저전압 처음 감지 now를 기록
                return
            # 전압이 계속 낮은 상태로 유지될 때, 지속 시간이 아직 부족
            if (now - self._low_v_since) < self.battery_low_hold_sec:
                return

            # 저전압이 충분히 오래 유지되었을 때
            # ✅ RTB 발동 + 래치 ON
            self._rtb_in_progress = True
            self._low_v_latched = True
            # “저전압 때문에 RTB가 발동됐다”는 원인 로그
            self.events.append({"t": now, "type": "WARN", "msg": f"IDLE low voltage -> RTB start (V={self.battery_voltage:.2f}V)"})

            self._start_rtb_future_chain()
        else:
            self._low_v_since = None

    def _finish_rtb(self, ok: bool):
        self.events.append({"t": time.time(),
                            "type": "INFO" if ok else "ERROR",
                            "msg": f"IDLE RTB {'done' if ok else 'failed'}"})
        self._rtb_goal_queue = []
        self._rtb_in_progress = False
        self._low_v_since = None

    # ---- send → accept 확인 → result 확인 → 다음 send → … 반복 ----
    def _start_rtb_future_chain(self):
        wp = self.waypoints.get("A")
        if wp is None:
            self.events.append({
                "t": time.time(),
                "type": "ERROR",
                "msg": "RTB failed: Home waypoint 'A' not found in waypoints"
            })
            self._finish_rtb(False)
            return

        ok, rx, ry, _ = self._tf_pose()  # 현재 로봇 위치 좌표 추출
        if not ok:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": "TF pose unavailable (RTB)"})
            self._finish_rtb(False)
            return
        
        s_wp, s_d = self._nearest_wp(rx, ry)   # 시작 waypoint명 / 현재 위치에서 시작 포인트까지 거리
        g_wp, g_d = self._nearest_wp(wp.x, wp.y)   # 초기 대기장소 A 좌표

        wp_timeout = float(self.get_parameter("wp_timeout_sec").value)
        final_timeout = float(self.get_parameter("final_timeout_sec").value)

        queue: List[Tuple[float, float, float, float, str]] = []

        if (s_wp is None or g_wp is None or
            s_d > self.wp_snap_max_dist or g_d > self.wp_snap_max_dist):

            self.events.append({"t": time.time(), "type": "WARN",
                                "msg": f"RTB WP snap failed: start_d={s_d:.2f}, goal_d={g_d:.2f} -> direct"})
            queue.append((wp.x, wp.y, wp.yaw, final_timeout, "home_direct"))
        else:
            chain = self._dijkstra_wp_path(s_wp, g_wp)
            if not chain:
                self.events.append({"t": time.time(), "type": "ERROR", "msg": f"RTB No WP path {s_wp}->{g_wp}"})
                self._finish_rtb(False)
                return

            self.events.append({"t": time.time(), "type": "INFO", "msg": f"RTB WP path {s_wp}->{g_wp}: {chain}"})

            for name in chain:
                w = self.waypoints[name]
                queue.append((w.x, w.y, w.yaw, wp_timeout, f"wp:{name}"))

            queue.append((wp.x, wp.y, wp.yaw, final_timeout, "home_final"))

        self._rtb_goal_queue = queue
        self._rtb_send_next_goal()

    def _rtb_send_next_goal(self):
        if not self._rtb_goal_queue:
            self._finish_rtb(True)
            return

        if not self.move_client.wait_for_server(timeout_sec=2.0):
            self.events.append({"t": time.time(), "type": "ERROR", "msg": "RTB MoveToPID server not available"})
            self._finish_rtb(False)
            return
        
        x, y, yaw, timeout_sec, label = self._rtb_goal_queue.pop(0)
        ps = self._pose_stamped(x, y, yaw)   # ros2 형식으로 포즈를 변경

        goal = MoveToPID.Goal()
        goal.target = ps
        goal.timeout_sec = float(timeout_sec)

        self.events.append({"t": time.time(), "type": "INFO",
                            "msg": f"RTB send MoveToPID ({label}) x={x:.2f},y={y:.2f},t={timeout_sec:.1f}s"})
        
        send_fut = self.move_client.send_goal_async(goal)
        send_fut.add_done_callback(lambda fut: self._rtb_on_goal_response(fut, label, x, y))

    def _rtb_on_goal_response(self, fut, label: str, x: float, y: float):
        try:
            goal_handle = fut.result()
        except Exception as e:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": f"RTB goal response error: {e}"})
            self._finish_rtb(False)
            return

        if not goal_handle.accepted:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": f"RTB goal rejected ({label})"})
            self._finish_rtb(False)
            return

        res_fut = goal_handle.get_result_async()
        res_fut.add_done_callback(lambda rfut: self._rtb_on_result(rfut, label, x, y))

    def _rtb_on_result(self, fut, label: str, x: float, y: float):
        try:
            res = fut.result().result
            ok = bool(res.success)
            status = int(getattr(res, "status", 0))
            msg = str(getattr(res, "message", ""))
        except Exception as e:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": f"RTB result exception: {e}"})
            self._finish_rtb(False)
            return

        self.history.append({
            "task": "rtb_move_to_pid",
            "label": label,
            "success": ok,
            "status": status,
            "message": msg,
            "x": x,
            "y": y,
        })

        if not ok:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": f"RTB failed at {label}: {msg}"})
            self._finish_rtb(False)
            return

        self._rtb_send_next_goal()
    ########## rtb ##########

    # ---------------- Waypoint graph ----------------
    def _load_waypoints(self):
        wps_raw = json.loads(self.get_parameter("waypoints_json").value)
        edges_raw = json.loads(self.get_parameter("waypoint_edges_json").value)
        self.waypoints: Dict[str, WP] = {
            k: WP(float(v["x"]), float(v["y"]), float(v.get("yaw", 0.0)))
            for k, v in wps_raw.items()
        }
        self.wp_edges: Dict[str, List[str]] = {k: list(v) for k, v in edges_raw.items()}
        self.wp_snap_max_dist = float(self.get_parameter("waypoint_snap_max_dist").value)

    def _dist2(self, ax, ay, bx, by):
        dx = ax - bx
        dy = ay - by
        return dx*dx + dy*dy
    
    # 현재 위치 (x, y)에서 가장 가까운 웨이포인트(waypoint)를 찾음 
    def _nearest_wp(self, x: float, y: float) -> Tuple[Optional[str], float]:
        best_name = None  # 현재까지 "가장 가까운 웨이포인트 이름"
        best_d2 = 1e18    # 현재까지 "가장 가까운 거리의 제곱" (아주 큰 값으로 시작)
        for name, wp in self.waypoints.items():
            d2 = self._dist2(x, y, wp.x, wp.y)   # 저장된 모든 waypoint들을 하나씩 확인
            if d2 < best_d2:
                best_d2 = d2
                best_name = name
        return best_name, math.sqrt(best_d2) if best_name is not None else 1e18

    def _dijkstra_wp_path(self, start: str, goal: str) -> List[str]:
        # 1) 준비물 세팅
        pq = [(0.0, start)]    # “지금까지 비용이 가장 작은 후보”를 빨리 꺼내기 위한 우선순위 큐
        dist = {start: 0.0}    # “start에서 각 노드까지 알고 있는 최소 비용”
        prev = {start: None}   # “최단경로로 올 때, 바로 이전 노드가 뭐였는지” (경로 복원용)

        while pq:
            # 2) 가장 싼 후보부터 하나씩 꺼내서 확장
            cost, u = heapq.heappop(pq)
            if u == goal:
                break
            if cost != dist.get(u, 1e18):
                continue
            # 3) u에서 갈 수 있는 이웃 v들을 검사
            for v in self.wp_edges.get(u, []):
                if v not in self.waypoints:
                    continue
                wu = self.waypoints[u]
                wv = self.waypoints[v]
                # 4) u→v로 가는 간선 비용(거리) 계산
                w = math.hypot(wv.x - wu.x, wv.y - wu.y)
                nd = cost + w   # start→u까지 비용(cost) + u→v 거리(w) = start→v까지 새 비용(nd)
                # 5) 더 짧게 갱신 가능하면 업데이트
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if goal not in prev and goal != start:
            return []

        path = []
        cur = goal
        path.append(cur)
        while cur != start:
            cur = prev.get(cur)
            if cur is None:
                return []
            path.append(cur)
        path.reverse()
        return path

    def _pose_stamped(self, x: float, y: float, yaw: float) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        qx, qy, qz, qw = quat_from_yaw(float(yaw))
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        return ps

    # ---------------- TF / State publish ----------------
    def _tf_pose(self):
        """map->base pose"""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),                 # 최신(가능한) 변환
                timeout=Duration(seconds=0.2),      # ✅ 기다려줌
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
            return True, float(t.x), float(t.y), float(yaw)
        except Exception as e:
            self.get_logger().warn(f"[TF] {self.map_frame}->{self.base_frame} lookup failed: {e}")
            return False, 0.0, 0.0, 0.0
        
    # 로봇의 현재 상태 스냅샷(자세/속도/배터리/미션/비전/안전/이벤트 등)을 /robot_state 같은 토픽에 계속 발행
    def publish_state(self):
        ok, x, y, yaw = self._tf_pose()

        msg = RobotState()
        msg.stamp = self.get_clock().now().to_msg()

        msg.robot_name = self.robot_name
        msg.system_state = self.system_state

        msg.pose_frame = self.map_frame
        msg.pose_x = x
        msg.pose_y = y
        msg.pose_yaw = yaw

        msg.vel_vx = float(self.latest_twist.linear.x)
        msg.vel_vy = float(self.latest_twist.linear.y)
        msg.vel_wz = float(self.latest_twist.angular.z)

        msg.battery_voltage = float(self.battery_voltage)

        msg.primary_id = int(self.primary_id)
        msg.lost_sec = float(self.lost_sec)
        msg.vision_json = self.vision_json

        msg.mission_id = self.mission_id
        msg.plan_json = self.plan_json

        msg.current_index = int(self.current_index)
        msg.queue_status = self.queue_status
        msg.history_json = json.dumps(self.history, ensure_ascii=False)

        msg.roe_ok = True
        msg.safe_backstop = True
        msg.max_speed = 0.6
        msg.last_violation = self.last_violation

        msg.events_json = json.dumps(self.events, ensure_ascii=False)

        self.state_pub.publish(msg)

    # ---------------- ROS action glue ----------------
    def _goal_cb(self, goal_request):
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        self.get_logger().warn("[System1] cancel requested")
        return CancelResponse.ACCEPT

    ########## rtb ##########
    def _on_battery_voltage(self, msg):
        self.battery_voltage = float(msg.data)
    ########## rtb ##########

    # ---------------- MoveToPID (async) ----------------
    async def _call_move_to_pid(self, goal_pose: PoseStamped, timeout_sec: float) -> bool:
        # 1) 가게(액션 서버)가 열려있는지 확인
        if not self.move_client.wait_for_server(timeout_sec=2.0):
            self.events.append({"t": time.time(), "type": "ERROR", "msg": "MoveToPID server not available"})
            return False

        # 2) 주문서(goal 메시지) 작성
        goal = MoveToPID.Goal()     # “주문서 양식”
        goal.target = goal_pose     # “어디로 갈지” (PoseStamped: x,y,yaw 포함)
        goal.timeout_sec = float(timeout_sec)  # “몇 초까지 기다릴지(시간 제한)”

        # 3) 주문 보내기 (비동기)
        send_fut = self.move_client.send_goal_async(goal)
        goal_handle = await send_fut   # await로 “가게가 주문을 받았는지” 응답을 기다려.

        # 4) 가게가 주문을 거절했는지 확인
        if not goal_handle.accepted:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": "MoveToPID goal rejected"})
            return False

        # 5) 배달 완료(액션 결과)까지 기다리기
        res_fut = goal_handle.get_result_async()
        res = await res_fut
        ok = bool(res.result.success)  # res.result.success가 True면 이동 성공, False면 실패.

        # 6) 실행 기록(history)에 결과 저장
        self.history.append({
            "task": "move_to_pid",
            "success": ok,
            "status": int(getattr(res.result, "status", 0)),
            "message": str(getattr(res.result, "message", "")),
            "x": float(goal_pose.pose.position.x),
            "y": float(goal_pose.pose.position.y),
        })
        return ok
    
    ########## follow ##########
    async def _call_follow_aruco(self, marker_id: int, timeout_sec: float = 40.0) -> bool:
        marker_id = int(marker_id)
        timeout_sec = float(timeout_sec)

        # 1) 서버 대기
        if not self.follow_client.wait_for_server(timeout_sec=5.0):
            self.events.append({"t": time.time(), "type": "ERROR", "msg": "FollowAruco server not available"})
            self.history.append({
                "task": "follow_aruco",
                "success": False,
                "marker_id": marker_id,
                "cause": "server_unavailable",
            })
            return False
        
        # 2) 주문서(goal 메시지) 작성
        goal = FollowAruco.Goal()
        goal.marker_id = marker_id

        # goal에 timeout_sec 필드가 실제로 있으면 사용 (없으면 제거해야 함)
        if hasattr(goal, "timeout_sec"):
            goal.timeout_sec = timeout_sec

        t0 = time.time()

        # 3) 주문 보내기 (비동기)
        # (주의) FollowAruco.goal에 timeout이 없다면 여기선 timeout은 “System1에서 기다리는 시간”으로만 사용
        send_fut = self.follow_client.send_goal_async(goal)
        goal_handle = await send_fut

        # 4) 주문을 거절했는지 확인
        if not goal_handle.accepted:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": "FollowAruco goal rejected"})
            self.history.append({
                "task": "follow_aruco",
                "success": False,
                "marker_id": marker_id,
                "cause": "goal_rejected",
            })
            return False
        
        # 5) 배달 완료(액션 결과)까지 기다리기
        res_fut = goal_handle.get_result_async()
        res = await res_fut

        ok = bool(res.result.success)
        msg = str(getattr(res.result, "message", ""))

        # 6) 실행 기록(history)에 결과 저장
        self.history.append({
            "task": "follow_aruco",
            "success": ok,
            "marker_id": marker_id,
            "timeout_sec": timeout_sec,
            "elapsed": time.time() - t0,
            "message": msg,
            "cause": "ok" if ok else "server_failed",
        })

        if not ok:
            self.events.append({"t": time.time(), "type": "WARN", "msg": f"FollowAruco failed: {msg}"})

        return ok
    ########## follow ##########

    # “지금 로봇 위치 → 목표 위치”를 갈 때, 웨이포인트 그래프(A/B/C/...)를 이용해서 ‘경유 경로’를 만들고,
    #  그 경로대로 MoveToPID를 여러 번 호출해서 단계적으로 이동시키는 함수야.
    async def _goto_via_waypoints(self, goal_x: float, goal_y: float, goal_yaw: float) -> bool:
        ok, rx, ry, _ = self._tf_pose()
        if not ok:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": "TF pose unavailable"})
            return False

        s_wp, s_d = self._nearest_wp(rx, ry)
        g_wp, g_d = self._nearest_wp(goal_x, goal_y)

        if s_wp is None or g_wp is None or s_d > self.wp_snap_max_dist or g_d > self.wp_snap_max_dist:
            # 스냅 실패 -> 직행 (원하면 여기서 False로 막아도 됨)
            self.events.append({
                "t": time.time(),
                "type": "WARN",
                "msg": f"WP snap failed: start_d={s_d:.2f}, goal_d={g_d:.2f} (max={self.wp_snap_max_dist:.2f}) -> direct"
            })
            ps = self._pose_stamped(goal_x, goal_y, goal_yaw)
            return await self._call_move_to_pid(ps, float(self.get_parameter("final_timeout_sec").value))

        # 스냅이 성공하면, 웨이포인트 그래프에서 최단 경로 찾기 (다익스트라)
        chain = self._dijkstra_wp_path(s_wp, g_wp)
        if not chain:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": f"No WP path {s_wp}->{g_wp}"})
            return False

        self.events.append({"t": time.time(), "type": "INFO", "msg": f"WP path {s_wp}->{g_wp}: {chain}"})

        wp_timeout = float(self.get_parameter("wp_timeout_sec").value)
        final_timeout = float(self.get_parameter("final_timeout_sec").value)

        # 경유지(waypoint)들을 순서대로 하나씩 이동시키기
        for name in chain:
            wp = self.waypoints[name]
            ps = self._pose_stamped(wp.x, wp.y, wp.yaw)
            ok = await self._call_move_to_pid(ps, wp_timeout)
            if not ok:
                return False

        ps = self._pose_stamped(goal_x, goal_y, goal_yaw)
        return await self._call_move_to_pid(ps, final_timeout)

    ########## rtb ##########
    async def _return_to_home(self) -> bool:
        """
        배터리 부족 등 비상 상황에서 홈(A)로 복귀.
        홈은 네 waypoint "A"를 사용.
        """
        wp = self.waypoints.get("A")
        if wp is None:
            self.events.append({"t": time.time(), "type": "ERROR", "msg": "Home waypoint 'A' not found"})
            return False

        self.events.append({
            "t": time.time(),
            "type": "WARN",
            "msg": f"RTB triggered: battery_voltage={float(self.battery_voltage):.2f}V (thr={self.battery_low_voltage_threshold:.2f}V)"
        })
        ok = await self._goto_via_waypoints(wp.x, wp.y, wp.yaw)
        return ok
    ########## rtb ##########
    
    # System1의 “ExecuteMission 액션 서버”가 goal(미션 계획)을 받았을 때 실행되는 메인 실행기
    async def _execute_mission_cb(self, goal_handle):
        ########## rtb ##########
        # RTB 중이면 새 미션은 즉시 거절(중복 방지)
        if self._rtb_in_progress:
            result = ExecuteMission.Result()
            goal_handle.abort()
            result.success = False
            result.message = "RTB already in progress"
            return result

        # 미션 시작 직후 바로 배터리 체크
        if float(self.battery_voltage) <= float(self.battery_low_voltage_threshold):
            self._rtb_in_progress = True
            try:
                ok_rtb = await self._return_to_home()
            finally:
                self._rtb_in_progress = False
                self._low_v_since = None

            result = ExecuteMission.Result()
            goal_handle.abort()
            result.success = False
            result.message = (
                f"battery low -> RTB {'done' if ok_rtb else 'failed'} (V={self.battery_voltage:.2f}V)"
            )
            return result
        ########## rtb ##########

        # 1) “미션 시작” 상태로 바꾸기
        self.system_state = "RUNNING"
        self.queue_status = "running"
        self.current_index = 0

        try:
            # 2) 받은 plan_json을 파싱하기 (JSON → dict)
            plan = json.loads(goal_handle.request.plan_json)
        except Exception as e:
            self.system_state = "ERROR"
            self.queue_status = "error"
            result = ExecuteMission.Result()
            result.success = False
            result.message = f"plan_json parse failed: {e}"
            return result

        # 3) 미션 ID와 원본 계획 저장
        self.mission_id = str(plan.get("mission_id", ""))
        self.plan_json = goal_handle.request.plan_json

        # 4) steps를 꺼내서 “순서대로” 실행
        steps = plan.get("steps", [])

        for i, s in enumerate(steps):
            # 5) 중간에 “취소 요청”이 들어오면 즉시 종료
            if goal_handle.is_cancel_requested:
                self.queue_status = "canceled"
                self.system_state = "IDLE"
                goal_handle.canceled()
                result = ExecuteMission.Result()
                result.success = False
                result.message = "canceled"
                return result

            self.current_index = i

            task = s.get("task", "")

            if task == "move_to":    
                # 7-1) step에서 목표 좌표를 꺼냄
                gx = float(s.get("x", 0.0))
                gy = float(s.get("y", 0.0))
                gyaw = float(s.get("yaw", 0.0))

                # 7-2) 웨이포인트 경유할지 결정
                use_wp = bool(s.get("use_waypoints", True))
                if use_wp:
                    ok = await self._goto_via_waypoints(gx, gy, gyaw)
                else:
                    ps = self._pose_stamped(gx, gy, gyaw)
                    ok = await self._call_move_to_pid(ps, float(s.get("timeout_sec", self.get_parameter("final_timeout_sec").value)))

                # 8) move_to가 실패하면 “미션 전체 실패”로 처리
                if not ok:
                    self.queue_status = "error"
                    self.system_state = "ERROR"
                    goal_handle.abort()
                    result = ExecuteMission.Result()
                    result.success = False
                    result.message = f"move_to failed at index {i}"
                    return result
            
            ########## follow ##########
            elif task == "follow_aruco":
                marker_id = int(s.get("marker_id", 0))
                timeout_sec = float(s.get("timeout_sec", 40.0))
                if marker_id <= 0:
                    self.queue_status = "error"
                    self.system_state = "ERROR"
                    goal_handle.abort()
                    result = ExecuteMission.Result()
                    result.success = False
                    result.message = f"follow_aruco invalid marker_id: {marker_id}"
                    return result
                
                ok = await self._call_follow_aruco(marker_id, timeout_sec=timeout_sec)
                if not ok:
                    self.queue_status = "error"
                    self.system_state = "ERROR"
                    goal_handle.abort()
                    result = ExecuteMission.Result()
                    result.success = False
                    result.message = f"follow_aruco failed at index {i}"
                    return result
            ########## follow ##########

            else:
                self.queue_status = "error"
                self.system_state = "ERROR"
                goal_handle.abort()
                result = ExecuteMission.Result()
                result.success = False
                result.message = f"unknown task: {task}"
                return result

        # 9) steps를 끝까지 다 성공하면 “미션 성공”
        self.queue_status = "done"
        self.system_state = "IDLE"
        goal_handle.succeed()
        result = ExecuteMission.Result()
        result.success = True
        result.message = "mission done"
        return result


from rclpy.executors import SingleThreadedExecutor

def main():
    rclpy.init()
    node = PinkySystem1()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == "__main__":
    main()



### 실행 예시
# ros2 run controller controller

# ros2 run controller controller \
#   --ros-args -p battery_low_voltage_threshold:=7.5


### 상위 제어에서 보낸 실행문 간단 버전
# ros2 action send_goal /pinky1/actions/execute_mission pinky_interfaces/action/ExecuteMission "{plan_json: '{\"mission_id\":\"m_test_001\",\"steps\":[{\"task\":\"move_to\",\"x\":0.450,\"y\":0.424,\"yaw\":1.602,\"use_waypoints\":true},{\"task\":\"move_to\",\"x\":0.269,\"y\":-1.141,\"yaw\":3.082,\"use_waypoints\":true}]}' }"

# ros2 action send_goal /pinky1/actions/execute_mission pinky_interfaces/action/ExecuteMission \
# "{plan_json: '{\"mission_id\":\"m_dock_test_001\",\"steps\":[{\"task\":\"move_to\",\"x\":0.450,\"y\":0.424,\"yaw\":1.602,\"use_waypoints\":true},{\"task\":\"move_to\",\"x\":0.269,\"y\":-1.141,\"yaw\":3.082,\"use_waypoints\":true}]}' }"

# ros2 action send_goal /pinky1/actions/execute_mission pinky_interfaces/action/ExecuteMission \
# "{plan_json: '{\"mission_id\":\"m_dock_test_001\",\"steps\":[{\"task\":\"move_to\",\"x\":0.269,\"y\":-1.141,\"yaw\":3.082,\"use_waypoints\":true},{\"task\":\"follow_aruco\",\"marker_id\":600,\"timeout_sec\":35.0}]}' }"


# ros2 action send_goal /pinky1/actions/execute_mission pinky_interfaces/action/ExecuteMission \
# "{plan_json: '{
#   \"mission_id\":\"m_dock_test_001\",
#   \"steps\":[
#     {\"task\":\"move_to\",\"x\":0.450,\"y\":0.424,\"yaw\":1.602,\"use_waypoints\":true},
#     {\"task\":\"move_to\",\"x\":0.269,\"y\":-1.141,\"yaw\":3.082,\"use_waypoints\":true},
#     {\"task\":\"follow_aruco\",\"marker_id\":600,\"timeout_sec\":35.0}
#   ] }' }"