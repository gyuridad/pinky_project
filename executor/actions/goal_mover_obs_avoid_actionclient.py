#!/usr/bin/env python3
import sys
import math
import asyncio
import time
from enum import IntEnum
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64, Bool

import tf2_ros
from tf2_ros import TransformException

from pinky_interfaces.action import MoveToPID  # ✅ 너 프로젝트 액션 타입


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, min_output=-1.0, max_output=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def update_gains(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd

    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def compute(self, error, current_time=None):
        if current_time is None:
            current_time = time.time()

        if self.last_time is None:
            self.last_time = current_time
            self.previous_error = error
            return 0.0

        dt = current_time - self.last_time
        if dt <= 0.0:
            return 0.0

        p = self.kp * error

        self.integral += error * dt
        max_integral = abs(self.max_output) / (abs(self.ki) + 1e-6)
        self.integral = max(-max_integral, min(max_integral, self.integral))
        i = self.ki * self.integral

        d = self.kd * (error - self.previous_error) / dt

        out = p + i + d
        out = max(self.min_output, min(self.max_output, out))

        self.previous_error = error
        self.last_time = current_time
        return out


class Mode(IntEnum):
    TURN_TO_GOAL = 0
    GO_STRAIGHT = 1
    FINAL_ALIGN = 2


class GoalMover(Node):
    """
    - TF(map->base_link)로 현재 pose를 읽어서 /cmd_vel PID 제어
    - goal_callback(x, y, yaw_deg)로 목표를 직접 세팅
    """

    def __init__(self):
        super().__init__("goal_mover_amcl_pose")

        # ---- Parameters ----
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("obstacle_topic", "/obstacle_detected")

        # ✅ action name can be overridden
        self.declare_parameter("action_name", "pinky1/actions/move_to_pid")

        # PID gains
        self.declare_parameter("linear_P", 0.5)
        self.declare_parameter("linear_I", 0.0)
        self.declare_parameter("linear_D", 0.0)
        self.declare_parameter("angular_P", 0.2)
        self.declare_parameter("angular_I", 0.0)
        self.declare_parameter("angular_D", 0.05)

        # Tolerances
        self.declare_parameter("angle_tolerance_deg", 12.0)
        self.declare_parameter("pos_tolerance", 0.03)
        self.declare_parameter("final_yaw_tolerance_deg", 5.0)

        self.declare_parameter("enable_pid", True)

        # Speed limits
        self.declare_parameter("max_linear_speed", 0.30)
        self.declare_parameter("max_angular_speed", 1.5)

        self.declare_parameter("min_linear_speed", 0.06)
        self.declare_parameter("min_angular_speed", 0.10)
        self.declare_parameter("min_speed_distance_threshold", 0.30)

        # TF lookup
        self.declare_parameter("tf_timeout_sec", 0.2)

        # Control rate
        self.declare_parameter("control_period_sec", 0.01)

        self._load_params()

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- PID ----
        self.linear_pid = PIDController(
            kp=self.linear_P, ki=self.linear_I, kd=self.linear_D,
            min_output=-self.max_linear_speed, max_output=self.max_linear_speed,
        )
        self.angular_pid = PIDController(
            kp=self.angular_P, ki=self.angular_I, kd=self.angular_D,
            min_output=-self.max_angular_speed, max_output=self.max_angular_speed,
        )

        # ---- State ----
        self.goal_msg: Optional[PoseStamped] = None
        self.mode: Mode = Mode.TURN_TO_GOAL
        self.obstacle_active = False 
        self._reached_flag = False

        # ---- ROS I/O ----
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        self.distance_error_pub = self.create_publisher(Float64, "/distance_error", 10)
        self.angle_error_pub = self.create_publisher(Float64, "/angle_error", 10)
        self.final_yaw_error_pub = self.create_publisher(Float64, "/final_yaw_error", 10)

        self.sub_ob = self.create_subscription(Bool, self.obstacle_topic, self._on_obstacle, 10)

        # ✅ ActionServer (System1이 호출하는 엔드포인트)
        self._as = ActionServer(
            self,
            MoveToPID,
            self.action_name,                 # ← System1에서 /pinky1/actions/move_to_pid 같은 네임스페이스면 launch에서 remap/namespace로 맞추기
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
        )
        

        self.timer = self.create_timer(self.control_period_sec, self.control_loop)
        self.get_logger().info("✅ GoalMover MoveToPID ActionServer ready: name={self.action_name}")

    def _load_params(self):
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.obstacle_topic = self.get_parameter("obstacle_topic").value
        self.action_name = self.get_parameter("action_name").value

        self.linear_P = float(self.get_parameter("linear_P").value)
        self.linear_I = float(self.get_parameter("linear_I").value)
        self.linear_D = float(self.get_parameter("linear_D").value)

        self.angular_P = float(self.get_parameter("angular_P").value)
        self.angular_I = float(self.get_parameter("angular_I").value)
        self.angular_D = float(self.get_parameter("angular_D").value)

        self.angle_tolerance = math.radians(float(self.get_parameter("angle_tolerance_deg").value))
        self.final_yaw_tolerance = math.radians(float(self.get_parameter("final_yaw_tolerance_deg").value))
        self.pos_tolerance = float(self.get_parameter("pos_tolerance").value)

        self.enable_pid = bool(self.get_parameter("enable_pid").value)

        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.min_linear_speed = float(self.get_parameter("min_linear_speed").value)
        self.min_angular_speed = float(self.get_parameter("min_angular_speed").value)
        self.min_speed_distance_threshold = float(self.get_parameter("min_speed_distance_threshold").value)

        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.control_period_sec = float(self.get_parameter("control_period_sec").value)

    # -------- Action callbacks --------
    def _goal_cb(self, goal_request):
        return GoalResponse.ACCEPT
    
    def _cancel_cb(self, goal_handle):
        self.get_logger().warn("[MoveToPID] cancel requested → stop")
        self.goal_msg = None
        self._reached_flag = False
        self.cmd_pub.publish(Twist())
        return CancelResponse.ACCEPT
    
    def _execute_cb(self, goal_handle):
        """
        System1이 보낸 goal_request:
          - goal_handle.request.target (PoseStamped)
          - goal_handle.request.timeout_sec (float)
        """
        req = goal_handle.request
        target: PoseStamped = req.target
        timeout_sec = float(getattr(req, "timeout_sec", 0.0)) if req is not None else 0.0

        # 목표 세팅
        self._reached_flag = False
        self.goal_msg = target
        self.mode = Mode.TURN_TO_GOAL
        self.linear_pid.reset()
        self.angular_pid.reset()

        t0 = time.time()
        self.get_logger().info(
            f"[MoveToPID] start: x={target.pose.position.x:.2f}, y={target.pose.position.y:.2f}, timeout={timeout_sec:.1f}s"
        )

        # 목표가 끝날 때까지 기다리면서 상태를 체크하는 루프
        while rclpy.ok():
            # 1) cancel
            if goal_handle.is_cancel_requested:
                self.get_logger().warn("[MoveToPID] cancel requested")
                self.goal_msg = None
                self._reached_flag = False
                self.cmd_pub.publish(Twist())
                goal_handle.canceled()

                res = MoveToPID.Result()
                res.success = False
                res.message = "canceled"
                res.status = 1
                return res

            # 2) reached
            if self._reached_flag:
                # ✅ 다음 goal에 영향 없게 정리
                self.goal_msg = None
                self._reached_flag = False
                self.cmd_pub.publish(Twist())
                goal_handle.succeed()

                res = MoveToPID.Result()
                res.success = True
                res.message = "reached"
                res.status = 0
                return res

            # 3) timeout
            if timeout_sec > 0.0 and (time.time() - t0) > timeout_sec:
                self.get_logger().warn("[MoveToPID] timeout → stop")
                self.goal_msg = None
                self._reached_flag = False
                self.cmd_pub.publish(Twist())
                goal_handle.abort()

                res = MoveToPID.Result()
                res.success = False
                res.message = "timeout"
                res.status = 2
                return res

            time.sleep(0.02)  # ✅ asyncio 대신 그냥 짧게 sleep

        # 4) shutdown (ROS 종료)
        self.goal_msg = None
        self._reached_flag = False
        self.cmd_pub.publish(Twist())
        goal_handle.abort()

        res = MoveToPID.Result()
        res.success = False
        res.message = "shutdown"
        res.status = 3
        return res

    # -------- Control loop --------
    def _get_robot_pose_in_map(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec),
            )
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {self.map_frame}->{self.base_frame}: {e}")
            return None

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        return x, y, yaw

    def _goal_xy_yaw_in_map(self) -> Optional[Tuple[float, float, float]]:
        if self.goal_msg is None:
            return None
        gx = self.goal_msg.pose.position.x
        gy = self.goal_msg.pose.position.y
        gq = self.goal_msg.pose.orientation
        gyaw = yaw_from_quat(gq.x, gq.y, gq.z, gq.w)
        return gx, gy, gyaw

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    def control_loop(self):
        # ✅ 장애물 상태면 PID 루프 자체를 멈추고 stop 유지
        if self.obstacle_active:
            self._publish_stop()
            return

        if self.goal_msg is None:
            return

        robot = self._get_robot_pose_in_map()
        goal = self._goal_xy_yaw_in_map()
        if robot is None or goal is None:
            return

        rx, ry, ryaw = robot
        gx, gy, gyaw = goal

        dx = gx - rx
        dy = gy - ry
        dist = math.hypot(dx, dy)
        heading_to_goal = math.atan2(dy, dx)
        heading_err = normalize_angle(heading_to_goal - ryaw)
        final_yaw_err = normalize_angle(gyaw - ryaw)

        dm = Float64(); dm.data = dist
        am = Float64(); am.data = heading_err
        fm = Float64(); fm.data = final_yaw_err
        self.distance_error_pub.publish(dm)
        self.angle_error_pub.publish(am)
        self.final_yaw_error_pub.publish(fm)

        if dist < self.pos_tolerance:
            self.mode = Mode.FINAL_ALIGN

        cmd = Twist()

        if self.mode == Mode.TURN_TO_GOAL:
            if abs(heading_err) <= self.angle_tolerance:
                self.mode = Mode.GO_STRAIGHT
                self.angular_pid.reset()
                cmd.angular.z = 0.0
                cmd.linear.x = 0.0
            else:
                w = self.angular_pid.compute(heading_err) if self.enable_pid else (self.angular_P * heading_err)
                w = max(-self.max_angular_speed, min(self.max_angular_speed, w))
                if abs(w) < self.min_angular_speed:
                    w = math.copysign(self.min_angular_speed, w)
                cmd.angular.z = w
                cmd.linear.x = 0.0

        elif self.mode == Mode.GO_STRAIGHT:
            if abs(heading_err) > self.angle_tolerance:
                self.mode = Mode.TURN_TO_GOAL
                self.linear_pid.reset()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                v = self.linear_pid.compute(dist) if self.enable_pid else (self.linear_P * dist)
                v = max(-self.max_linear_speed, min(self.max_linear_speed, v))
                if dist > self.min_speed_distance_threshold:
                    if abs(v) < self.min_linear_speed:
                        v = math.copysign(self.min_linear_speed, v)
                cmd.linear.x = v
                cmd.angular.z = 0.0

        elif self.mode == Mode.FINAL_ALIGN:
            if abs(final_yaw_err) <= self.final_yaw_tolerance:
                self.get_logger().info("✅ reached goal pose. stop.")
                self._reached_flag = True
                self.goal_msg = None
                self.mode = Mode.TURN_TO_GOAL
                self.linear_pid.reset()
                self.angular_pid.reset()
                self._publish_stop()
                return
            else:
                w = self.angular_pid.compute(final_yaw_err) if self.enable_pid else (self.angular_P * final_yaw_err)
                w = max(-self.max_angular_speed, min(self.max_angular_speed, w))
                if abs(w) < self.min_angular_speed:
                    w = math.copysign(self.min_angular_speed, w)
                cmd.angular.z = w
                cmd.linear.x = 0.0

        self.cmd_pub.publish(cmd)

    def _on_obstacle(self, msg: Bool):
        self.obstacle_active = bool(msg.data)
        if self.obstacle_active:
            self._publish_stop()



from rclpy.executors import MultiThreadedExecutor

def main(args=None):

    # ✅ 2) ROS init + node 생성
    rclpy.init(args=args)
    node = GoalMover()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    # ✅ 4) spin
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()


### 실행예시
# ros2 run actions goal_mover_obs_avoid_action 

# ros2 run actions goal_mover_obs_avoid_action --ros-args \
#   -r __ns:=/pinky1 \
#   -r move_to_pid:=actions/move_to_pid

