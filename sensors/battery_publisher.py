#!/usr/bin/env python3
import re
import subprocess
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import Float32

class BatteryBridge(Node):
    def __init__(self):
        super().__init__("battery_bridge")

        self.declare_parameter("cmd", "/home/pinky/ap/check_battery_cli.py")
        self.declare_parameter("period_sec", 1.0)
        self.declare_parameter("charger_topic", "/pinky1/charger_connected")
        self.declare_parameter("soc_topic", "/pinky1/battery_soc")
        self.declare_parameter("volt_topic", "/pinky1/battery_voltage")

        self.cmd = str(self.get_parameter("cmd").value)
        self.period = float(self.get_parameter("period_sec").value)

        self.pub_chg = self.create_publisher(Bool, self.get_parameter("charger_topic").value, 10)
        self.pub_soc = self.create_publisher(Float32, self.get_parameter("soc_topic").value, 10)
        self.pub_v   = self.create_publisher(Float32, self.get_parameter("volt_topic").value, 10)

        self.create_timer(self.period, self._tick)
        self.get_logger().info(f"BatteryBridge started: cmd='{self.cmd}', period={self.period}s")

    def _tick(self):
        try:
            # out = subprocess.check_output(self.cmd, shell=True, text=True, timeout=1.5)
            out = subprocess.check_output(
                ["python3", self.cmd],
                text=True,
                timeout=2.0
            ).strip()
        except Exception as e:
            self.get_logger().warn(f"battery cmd failed: {e}")
            return
        
        # ✅ CLI 출력 예: "SOC=100.0 V=8.82"
        m = re.search(r"SOC=([0-9]+(?:\.[0-9]+)?)\s+V=([0-9]+(?:\.[0-9]+)?)", out)
        if not m:
            self.get_logger().warn(f"parse failed. out='{out}'")
            return

        soc = float(m.group(1))
        volt = float(m.group(2))
        self.pub_soc.publish(Float32(data=soc))
        self.pub_v.publish(Float32(data=volt))

        # ⚠️ 현재 CLI에는 충전 여부가 없으니 일단 False로 publish
        # (나중에 CLI 출력에 CHG=0/1을 추가하면 여기서 파싱 가능)
        self.pub_chg.publish(Bool(data=False))

        self.get_logger().debug(f"parsed: soc={soc:.1f} volt={volt:.2f}")

def main():
    rclpy.init()
    node = BatteryBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()



