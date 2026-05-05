#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import random
import time

JOINTS = ['Revolute 1', 'Revolute 2', 'Revolute 3',
          'Revolute 4', 'Revolute 5', 'Revolute 6']

def deg_to_rad(deg):
    return math.radians(deg - 90.0)  # center=0, range=-pi/2 to pi/2

class RandomMotion(Node):
    def __init__(self):
        super().__init__('random_motion')
        self.pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )
        self.timer = self.create_timer(3.0, self.send_random)
        self.get_logger().info('Random motion node started, sending every 3 seconds...')

    def send_random(self):
        msg = JointTrajectory()
        msg.joint_names = JOINTS

        point = JointTrajectoryPoint()
        # Random angles between 30 and 150 degrees (safe range)
        angles_deg = [random.uniform(30, 150) for _ in JOINTS]
        point.positions = [deg_to_rad(a) for a in angles_deg]
        point.time_from_start = Duration(sec=1, nanosec=0)

        msg.points = [point]
        self.pub.publish(msg)
        self.get_logger().info(f'Sent: {[f"{a:.0f}°" for a in angles_deg]}')

def main():
    rclpy.init()
    node = RandomMotion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
