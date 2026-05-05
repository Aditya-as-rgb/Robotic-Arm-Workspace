#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import random

JOINTS = ['Revolute 1', 'Revolute 2', 'Revolute 3',
          'Revolute 4', 'Revolute 5', 'Revolute 6']

TOTAL_LIMIT = 180.0  # max combined displacement from start (degrees)
START_ANGLE = 90.0   # all joints start at 90 degrees

def deg_to_rad(deg):
    return math.radians(deg - 90.0)

class RandomMotion(Node):
    def __init__(self):
        super().__init__('random_motion')
        self.pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        # Track current angle of each joint (degrees)
        self.current_angles = {j: START_ANGLE for j in JOINTS}
        self.reversing = False

        self.timer = self.create_timer(3.0, self.send_next)
        self.get_logger().info('Random motion started...')

    def total_displacement(self):
        return sum(abs(self.current_angles[j] - START_ANGLE) for j in JOINTS)

    def send_next(self):
        # Pick one random joint to move
        joint = random.choice(JOINTS)

        if self.reversing:
            # Move all displaced joints back toward 90
            displaced = [j for j in JOINTS if abs(self.current_angles[j] - START_ANGLE) > 1.0]
            if not displaced:
                self.reversing = False
                self.get_logger().info('All joints back to center, resuming random motion')
                return
            joint = max(displaced, key=lambda j: abs(self.current_angles[j] - START_ANGLE))
            # Move toward 90
            current = self.current_angles[joint]
            step = random.uniform(10, 30)
            new_angle = current + step if current < START_ANGLE else current - step
            new_angle = max(0.0, min(180.0, new_angle))
        else:
            # Random step between 10 and 40 degrees in either direction
            current = self.current_angles[joint]
            step = random.uniform(10, 40) * random.choice([-1, 1])
            new_angle = max(0.0, min(180.0, current + step))

            # Check if this would exceed total limit
            projected = self.total_displacement() \
                        - abs(current - START_ANGLE) \
                        + abs(new_angle - START_ANGLE)

            if projected >= TOTAL_LIMIT:
                self.reversing = True
                self.get_logger().info(
                    f'Total displacement {projected:.1f}° >= {TOTAL_LIMIT}°, reversing...'
                )
                return

        self.current_angles[joint] = new_angle

        # Build and publish trajectory with only the moving joint
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        point = JointTrajectoryPoint()
        point.positions = [deg_to_rad(self.current_angles[j]) for j in JOINTS]
        point.time_from_start = Duration(sec=1, nanosec=0)
        msg.points = [point]
        self.pub.publish(msg)

        self.get_logger().info(
            f'{"[REV] " if self.reversing else ""}Moving {joint} to '
            f'{new_angle:.1f}° | total displacement: {self.total_displacement():.1f}°'
        )

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
