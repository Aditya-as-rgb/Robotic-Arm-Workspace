from adafruit_servokit import ServoKit
from std_msgs.msg import String
import rclpy
from rclpy.node import Node
import time

kit = ServoKit(channels=16)

MG996R_PULSE = (500, 2500)
SG90_PULSE   = (500, 2400)

for ch in [1, 2, 3]:
    kit.servo[ch].set_pulse_width_range(*MG996R_PULSE)
for ch in [4, 5, 6]:
    kit.servo[ch].set_pulse_width_range(*SG90_PULSE)

# ── Positions (ch1, ch2, ch3, ch4, ch5, ch6) ─────────────────────────────────
HOME     = [110, 100, 160, 90, 90, 90]   # waiting position
RAISE    = [110,  60, 120, 90, 90, 90]   # lift arm before swinging
DEFECT   = [ 45,  60, 120, 90, 90, 90]   # sort defective
NON_DEF  = [135,  60, 120, 90, 90, 90]   # sort non-defective
# ─────────────────────────────────────────────────────────────────────────────

CHANNELS = [1, 2, 3, 4, 5, 6]

def move(ch, angle, delay=1.5):
    angle = max(0, min(180, angle))
    print(f"  ch{ch} → {angle}°")
    kit.servo[ch].angle = angle
    time.sleep(delay)

def move_to(position, delay=1.5):
    for ch, angle in zip(CHANNELS, position):
        move(ch, angle, delay)

class SortingNode(Node):
    def __init__(self):
        super().__init__('sorting_node')
        self.sub = self.create_subscription(
            String,
            '/box_classification',
            self.classification_callback,
            10
        )
        self.busy = False

        # Go home on startup
        print("Moving to home position...")
        move_to(HOME)
        self.get_logger().info('Sorting node ready — waiting for detections...')

    def classification_callback(self, msg):
        if self.busy:
            self.get_logger().warn('Still sorting, ignoring new detection')
            return

        classification = msg.data
        self.busy = True
        self.get_logger().info(f'Got: {classification}')

        try:
            # 1. Raise arm
            self.get_logger().info('Raising...')
            move_to(RAISE)

            # 2. Swing to correct bin
            if classification == 'defective':
                self.get_logger().info('Sorting DEFECTIVE...')
                move_to(DEFECT)
            else:
                self.get_logger().info('Sorting NON-DEFECTIVE...')
                move_to(NON_DEF)

            # 3. Pause at bin
            time.sleep(1.0)

            # 4. Raise before returning
            move_to(RAISE)

            # 5. Return home
            self.get_logger().info('Returning home...')
            move_to(HOME)

        except Exception as e:
            self.get_logger().error(f'Error: {e}')
            move_to(HOME)
        finally:
            self.busy = False
            self.get_logger().info('Ready for next box')

def main():
    rclpy.init()
    node = SortingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        move_to(HOME)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
