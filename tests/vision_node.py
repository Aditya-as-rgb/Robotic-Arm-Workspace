#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from ultralytics import YOLO
import time

MODEL_PATH = '/home/aditya/Documents/Robotic Arm /best.pt'  # <-- replace with your model path
CONFIDENCE = 0.6
CONFIRM_SECONDS = 3.5  # wait this long before publishing classification

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.pub = self.create_publisher(String, '/box_classification', 10)

        self.model = YOLO(MODEL_PATH)
        self.cap = cv2.VideoCapture(1)  # USB webcam

        if not self.cap.isOpened():
            self.get_logger().error('Cannot open webcam!')
            return

        self.current_class = None
        self.class_start_time = None
        self.published = False  # don't re-publish until box changes

        self.timer = self.create_timer(0.1, self.process_frame)  # 10 FPS
        self.get_logger().info('Vision node started')

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = self.model(frame, verbose=False)
        detected_class = None

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE:
                    continue
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id].lower()
                if cls_name in ('x', 'defective'):
                    detected_class = 'defective'
                elif cls_name in ('o', 'non_defective'):
                    detected_class = 'non_defective'

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if detected_class == 'defective' else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{cls_name} {conf:.2f}',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

        # Confirmation logic
        if detected_class != self.current_class:
            self.current_class = detected_class
            self.class_start_time = time.time()
            self.published = False

        if (detected_class is not None
                and not self.published
                and time.time() - self.class_start_time >= CONFIRM_SECONDS):
            msg = String()
            msg.data = detected_class
            self.pub.publish(msg)
            self.published = True
            self.get_logger().info(f'Published: {detected_class}')

        # Show feed
        cv2.imshow('Vision', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main():
    rclpy.init()
    node = VisionNode()
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
