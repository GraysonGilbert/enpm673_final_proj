#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraSubscriberPublisher(Node):
    def __init__(self):
        super().__init__('camera_subscriber_publisher')
        self.bridge = CvBridge()
        
        # Subscribe to the camera topic
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw', 
            self.image_callback,
            10
        )
        
        # Publisher for the processed image
        self.image_pub = self.create_publisher(
            Image,
            '/camera/processed_image',  
            10
        )
    
    def image_callback(self, msg):
        # Convert the ROS image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Draw a line on the image
        start_point = (50, 50)  # Starting point of the line
        end_point = (200, 200)  # Ending point of the line
        color = (0, 255, 0)  # Line color (green in BGR format)
        thickness = 2  # Line thickness
        cv2.line(cv_image, start_point, end_point, color, thickness)
        
        # Convert the processed OpenCV image back to ROS Image message
        processed_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        
        # Publish the processed image on a new topic
        self.image_pub.publish(processed_msg)

        # Optionally, display the processed image in a window
        cv2.imshow("Processed Camera Feed", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriberPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
