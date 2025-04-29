#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import copy
import gc
import math

class CameraSubscriberPublisher(Node):
    def __init__(self):
        super().__init__('camera_subscriber_publisher')
        self.bridge = CvBridge()

        self.checkerboard_dims = (7, 5)  # (columns, rows)
        
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

        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        
        # Find the checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_dims, None)

        # Log the result
        self.get_logger().info(f"findChessboardCorners returned: ret={ret}, number_of_corners={len(corners) if corners is not None else 0}")


        image_with_lines = cv_image.copy()
        """
        if ret:
            print(f"Checkerboard detected with {len(corners)} points!")

            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Reshape for easier indexing: shape (rows, columns, 2)
            corners = corners.reshape(self.checkerboard_dims[1], self.checkerboard_dims[0], 2)

            # Make a copy for drawing
            image_with_lines = cv_image.copy()

            # Intersections:

            # Store lines
            horizontal_lines = []
            vertical_lines = []


            # Pick first and last rows of checkerboard
            row1_pts = (corners[0, 0], corners[0, -1])        # First row: first and last corner
            row2_pts = (corners[-1, 0], corners[-1, -1])       # Last row: first and last corner

            # Compute the lines
            line1 = self.compute_line_equation(*row1_pts)
            line2 = self.compute_line_equation(*row2_pts)

            # Compute their intersection (homogeneous coordinates)
            vp = np.cross(line1, line2)
            vp = vp / vp[2]  # Normalize to (x, y, 1)

            # vp is now the **vanishing point** for horizontal lines
            print(f"Vanishing point (horizontal lines): {vp}")


            # Pick first and last columns of the checkerboard
            col1_pts = (corners[0, 0], corners[-1, 0])       # First column: top and bottom
            col2_pts = (corners[0, -1], corners[-1, -1])      # Last column: top and bottom

            # Compute the lines
            line3 = self.compute_line_equation(*col1_pts)
            line4 = self.compute_line_equation(*col2_pts)

            # Compute their intersection (homogeneous coordinates)
            vp_vertical = np.cross(line3, line4)
            vp_vertical = vp_vertical / vp_vertical[2]  # Normalize

            print(f"Vanishing point (vertical lines): {vp_vertical}")

                # --- Only 4 lines total ---

            # 2 horizontal lines (rows)
            rows_to_draw = [0, self.checkerboard_dims[1]-1]  # first and last row
            for row in rows_to_draw:
                pt1 = corners[row, 0]
                pt2 = corners[row, -1]
                self.draw_extended_line(image_with_lines, pt1, pt2, color=(0, 255, 0), thickness=5)  # green for rows

            # 2 vertical lines (columns)
            cols_to_draw = [0, self.checkerboard_dims[0]-1]  # first and last column
            for col in cols_to_draw:
                pt1 = corners[0, col]
                pt2 = corners[-1, col]
                self.draw_extended_line(image_with_lines, pt1, pt2, color=(255, 0, 0), thickness=5)  # blue for columns


            # Draw horizon line

            self.draw_horizontal_through_point(image_with_lines, vp_vertical)
        """

        # # Draw a line on the image
        # start_point = (50, 50)  # Starting point of the line
        # end_point = (200, 200)  # Ending point of the line
        # color = (0, 255, 0)  # Line color (green in BGR format)
        # thickness = 2  # Line thickness
        # cv2.line(cv_image, start_point, end_point, color, thickness)

        # Resize the image to, for example, 1280x720
        resized_image = cv2.resize(image_with_lines, (1280, 720), interpolation=cv2.INTER_AREA)
        
        # Convert the processed OpenCV image back to ROS Image message
        processed_msg = self.bridge.cv2_to_imgmsg(image_with_lines, encoding="bgr8")
        #processed_msg = self.bridge.cv2_to_imgmsg(gray, encoding="mono8")
        
        # Publish the processed image on a new topic
        self.image_pub.publish(processed_msg)

        # Optionally, display the processed image in a window
        cv2.namedWindow("Processed Camera Feed", cv2.WINDOW_NORMAL)
        cv2.imshow("Processed Camera Feed", resized_image)
        cv2.waitKey(1)


    def draw_extended_line(self, img, pt1, pt2, color=(0, 255, 0), thickness=2):
        h, w = img.shape[:2]
        a = pt2[1] - pt1[1]
        b = pt1[0] - pt2[0]
        c = pt2[0]*pt1[1] - pt1[0]*pt2[1]

        points = []
        if b != 0:
            y0 = int((-c - a*0) / b)
            yw = int((-c - a*(w-1)) / b)
            if 0 <= y0 < h:
                points.append((0, y0))
            if 0 <= yw < h:
                points.append((w-1, yw))
        if a != 0:
            x0 = int((-c - b*0) / a)
            xh = int((-c - b*(h-1)) / a)
            if 0 <= x0 < w:
                points.append((x0, 0))
            if 0 <= xh < w:
                points.append((xh, h-1))

        if len(points) >= 2:
            cv2.line(img, points[0], points[1], color, thickness)


    def compute_line_equation(self, p1, p2):
        """Given two points, return line coefficients a, b, c for ax + by + c = 0"""
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p1[0]*p2[1] - p2[0]*p1[1]
        return np.array([a, b, c])

    def draw_horizontal_through_point(self, image, vp, color=(255, 0, 0), thickness=5):
        """Draw a horizontal line passing through the given vanishing point."""
        h, w = image.shape[:2]

        # Get the y-coordinate from the vanishing point
        y = int(vp[1])

        # Draw a horizontal line across the whole width
        start_point = (0, y)
        end_point = (w, y)

        cv2.line(image, start_point, end_point, color, thickness)





def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriberPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
