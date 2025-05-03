#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class HorizonLine(Node):
  def __init__(self):
    super().__init__('optical_flow')
    # self.declare_parameter('use_sim_time', True)
    # Subscribe to camera topic
    self._img_sub = self.create_subscription(
        Image,
        '/camera/image_raw',
        self.horizon_callback,
        5)

    # Publisher for the processed image
    self.image_pub = self.create_publisher(
        Image,
        '/camera/processed_image',  
        10
    )
    
    # Convert raw ROS image to OpenCV type
    self._bridge = CvBridge()
    
    self.checkerboard_dims = (7,5)

    self.image_ready = True  # Flag to control image grabbing
    self.timer = self.create_timer(5.0, self.enable_next_image)

  def enable_next_image(self):
    self.image_ready = True
    self.get_logger().info("Ready to grab next image...")

  # Subscription callback for performing optical flow detection and tracking
  def horizon_callback(self,msg):

    if not self.image_ready:
        return

    self.image_ready = False  # Block until the next 5 seconds
    self.get_logger().info("Image received and processed.")

    cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
    gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7,5), None)
    self.get_logger().info(f"findChessboardCorners returned: ret={ret}, number_of_corners={len(corners) if corners is not None else 0}")

    if ret:
        print(f"Checkerboard detected with {len(corners)} points!")

        # Refine corner locations
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Reshape for easier indexing: shape (rows, columns, 2)
        corners = corners.reshape(self.checkerboard_dims[1], self.checkerboard_dims[0], 2)

        # Make a copy for drawing
        image_with_lines = cv_image.copy()


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
        
        # Resize the image
        resized_image = cv.resize(image_with_lines, (1280, 720), interpolation=cv.INTER_AREA)
        
        # Convert the processed OpenCV image back to ROS Image message
        processed_msg = self._bridge.cv2_to_imgmsg(resized_image, encoding="rgb8")
        
        # Publish the processed image on a new topic
        self.image_pub.publish(processed_msg)

        # Optionally, display the processed image in a window
        cv.namedWindow("Processed Camera Feed", cv.WINDOW_NORMAL)
        cv.imshow("Processed Camera Feed", image_with_lines)
        cv.imwrite("screenshots/horizon_line_output.jpg", resized_image)
        cv.waitKey(1)
        

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
            cv.line(img, points[0], points[1], color, thickness)


  def compute_line_equation(self, p1, p2):
        """Given two points, return line coefficients a, b, c for ax + by + c = 0"""
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p1[0]*p2[1] - p2[0]*p1[1]
        return np.array([a, b, c])

  def draw_horizontal_through_point(self, image, vp, color=(255, 0, 0), thickness=5):
        """Draw a horizontal line passing through the given vanishing point."""
        _, w = image.shape[:2]

        # Get the y-coordinate from the vanishing point
        y = int(vp[1])

        # Draw a horizontal line across the whole width
        start_point = (0, y)
        end_point = (w, y)

        cv.line(image, start_point, end_point, color, thickness)

    

# Spin node
def main(args=None):
    rclpy.init(args=args)
    node = HorizonLine()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Initialization error: {e}")
    finally:
        node.get_logger().error("After Ctrl-C")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': 
    main()