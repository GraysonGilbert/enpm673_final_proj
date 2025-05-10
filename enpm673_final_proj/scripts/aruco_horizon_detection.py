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
    super().__init__('aruco_horizon_detection')
    # self.declare_parameter('use_sim_time', True)

    # Convert raw ROS image to OpenCV type
    self._bridge = CvBridge()

    # Subscribe to camera topic
    self._img_sub = self.create_subscription(
        Image,
        '/tb4_2/oakd/rgb/preview/image_raw',
        self.horizon_callback,
        5)
    

    # # Subscribe to camera topic (Turtlebot4)
    # self._img_sub = self.create_subscription(
    #     Image,
    #     '/tb4_1/oakd/rgb/preview/image_raw'
    #     self.horizon_callback,
    #     5)


    # Publisher for the processed image
    self.image_pub = self.create_publisher(
        Image,
        '/camera/processed_image',  
        10
    )
    
    #self.checkerboard_dims = (7,5) # Checkerboard Dimensions

    self.image_ready = True  # Flag to control image grabbing
    self.timer = self.create_timer(1.0, self.enable_next_image)

  def enable_next_image(self):
    self.image_ready = True
    self.get_logger().info("Ready to grab next image...")

  # Subscription callback for horizon line detection
  def horizon_callback(self,msg):

    if not self.image_ready:
        return

    self.image_ready = False  # Block until the next 5 seconds
    #self.get_logger().info("Image received and processed.")

    cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') # Converting ROS message to OpenCV image
    
    gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY) # Greyscale image

    # Detect and log checkerboard corners found
    #ret, corners = cv.findChessboardCorners(gray, self.checkerboard_dims, None)
    #self.get_logger().info(f"findChessboardCorners returned: ret={ret}, number_of_corners={len(corners) if corners is not None else 0}")

    #_____________________________________


    # ArUco detection setup
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(corners) > 0:
        aruco_corners = corners[0][0]  # shape: (4, 2)
        print("First marker ID:", ids[0][0])
        print("First marker corners:")

        for i, corner in enumerate(aruco_corners):
            print(f"Corner {i}: x = {corner[0]:.2f}, y = {corner[1]:.2f}")
            x, y = int(corner[0]), int(corner[1])
            cv.circle(cv_image, (x, y), 5, (0, 255, 0), -1)
            cv.putText(cv_image, str(i), (x + 5, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Example: draw a line between corners 0 and 2, and extend it
        pt0 = aruco_corners[0]
        pt1 = aruco_corners[1]
        pt2 = aruco_corners[2]
        pt3 = aruco_corners[3]

        #vertical points
        self.draw_extended_line(cv_image, pt0, pt1)
        self.draw_extended_line(cv_image, pt2, pt3)

        #horizontal points
        self.draw_extended_line(cv_image, pt1, pt2)
        self.draw_extended_line(cv_image, pt0, pt3)


        # Pick first and last columns of the checkerboard
        horz1_pts = (pt0, pt1)       # First column: top and bottom
        horz2_pts = (pt2, pt3)      # Last column: top and bottom

        # Compute the lines
        line1 = self.compute_line_equation(*horz1_pts)
        line2 = self.compute_line_equation(*horz2_pts)

        # Compute their intersection (homogeneous coordinates)
        vp_horz = np.cross(line1, line2)
        vp_horz = vp_horz / vp_horz[2]  # Normalize to (x, y, 1)

        # vp is now the **vanishing point** for horizontal lines
        print(f"Vanishing point (horizontal lines): {vp_horz}")

            # Pick first and last columns of the checkerboard

        vert1_pts = (pt0, pt1)       # First column: top and bottom
        vert2_pts = (pt2, pt3)      # Last column: top and bottom

        # Compute the lines
        line3 = self.compute_line_equation(*vert1_pts)
        line4 = self.compute_line_equation(*vert2_pts)


        # Compute their intersection (homogeneous coordinates)
        vp_vertical = np.cross(line3, line4)
        vp_vertical = vp_vertical / vp_vertical[2]  # Normalize

        print(f"Vanishing point (vertical lines): {vp_vertical}")

        print(line3)


        self.draw_horizontal_through_point(cv_image, vp_vertical)
        self.draw_line_through_points(cv_image, vp_vertical, vp_horz)

    #___________________________________

        
        resized_image = cv.resize(cv_image, (1280, 720), interpolation=cv.INTER_AREA) # Resize the image
        
        processed_msg = self._bridge.cv2_to_imgmsg(resized_image, encoding="rgb8") # Convert the processed OpenCV image back to ROS Image message
        
        self.image_pub.publish(processed_msg) # Publish the processed image on a new topic

        # Display the processed image in a window
        cv.namedWindow("Processed Camera Feed", cv.WINDOW_NORMAL)
        cv.imshow("Processed Camera Feed", cv_image)
        cv.imwrite("screenshots/horizon_line_output.jpg", resized_image)
        cv.waitKey(1)
        

  # Extend the parallel lines between checkerboard corners
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

  # Computes the line coefficients between two points
  def compute_line_equation(self, p1, p2):
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p1[0]*p2[1] - p2[0]*p1[1]
        return np.array([a, b, c])

  # Draw a horizontal line through a calculated vanishing point
  def draw_horizontal_through_point(self, image, vp, color=(255, 0, 0), thickness=5):
        _, w = image.shape[:2]

        # Get the y-coordinate from the vanishing point
        y = int(vp[1])

        # Draw a horizontal line across the whole width
        start_point = (0, y)
        end_point = (w, y)

        cv.line(image, start_point, end_point, color, thickness)

  def draw_line_through_points(self, image, vp1, vp2, color=(255, 0, 255), thickness=5):
        _, w = image.shape[:2]
        x1 = int(vp1[0])
        y1 = int(vp1[1])

        x2 = int(vp2[0])
        y2 = int(vp2[1])

        start_point = (x1, y1)
        end_point = (x2, y2)

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