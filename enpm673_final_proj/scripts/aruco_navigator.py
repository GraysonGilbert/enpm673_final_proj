#!/usr/bin/env python3
import signal
import sys
import numpy as np
np.float = float  # bandaid for issue with transforms3d.euler library
import math
import rclpy
import cv2
import time
import matplotlib.pyplot as plt
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from transforms3d.euler import quat2euler
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo 

class TB3ArUcoFollower(Node):

    def __init__(self):
        super().__init__('tb3_aruco_follower')
        print('TurtleBot3 Aruco Follower Node Started')

        # Declare parameters and defaults
        self.declare_parameter('kp_linear', 0.3)
        self.declare_parameter('kp_angular', 0.4) # good for now but could be increased to 0.5
        self.declare_parameter('ki_linear', 0.0)  # linear velocity integral gain
        self.declare_parameter('ki_angular', 0.0)  # Angular velocity integral gain
        self.declare_parameter('kd_linear', 0.0)
        self.declare_parameter('kd_angular', 0.0)
        self.declare_parameter('max_lin_vel', 0.12)
        self.declare_parameter('max_ang_vel', 0.5)
        self.declare_parameter('marker_size', 0.1)

        # Read parameters
        self.kp_v = self.get_parameter('kp_linear').value
        self.kp_w = self.get_parameter('kp_angular').value
        self.ki_v = self.get_parameter('ki_linear').value
        self.ki_w = self.get_parameter('ki_angular').value
        self.kd_v = self.get_parameter('kd_linear').value
        self.kd_w = self.get_parameter('kd_angular').value
        self.max_v = self.get_parameter('max_lin_vel').value
        self.max_w = self.get_parameter('max_ang_vel').value
        self.marker_size = self.get_parameter('marker_size').value
        self.marker_id = 0   # only one marker with ID = 0 used on the track.

        # Initialize integral error variables
        self.integral_error_v = 0.0
        self.integral_error_w = 0.0

        # Lists to store data to troubleshoot and tune pid gains
        self.dist_err_list = []
        self.ang_err_list = []
        self.time_list = []
        self.marker_tgt_glob_x_list = []
        self.marker_tgt_glob_y_list = []
        self.start_time = time.time()
        self.reached_marker_x_list = []
        self.reached_marker_y_list = []

        # Camera offset from base_link (in robot frame, meters)
        self.camera_offset_x = 0.076  # meters. From SDF file
        self.camera_offset_y = 0.0    # meters. From SDF file
 
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Marker tracking variables 
        self.last_marker_detection_time = time.time()
        self.max_marker_distance = 1.3  # Maximum distance for reliable marker detection
        self.current_marker_id = None
        
        # Variables for marker measurement averaging
        self.target_measurements = []  # Store consecutive measurements of current target
        self.measurements_needed = 3   # Number of measurements to average
        self.first_marker_reached = False  # Flag to know we've processed the first marker
        
        # Initialize CV bridge
        self.bridge = CvBridge()
 
        # Create a QoS profile for camera subscription
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publisher and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        # Subscribe to camera feed
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            camera_qos)
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10)

        # Initialize state variables
        self.last_time = self.get_clock().now().nanoseconds / 1e9
        self.previous_error_v = 0
        self.previous_error_w = 0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.route_completed = False
        
        # Camera calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        self.got_camera_info = False

        # ArUco marker tracking variables
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.marker_detected = False
        self.marker_direction = 0.0
        self.marker_actual_x = 0.0
        self.marker_actual_y = 0.0
        self.marker_queue = []
        self.first_marker_seen = False

        # Filter to smooth out ArUco detection
        self.goal_x_filter = []
        self.goal_y_filter = []
        self.marker_direction_filter = []
        self.filter_size = 3

        # Startup phase
        self.startup_phase = True
        self.startup_time = None
        self.startup_duration = 3.0

        self.markers_completed = 0  # Count of markers we've reached
        self.final_marker_reached = False  # Flag for when we've reached the last marker
        self.final_drive_start_time = None  # Timer for the final 3 second drive
        self.final_drive_duration = 3.0  # How long to drive after final marker
        self.final_marker_direction = 0.0  # Initialize the final drive direction

        self.get_logger().info('TurtleBot3 Controller initialized, waiting for camera info...')

    def has_valid_markers(self):
        """Check if we have valid markers in the queue."""
        return bool(self.marker_queue)  # Returns True if queue is not empty

    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_aruco_detection(cv_image)
            cv2.imshow("TurtleBot3 Camera View", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing camera image: {str(e)}")
    
    def camera_info_callback(self, msg: CameraInfo):
        """Store camera calibration parameters."""
        if not self.got_camera_info:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.got_camera_info = True
            self.get_logger().info('Received camera calibration info.')

    def process_aruco_detection(self, cv_image):
        if not self.got_camera_info:
            self.get_logger().warn("Waiting for camera info, skipping ArUco detection.")
            return

        if self.final_marker_reached:
            return

        try:
            gray_cv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray_cv_img, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None and len(ids) > 0:
                indices = np.where(ids.flatten() == self.marker_id)[0]
                filtered_corners = [corners[i] for i in indices]
                filtered_ids = ids[indices]

                if len(filtered_ids) > 0:
                    self.first_marker_seen = True
                    self.marker_detected = True
                    cv2.aruco.drawDetectedMarkers(cv_image, filtered_corners, filtered_ids)
                    detected_markers = []

                    for i in range(len(filtered_ids)):
                        marker_corners = filtered_corners[i][0]
                        if self.camera_matrix is not None and self.dist_coeffs is not None:
                            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                                [marker_corners], self.marker_size, self.camera_matrix, self.dist_coeffs)
                            marker_x = tvecs[0][0][0]
                            marker_z = tvecs[0][0][2]
                            marker_in_robot_x = self.camera_offset_x + marker_z
                            marker_in_robot_y = self.camera_offset_y - marker_x
                            marker_distance = math.hypot(marker_in_robot_x, marker_in_robot_y)
                            if marker_distance <= self.max_marker_distance or not self.first_marker_reached:
                                marker_x_g = self.current_x + marker_in_robot_x * math.cos(self.current_yaw) - marker_in_robot_y * math.sin(self.current_yaw)
                                marker_y_g = self.current_y + marker_in_robot_x * math.sin(self.current_yaw) + marker_in_robot_y * math.cos(self.current_yaw)
                                R_ca, _ = cv2.Rodrigues(rvecs[0])
                                yaw_cam = math.atan2(R_ca[1, 0], R_ca[0, 0])
                                marker_yaw_robot = yaw_cam - math.pi/2
                                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[0], tvecs[0], 0.03)
                                detected_markers.append({
                                    'id': filtered_ids[i][0],
                                    'x_g': marker_x_g,
                                    'y_g': marker_y_g,
                                    'marker_x': marker_in_robot_x,
                                    'marker_y': marker_in_robot_y,
                                    'direction': marker_yaw_robot,
                                    'distance': marker_distance,
                                    'last_seen': time.time()
                                })
                        else:
                            self.get_logger().warn("Camera info not available for pose estimation.")

                    # Sort markers by distance (closest first)
                    detected_markers.sort(key=lambda m: m['distance'])

                    # Take the closest marker for averaging
                    if detected_markers:
                        closest_marker = detected_markers[0]
                        self.target_measurements.append(closest_marker)
                        # Keep only the last N measurements
                        self.target_measurements = self.target_measurements[-self.measurements_needed:]
                        
                        # Average the measurements and set as current target
                        avg_marker = self._average_marker_measurements(self.target_measurements)
                        self.marker_queue = [avg_marker]
                        self._update_display_info(cv_image, avg_marker)

                    self.last_marker_detection_time = time.time()
                else:
                    self.marker_detected = False
                    cv2.putText(cv_image, "Searching for ArUco markers", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.marker_detected = False
                cv2.putText(cv_image, "Searching for ArUco markers", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        except Exception as e:
            self.marker_detected = False
            self.get_logger().error(f"Error during ArUco detection: {str(e)}")

    def _average_marker_measurements(self, measurements):
        """Average multiple measurements of the same marker."""
        if not measurements:
            return None
        
        avg_x_g = sum(m['x_g'] for m in measurements) / len(measurements)
        avg_y_g = sum(m['y_g'] for m in measurements) / len(measurements)
        avg_marker_x = sum(m['marker_x'] for m in measurements) / len(measurements)
        avg_marker_y = sum(m['marker_y'] for m in measurements) / len(measurements)
        avg_direction = sum(m['direction'] for m in measurements) / len(measurements)
        avg_distance = sum(m['distance'] for m in measurements) / len(measurements)
        
        return {
            'id': measurements[0]['id'],  # Use ID from first measurement
            'x_g': avg_x_g,
            'y_g': avg_y_g,
            'marker_x': avg_marker_x,
            'marker_y': avg_marker_y,
            'direction': avg_direction,
            'distance': avg_distance,
            'last_seen': time.time()
        }

    def _update_display_info(self, cv_image, marker):
        """Update display with marker information."""
        cv2.putText(cv_image, f"Target: ID {marker['id']}, Pos: ({marker['x_g']:.2f}, {marker['y_g']:.2f}), Dist: {marker['distance']:.2f}m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def odom_callback(self, msg: Odometry):
        # Extract current time and calculate delta-time
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = max(0.001, current_time - self.last_time)
        self.last_time = current_time
        
        # Extract current pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = quat2euler([q.w, q.x, q.y, q.z])
        
        self.current_x = x
        self.current_y = y
        self.current_yaw = yaw
        
        # Control logic for ArUco following
        self.follow_aruco_control(dt)

    def follow_aruco_control(self, dt):
        twist = Twist()
        
        # Initialize error values
        dist_err = 0.0
        ang_err = 0.0
        target_angle_global = 0.0
        
        # --- Calculate errors and collect data --- 
        if self.has_valid_markers():
            goal_x_g = self.marker_queue[0]['x_g']
            goal_y_g = self.marker_queue[0]['y_g']
            dx = goal_x_g - self.current_x
            dy = goal_y_g - self.current_y
            dist_err = math.hypot(dx, dy)
            target_angle_global = math.atan2(dy, dx)
            
            # Special case for alignment when close to marker
            if dist_err < 0.15:
                target_angle_global = self._normalize_angle(self.current_yaw + self.marker_queue[0]['direction'])
                self.get_logger().info(f"REACHED MARKER, ALIGNING WITH MARKER DIRECTION {math.degrees(target_angle_global):.1f}°")
                
            ang_err = self._normalize_angle(target_angle_global - self.current_yaw)
            
            # Data collection for plotting
            self.dist_err_list.append(dist_err)
            self.ang_err_list.append(abs(ang_err))
            self.time_list.append(time.time() - self.start_time)
            self.marker_tgt_glob_x_list.append(goal_x_g)
            self.marker_tgt_glob_y_list.append(goal_y_g)
        
        # --- Handle special cases ---
        if self.route_completed:
            # Handle route completed case
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return
            
        if self.startup_phase:
            # Handle startup phase
            if self.startup_time is None:
                self.startup_time = time.time()
                self.get_logger().info("Starting initial forward movement")
            if (time.time() - self.startup_time > self.startup_duration) or self.first_marker_seen:
                self.startup_phase = False
                self.get_logger().info("Exiting startup phase")
            else:
                twist.linear.x = 0.08
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return

        # Handle final marker case    
        if self.final_marker_reached:
            elapsed = time.time() - self.final_drive_start_time
            if elapsed < self.final_drive_duration:
                twist.linear.x = 0.1
                ang_err = self._normalize_angle(self.final_marker_direction - self.current_yaw)
                twist.angular.z = self.kp_w * ang_err
                self.cmd_pub.publish(twist)
                self.get_logger().info(f"Final drive: {elapsed:.1f}/{self.final_drive_duration:.1f} seconds")
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                self.route_completed = True
                self.get_logger().info("Final drive complete, stopping robot. Route completed!")
            return
            
        if not self.has_valid_markers():
            # Handle no markers case
            twist.linear.x = 0.05
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return
            
        # --- Goal detection logic ---
        at_goal = dist_err < 0.15 and abs(ang_err) < math.radians(10)  # Using the computed dist_err
        angle_ok = abs(ang_err) < math.radians(12)

        # Handle reaching goal marker
        if at_goal and angle_ok:
            # Reset integral errors when reaching a goal
            self.integral_error_v = 0.0
            self.integral_error_w = 0.0
            # updated data collection lists
            self.reached_marker_x_list.append(self.marker_queue[0]['x_g'])
            self.reached_marker_y_list.append(self.marker_queue[0]['y_g'])
            self.get_logger().info(f"Reached marker {self.current_marker_id}")
            self.markers_completed += 1
            print(f'The number of completed markers is: {self.markers_completed}')
            # Final drive after last marker is completed
            last_marker_direction = self.marker_queue[0]['direction']
            if self.markers_completed >= 8:
                self.get_logger().info("Final marker reached! Driving forward for 3 seconds...")
                self.final_marker_reached = True
                self.final_drive_start_time = time.time()
                self.final_marker_direction = self._normalize_angle(self.current_yaw + last_marker_direction)
                return
            self.marker_queue.pop(0)
            self.integral_error_v = 0.0  # Reset when switching targets
            self.integral_error_w = 0.0  # Reset when switching targets
            # Reset measurements for the next marker
            self.target_measurements = []
            self.last_marker_detection_time = time.time()
            return
        
        # --- PID Control logic (using the already computed values) ---
        # Linear Velocity control
        v_error = dist_err
        self.integral_error_v += v_error * dt  # Accumulate error over time
        self.integral_error_v = max(-2.0, min(2.0, self.integral_error_v)) # anti-windup for linear integral term
        v_derivative = (v_error - self.previous_error_v) / dt
        linear_cmd = self.kp_v * v_error + self.kd_v * v_derivative + self.ki_v * self.integral_error_v
        self.previous_error_v = v_error

        # Angular Velocity control  
        self.integral_error_w += ang_err * dt
        self.integral_error_w = max(-2.0, min(2.0, self.integral_error_w)) # anti-windup for angular integral term
        w_derivative = (ang_err - self.previous_error_w) / dt
        angular_cmd = self.kp_w * ang_err + self.kd_w * w_derivative + self.ki_w * self.integral_error_w
        self.previous_error_w = ang_err
        # slow down linear velocity when ang_err is large
        if abs(ang_err) > math.radians(20):
            slow_factor = max(0.05, 1.0 - (abs(ang_err)/math.pi)*1.5)  # More aggressive slowing
        else:
            slow_factor = max(0.1, 1.0 - abs(ang_err)/math.pi)
        # Clip Linear & angular velocity commands
        linear_cmd = min(self.max_v * slow_factor, linear_cmd)
        angular_cmd = max(-self.max_w, min(self.max_w, angular_cmd))
        # Publish linear & angular velocity commands
        twist.linear.x = linear_cmd
        twist.angular.z = angular_cmd
        self.cmd_pub.publish(twist)

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

# Signal handler for commands when node is ended
def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Creating plots before exit...')
    # Command robot to stop
    stop_twist = Twist()
    stop_twist.linear.x = 0.0
    stop_twist.angular.z = 0.0
    node.cmd_pub.publish(stop_twist)
    time.sleep(0.2)  # Give time for the message to be sent
    
    # Make sure we have data to plot
    if len(node.time_list) > 0:
        # Plot distance error and angular error over time
        # plt.figure()
        # plt.plot(node.time_list, node.dist_err_list, label='Distance Error (m)')
        # plt.plot(node.time_list, node.ang_err_list, label='Angular Error (rad)')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Error (m)')
        # plt.legend()
        # plt.title('Distance and Angular Error vs Time')
        # plt.grid()

        # Plot detected marker global positions
        plt.figure()
        plt.plot(node.marker_tgt_glob_x_list, node.marker_tgt_glob_y_list, 'o-')
        plt.xlabel('Marker Target Global X (m)')
        plt.ylabel('Marker Target Global Y (m)')
        plt.title('Marker Target Global Positions')
        plt.grid()
        plt.axis('equal')
        # # Plot reached marker blobal position
        # plt.figure()
        # plt.plot(node.reached_marker_x_list, node.reached_marker_y_list, 'ro-')
        # plt.xlabel('Reached Marker Global X (m)')
        # plt.ylabel('Reached Marker Global Y (m)')
        # plt.title('Markers Successfully Reached')
        # plt.grid()
        # plt.axis('equal')

        # plt.show(block=True)  # Make sure it blocks until closed
    else:
        print("No data collected for plotting")
    
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0)

# Main function 
def main(args=None):
    global node  # Need to make node global for the signal handler
    rclpy.init(args=args)
    node = TB3ArUcoFollower()
    
    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Signal handler will handle this
    finally:
        # Only run this if the signal handler didn't already run
        if node.context.ok():
            stop_twist = Twist()
            stop_twist.linear.x = 0.0
            stop_twist.angular.z = 0.0
            node.cmd_pub.publish(stop_twist)
            time.sleep(0.2)
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()