#!/usr/bin/env python3

import numpy as np
np.float = float  # bandaid for issue with transforms3d.euler library
import math
import rclpy
import cv2
import time
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
        self.declare_parameter('kp_linear', 0.4)
        self.declare_parameter('kp_angular', 0.15)
        self.declare_parameter('kd_linear', 0.0)
        self.declare_parameter('kd_angular', 0.0)
        self.declare_parameter('tolerance', 0.1)
        self.declare_parameter('max_lin_vel', 0.12)
        self.declare_parameter('max_ang_vel', 0.5)
        self.declare_parameter('marker_size', 0.1)

        # Read parameters
        self.kp_v = self.get_parameter('kp_linear').value
        self.kp_w = self.get_parameter('kp_angular').value
        self.kd_v = self.get_parameter('kd_linear').value
        self.kd_w = self.get_parameter('kd_angular').value
        self.tol = self.get_parameter('tolerance').value
        self.max_v = self.get_parameter('max_lin_vel').value
        self.max_w = self.get_parameter('max_ang_vel').value
        self.marker_size = self.get_parameter('marker_size').value
        self.marker_id = 0   # only one marker with ID = 0 used on the track.

        # Camera offset from base_link (in robot frame, meters)
        self.camera_offset_x = 0.076  # meters. From SDF file
        self.camera_offset_y = 0.0    # meters. From SDF file
 
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # New marker tracking variables
        self.last_marker_detection_time = time.time()
        self.max_marker_distance = 1.3  # Maximum distance for reliable marker detection
        self.current_marker_id = None
        
        # Variables for marker measurement averaging
        self.current_target_measurements = []  # Store consecutive measurements of current target
        self.next_target_measurements = []     # Store consecutive measurements of next target
        self.measurements_needed = 3           # Number of measurements to average
        self.first_marker_reached = False      # Flag to know we've processed the first marker
        
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
        
        # Next marker storage - stores the next marker after the current one
        self.next_marker = None

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
            # Ignore all further marker detections after the last marker is done
            return 
        
        try:
            # Detect ArUco markers
            gray_cv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_cv_img, self.aruco_dict, parameters=self.aruco_params)
            if ids is not None and len(ids) > 0:
                # Find indices where id == 0
                indices = np.where(ids.flatten() == self.marker_id)[0]
                # Filter corners and ids to only those with id == 0
                filtered_corners = [corners[i] for i in indices]
                filtered_ids = ids[indices]
                
                if len(filtered_ids) > 0:
                    self.first_marker_seen = True
                    self.marker_detected = True
                    
                    # Draw all detected markers with id=0
                    cv2.aruco.drawDetectedMarkers(cv_image, filtered_corners, filtered_ids)
                    
                    # Process all markers and store their positions
                    detected_markers = []
                    
                    for i in range(len(filtered_ids)):
                        marker_corners = filtered_corners[i][0]
                        
                        if self.camera_matrix is not None and self.dist_coeffs is not None:
                            # Estimate pose of the marker
                            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                                [marker_corners], self.marker_size, self.camera_matrix, self.dist_coeffs)
                            
                            # Extract position information relative to camera
                            marker_x = tvecs[0][0][0]  # x in camera frame (right)
                            marker_z = tvecs[0][0][2]  # z in camera frame (forward)
                            
                            # Convert to robot's coordinate frame
                            cam_to_robot_axes_dir_x = marker_z        # Camera z -> Robot x
                            cam_to_robot_axes_dir_y = -marker_x       # Camera -x -> Robot y
                            
                            # Apply camera offset
                            marker_in_robot_x = self.camera_offset_x + cam_to_robot_axes_dir_x
                            marker_in_robot_y = self.camera_offset_y + cam_to_robot_axes_dir_y
                            
                            # Calculate distance to marker in robot frame
                            marker_distance = math.hypot(marker_in_robot_x, marker_in_robot_y)
                            
                            # Only process markers within the reliable detection range
                            if marker_distance <= self.max_marker_distance or not self.first_marker_reached:
                                # Convert to global frame
                                marker_x_g = self.current_x + marker_in_robot_x * math.cos(self.current_yaw) - marker_in_robot_y * math.sin(self.current_yaw)
                                marker_y_g = self.current_y + marker_in_robot_x * math.sin(self.current_yaw) + marker_in_robot_y * math.cos(self.current_yaw)
                                
                                # Calculate marker orientation
                                R_ca, _ = cv2.Rodrigues(rvecs[0])
                                yaw_cam = math.atan2(R_ca[1, 0], R_ca[0, 0])
                                marker_yaw_robot = yaw_cam - math.pi/2
                                
                                # Draw axis for each marker
                                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs,
                                                rvecs[0], tvecs[0], 0.03)
                                
                                # Store this marker's information
                                detected_markers.append({
                                    'id': filtered_ids[i][0],
                                    'x_g': marker_x_g,
                                    'y_g': marker_y_g,
                                    'marker_x': cam_to_robot_axes_dir_x,
                                    'marker_y': cam_to_robot_axes_dir_y,
                                    'direction': marker_yaw_robot,
                                    'distance': marker_distance,
                                    'last_seen': time.time()
                                })
                                
                                self.get_logger().info(
                                    f"Marker {i} global pos ({marker_x_g:.3f}, {marker_y_g:.3f}) | "
                                    f"distance: {marker_distance:.3f}m | "
                                    f"marker_yaw_robot: {math.degrees(marker_yaw_robot):+.1f}°")
                            else:
                                self.get_logger().info(f"Ignoring marker at distance {marker_distance:.3f}m (exceeds max reliable distance)")
                        else:
                            self.get_logger().warn("Camera info not available for pose estimation.")
                    
                    # Sort markers by distance (closest first)
                    detected_markers.sort(key=lambda m: m['distance'])
                    
                    # Handle marker measurements according to our strategy
                    if detected_markers:
                        # Reset detection timer since we found markers
                        self.last_marker_detection_time = time.time()
                        
                        # BEFORE reaching the first marker: just track the closest marker
                        if not self.first_marker_reached:
                            # Add current closest marker to measurements
                            if detected_markers:
                                self.current_target_measurements.append(detected_markers[0])
                                # Keep only the most recent N measurements
                                if len(self.current_target_measurements) > self.measurements_needed:
                                    self.current_target_measurements.pop(0)
                                
                                # If we have enough measurements, compute average and set as target
                                if len(self.current_target_measurements) == self.measurements_needed:
                                    avg_marker = self._average_marker_measurements(self.current_target_measurements)
                                    self.marker_queue = [avg_marker]  # Set as our target
                                    
                                    # Display info with global coordinates
                                    self._update_display_info(cv_image, avg_marker)
                                else:
                                    self.get_logger().info(f"Collecting measurements for first marker: {len(self.current_target_measurements)}/{self.measurements_needed}")
                        
                        # AFTER reaching the first marker: track two markers
                        else:
                            # Always use the closest marker as the current target
                            if detected_markers:
                                # Add current closest marker to measurements
                                self.current_target_measurements.append(detected_markers[0])
                                # Keep only the most recent N measurements
                                if len(self.current_target_measurements) > self.measurements_needed:
                                    self.current_target_measurements.pop(0)
                                
                                # If we have enough measurements, compute average and set as target
                                if len(self.current_target_measurements) == self.measurements_needed:
                                    avg_current = self._average_marker_measurements(self.current_target_measurements)
                                    
                                    # Update the queue with the current target
                                    self.marker_queue = [avg_current]
                                    
                                    # If there's a second marker detected, process it as the next target
                                    if len(detected_markers) > 1:
                                        # Store measurements for the second closest marker
                                        self.next_target_measurements.append(detected_markers[1])
                                        # Keep only the most recent N measurements
                                        if len(self.next_target_measurements) > self.measurements_needed:
                                            self.next_target_measurements.pop(0)
                                        
                                        # If we have enough measurements, compute average and store as next target
                                        if len(self.next_target_measurements) == self.measurements_needed:
                                            avg_next = self._average_marker_measurements(self.next_target_measurements)
                                            self.next_marker = avg_next
                                            self.get_logger().info(f"Next marker updated: distance {avg_next['distance']:.3f}m")
                                    
                                    # Display info with global coordinates
                                    self._update_display_info(cv_image, avg_current)
                else:
                    self.marker_detected = False
                    cv2.putText(cv_image, "Searching for ArUco markers", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add logging for marker tracking status
                if self.marker_detected:
                    self.get_logger().debug(f"Tracking {len(self.marker_queue)} markers")
                elif self.first_marker_seen:
                    time_since_last = time.time() - self.last_marker_detection_time
                    self.get_logger().debug(f"Lost marker tracking for {time_since_last:.1f} seconds")
                
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
        cv2.putText(cv_image, f"Target: ID {marker['id']}, Pos: ({marker['x_g']:.2f}, {marker['y_g']:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.next_marker:
            cv2.putText(cv_image, f"Next marker: Dist {self.next_marker['distance']:.2f}m",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

        self.get_logger().debug(f"Robot state: markers_completed={self.markers_completed}, " +
                            f"final_marker_reached={self.final_marker_reached}")
        
        if self.route_completed:
            # If we've completed the route, just stay stopped
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return
        
        # Handle startup phase
        if self.startup_phase:
            if self.startup_time is None:
                self.startup_time = time.time()
                self.get_logger().info("Starting initial forward movement")
            
            if (time.time() - self.startup_time > self.startup_duration) or self.marker_detected:
                self.startup_phase = False
                self.get_logger().info("Exiting startup phase")
            else:
                twist.linear.x = 0.08
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return
        
        # Handle final drive after reaching the last marker
        if self.final_marker_reached:
            elapsed = time.time() - self.final_drive_start_time
            if elapsed < self.final_drive_duration:
                # Continue driving forward in the exact direction when last marker was reached
                twist.linear.x = 0.1
                # Calculate angular error to maintain final orientation
                ang_err = self._normalize_angle(self.final_marker_direction - self.current_yaw)
                # Small correction to keep straight
                twist.angular.z = self.kp_w * ang_err
                self.cmd_pub.publish(twist)
                self.get_logger().info(f"Final drive: {elapsed:.1f}/{self.final_drive_duration:.1f} seconds")
            else:
                # Stop after the duration
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                self.route_completed = True  # Mark the entire route as completed
                self.get_logger().info("Final drive complete, stopping robot. Route completed!")
            return
               
        # Check if we're approaching a marker
        if self.has_valid_markers():
            goal_x_g = self.marker_queue[0]['x_g']
            goal_y_g = self.marker_queue[0]['y_g']
            dx = goal_x_g - self.current_x
            dy = goal_y_g - self.current_y
            
            dist_err = math.hypot(dx, dy)
            dist_err = max(0.0, dist_err)
            target_angle_global = math.atan2(dy, dx)
            ang_err = self._normalize_angle(target_angle_global - self.current_yaw)
            
            self.get_logger().info(f"[Debug] Approaching Marker {self.current_marker_id}: dist_err={dist_err:.3f}, ang_err={math.degrees(ang_err):.1f} deg, markers_completed={self.markers_completed}")
            at_goal = dist_err < 0.25
            angle_ok = abs(ang_err) < math.radians(12)
            
            # If we've reached the marker
            if at_goal and angle_ok:
                self.get_logger().info(f"Reached marker {self.current_marker_id}")
                self.markers_completed += 1
                print(f'The number of completed markers is: {self.markers_completed}')
                
                # Record that we've passed the first marker
                if self.markers_completed == 1:
                    self.first_marker_reached = True
                    self.get_logger().info("First marker reached - now tracking two markers")
                
                # Save direction before popping
                last_marker_direction = self.marker_direction 
                
                # FINAL MARKER CHECK 
                if self.markers_completed >= 8:
                    self.get_logger().info("Final marker reached! Driving forward for 3 seconds...")
                    self.final_marker_reached = True
                    self.final_drive_start_time = time.time()
                    # Store the orientation for final drive
                    self.final_marker_direction = self._normalize_angle(self.current_yaw + last_marker_direction)
                    return # Exit: Final marker logic takes over on next callback

                # When we've reached the current marker, make the next marker (if we have one) our current target
                self.marker_queue.pop(0)  # Remove the current marker
                
                # If we have a next marker stored, make it the current target
                if self.next_marker:
                    self.marker_queue.append(self.next_marker)
                    self.next_marker = None  # Clear the next marker storage
                    self.current_target_measurements = []  # Reset measurements for new target
                    self.next_target_measurements = []     # Reset measurements for new next target
                
                self.last_marker_detection_time = time.time()
                    
        # If no markers detected and never seen a marker, move forward slowly
        if not self.marker_detected and not self.first_marker_seen:
            self.get_logger().info("NO ARUCO MARKERS DETECTED YET, MOVING SLOW TO LOOK")
            twist.linear.x = 0.05
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return
            
        # If no current marker detected but have stored markers
        if not self.marker_detected and self.has_valid_markers():
            self.goal_x = self.marker_queue[0]['x_g']
            self.goal_y = self.marker_queue[0]['y_g']
            self.marker_direction = self.marker_queue[0]['direction']
            self.current_marker_id = self.marker_queue[0]['id']
            self.marker_actual_x = self.marker_queue[0]['marker_x']
            self.marker_actual_y = self.marker_queue[0]['marker_y']
            
            self.get_logger().info(f"Using stored marker ID {self.current_marker_id} as target")
        
        # If we have no valid markers, drive slowly forward
        if not self.has_valid_markers():
            twist.linear.x = 0.05
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return
            
        # Compute error to goal
        goal_x_g = self.marker_queue[0]['x_g']
        goal_y_g = self.marker_queue[0]['y_g']
        dx = goal_x_g - self.current_x
        dy = goal_y_g - self.current_y
        dist_err = math.hypot(dx, dy)
        
        # Calculate target angle
        target_angle_global = math.atan2(dy, dx)
        
        # When close to marker, align with marker orientation
        if dist_err < 0.15:
            target_angle_global = self._normalize_angle(self.current_yaw + self.marker_direction)
            self.get_logger().info(f"REACHED MARKER, ALIGNING WITH MARKER DIRECTION {math.degrees(target_angle_global):.1f}°")
        
        # Angular error
        ang_err = self._normalize_angle(target_angle_global - self.current_yaw)
        
        # PD Control on linear velocity
        v_error = dist_err
        v_derivative = (v_error - self.previous_error_v) / dt
        linear_cmd = self.kp_v * v_error + self.kd_v * v_derivative
        self.previous_error_v = v_error
        
        # PD Control on angular velocity
        w_derivative = (ang_err - self.previous_error_w) / dt
        angular_cmd = self.kp_w * ang_err + self.kd_w * w_derivative
        self.previous_error_w = ang_err
        
        # Speed limiting based on angular error
        slow_factor = max(0.1, 1.0 - abs(ang_err)/math.pi)
        linear_cmd = min(self.max_v * slow_factor, linear_cmd)
        angular_cmd = max(-self.max_w, min(self.max_w, angular_cmd))
        
        # Set twist commands
        twist.linear.x = linear_cmd
        twist.angular.z = angular_cmd
        
        # Publish command
        self.cmd_pub.publish(twist)

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    
def main(args=None):
    rclpy.init(args=args)
    node = TB3ArUcoFollower()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
