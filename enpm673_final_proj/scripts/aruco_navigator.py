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
        self.camera_offset_x = 0.076  # meters. From SDF file (camera is behind base_footprint)
        self.camera_offset_y = 0.0      # meters. From SDF file
 
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Marker Recovery (Back-up Markers and 'Look-around')
        self.search_mode = False
        self.search_direction = 1
        self.search_start_angle = 0.0
        self.search_timeout = 4.0
        self.last_marker_detection_time = time.time()
        self.search_angle_limit = math.radians(70)
        self.backup_markers = []
        self.current_marker_id = None
        self.sweep_done = False
        
        self.last_completed_marker_pos = None  # To store position of last completed marker
        self.max_marker_distance = 1.5  # Maximum allowed distance between consecutive markers
        self.search_mode_start_time = None  # For search mode timeout
        self.search_mode_timeout = 20.0  # 20 seconds max in search mode
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
            '/camera/camera_info', # Make sure topic name matches Gazebo output
            self.camera_info_callback,
            10) # QoS profile can be simple for camera_info

        # Initialize state variables
        self.last_time = self.get_clock().now().nanoseconds / 1e9
        self.previous_error_v = 0
        self.previous_error_w = 0
        self.goal_reached = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.timer = None
        self.route_completed = False
        # Initialize with provided camera calibration values
        # self.camera_matrix = np.array([
        #     [451.76715714, 0.0, 315.11564342],
        #     [0.0, 129.83982724, 266.57573962],
        #     [0.0, 0.0, 1.0]
        # ])
        # self.dist_coeffs = np.array([[-0.08727634, -0.01117336, -0.05632693, 0.03562648, 0.0262442]])
        self.camera_matrix = None
        self.dist_coeffs = None
        self.got_camera_info = False # Flag to wait for info

        # ArUco marker tracking variables
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.marker_detected = False
        self.marker_direction = 0.0
        self.marker_actual_x = 0.0
        self.marker_actual_y = 0.0
        self.marker_queue = []
        self.first_marker_seen = False
        # self.aruco_id = 0

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
        self.final_marker_direction = 0.0 # initialize the final drive direction.

        # self.get_logger().info('TurtleBot3 Controller initialized with pre-calibrated camera parameters')
        self.get_logger().info('TurtleBot3 Controller initialized, waiting for camera info...')

        # Add these class variables in the __init__ method:
        self.search_state = "IDLE"  # States: IDLE, TURNING_LEFT, TURNING_CENTER, TURNING_RIGHT, TURNING_BACK, MOVING_FORWARD, PAUSED
        self.search_original_angle = 0.0  # Original angle when search starts
        self.forward_drive_start_time = None  # Timer for forward movement
        self.forward_drive_duration = 2.0  # How long to drive forward in seconds
        self.get_logger().info(f'the camera calibration matrix is:{self.camera_matrix}')
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
            # Unsubscribe if you only need it once
            # self.destroy_subscription(self.camera_info_sub) 

    def enter_search_mode(self):
        """Centralized method to enter search mode."""
        if self.final_marker_reached:
            # Never enter search mode after the final marker is done
            self.search_mode = False
            return
        self.search_mode = True
        self.search_state = "TURNING_LEFT"  # Start by turning left
        self.search_original_angle = self.current_yaw  # Store original orientation
        self.search_start_angle = self.current_yaw    # Current reference point
        self.search_mode_start_time = time.time()  # Start timing search mode
        
        # Force a complete stop before starting search
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        
        self.sweep_done = False
        self.get_logger().info("Entering search mode - stopping and starting left turn")

    def process_aruco_detection(self, cv_image):
                
        if not self.got_camera_info:
            self.get_logger().warn("Waiting for camera info, skipping ArUco detection.")
            return 

        if self.final_marker_reached:
            # Ignore all further marker detections after the last marker is done
            return 
        
        try:
            # Detect ArUco markers
            corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
            if ids is not None and len(ids)>0:
                # Find indices where id == 0
                indices = np.where(ids.flatten() == self.marker_id)[0]
                # Filter corners and ids to only those with id == 0
                filtered_corners = [corners[i] for i in indices] # list of numpy arrays each matching detected marker with id=self.marker_id
                filtered_ids = ids[indices] # numpy array with ids=sel.marker_id=0
                if len(filtered_ids) > 0:
                    self.first_marker_seen = True
                    self.marker_detected = True
                    
                    # Draw all detected markers with id=0
                    cv2.aruco.drawDetectedMarkers(cv_image, filtered_corners, filtered_ids)
                    # Process all markers and store their positions
                    new_marker_positions = []
                    # ids_list = ids.flatten().tolist()
                    for i in range(len(filtered_ids)):

                        # if ids_list[i] != 0: # skip markers that don't have ID's of 0
                        #     self.get_logger().info(f"Ignoring marker with ID {ids_list[i]} (not 0)")
                        #     continue 

                        marker_corners = filtered_corners[i][0]
                        if self.camera_matrix is not None and self.dist_coeffs is not None:
                            self.get_logger().info(f'the camera calibration matrix is:{self.camera_matrix}')
                            # Estimate pose of the marker
                            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                                [marker_corners], self.marker_size, self.camera_matrix, self.dist_coeffs)
                            self.get_logger().info(f"Raw tvecs for marker ID {ids[i][0]}: {tvecs[0][0]}")
                            # Extract position information relative to camera-frame (opencv convention for this frame= x-right, y-down, z-forward)
                            marker_x = tvecs[0][0][0]
                            marker_z = tvecs[0][0][2]
                            
                            # Convert to robot's camera frame
                            # Convert from Camera axis orientation to ROS base-link axis orientation conventation (x-forward, y-left, z-up)
                            cam_to_robot_axes_dir_x = marker_z
                            cam_to_robot_axes_dir_y = -marker_x
                            # --- Compensate for camera offset in robot frame (adjust self.camera_offset_coordinate for turtlebot4) ---
                            # Marker position in robot frame = camera offset + marker position relative to camera
                            marker_in_robot_x = self.camera_offset_x + cam_to_robot_axes_dir_x
                            marker_in_robot_y = self.camera_offset_y + cam_to_robot_axes_dir_y
                            
                            # Convert to global frame (likely works best for turtlebot4- uncomment to use)
                            # marker_x_g = self.current_x + robot_x * math.cos(self.current_yaw) - robot_y * math.sin(self.current_yaw)
                            # marker_y_g = self.current_y + robot_x * math.sin(self.current_yaw) + robot_y * math.cos(self.current_yaw)
                            # Convert to global frame only for simulation with turtlebot3, uncomment above for turtlebot4 and comment-out below.
                            marker_x_g = self.current_x + marker_in_robot_x * math.cos(self.current_yaw) - marker_in_robot_y * math.sin(self.current_yaw)
                            marker_y_g = self.current_y + marker_in_robot_x * math.sin(self.current_yaw) + marker_in_robot_y * math.cos(self.current_yaw)
                        else:
                            self.get_logger().warn("Camera info not available for pose estimation.")

                        # Skip markers that are too far from the previous marker
                        # if self.last_completed_marker_pos is not None and self.markers_completed > 0:
                        #     dist_to_last = math.hypot(
                        #         marker_x_g - self.last_completed_marker_pos[0],
                        #         marker_y_g - self.last_completed_marker_pos[1]
                        #     )

                        #     if dist_to_last > self.max_marker_distance and self.startup_phase == False:
                        #         self.get_logger().info(f"Ignoring marker at ({marker_x_g:.2f}, {marker_y_g:.2f}) - too far from last marker: {dist_to_last:.2f}m")
                        #         continue   

                        # Draw axis for each marker
                        cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs,
                                        rvecs[0], tvecs[0], 0.03)
                        
                        # Extract position information (relative to camera)
                        marker_x = tvecs[0][0][0]
                        marker_z = tvecs[0][0][2]
                        
                        # Convert to robot camera frame
                        # Convert from Camera coord axis (x-right, y-down, z-forward) to ROS base-link axis orientation conventation (x-forward, y-left, z-up)
                        cam_to_robot_axes_dir_x = marker_z
                        cam_to_robot_axes_dir_y = -marker_x
                        # Convert from camera frame to robot (base-link) frame
                        marker_in_robot_x = self.camera_offset_x + cam_to_robot_axes_dir_x
                        marker_in_robot_y = self.camera_offset_y + cam_to_robot_axes_dir_y

                        # Calculate marker orientation
                        R_ca, _ = cv2.Rodrigues(rvecs[0])
                        yaw_cam = math.atan2(R_ca[1, 0], R_ca[0, 0])
                        marker_yaw_robot = yaw_cam - math.pi/2
                        
                        # Convert to global frame
                        marker_x_g = self.current_x + marker_in_robot_x * math.cos(self.current_yaw) - marker_in_robot_y * math.sin(self.current_yaw)
                        marker_y_g = self.current_y + marker_in_robot_x * math.sin(self.current_yaw) + marker_in_robot_y * math.cos(self.current_yaw)
                        
                        self.get_logger().info(
                            f"Marker global pos ({marker_x_g:.3f}, {marker_y_g:.3f}) | "
                            f"marker_yaw_robot: {math.degrees(marker_yaw_robot):+.1f}°")
                        
                        # Store this marker's information
                        new_marker_positions.append({
                            'id': filtered_ids[i][0],
                            'x_g': marker_x_g,
                            'y_g': marker_y_g,
                            'marker_x': cam_to_robot_axes_dir_x,
                            'marker_y': cam_to_robot_axes_dir_y,
                            'direction': marker_yaw_robot,
                            'distance': math.hypot(marker_in_robot_x, marker_in_robot_y),
                            'last_seen': time.time()
                        })
                    
                    # Replace queue with currently visible markers
                    self.marker_queue = new_marker_positions
                    self.marker_queue.sort(key=lambda m: m['distance'])
                    
                    # Limit queue size
                    if len(self.marker_queue) > 4:
                        self.marker_queue = self.marker_queue[:4]
                    
                    # Set the closest marker as current target
                    if self.marker_queue:
                        self.goal_x = self.marker_queue[0]['x_g']
                        self.goal_y = self.marker_queue[0]['y_g']
                        self.marker_direction = self.marker_queue[0]['direction']
                        self.current_marker_id = self.marker_queue[0]['id']
                        self.marker_actual_x = self.marker_queue[0]['marker_x']
                        self.marker_actual_y = self.marker_queue[0]['marker_y']
                        
                        # Apply filter to smooth current target
                        self.goal_x_filter.append(self.goal_x)
                        self.goal_y_filter.append(self.goal_y)
                        self.marker_direction_filter.append(self.marker_direction)
                        
                        # Keep filter size fixed
                        if len(self.goal_x_filter) > self.filter_size:
                            self.goal_x_filter.pop(0)
                            self.goal_y_filter.pop(0)
                            self.marker_direction_filter.pop(0)
                        
                        # Use filtered values
                        self.goal_x = sum(self.goal_x_filter) / len(self.goal_x_filter)
                        self.goal_y = sum(self.goal_y_filter) / len(self.goal_y_filter)
                        self.marker_direction = sum(self.marker_direction_filter) / len(self.marker_direction_filter)
                    
                    # Reset search timer since we found markers
                    self.last_marker_detection_time = time.time()
                    self.search_mode = False
                    
                    # Display info with global coordinates
                    cv2.putText(cv_image, f"Target: ID {self.current_marker_id}, Global Pos: ({self.goal_x:.2f}, {self.goal_y:.2f})", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if len(self.marker_queue) > 1:
                        next_markers = [m['id'] for m in self.marker_queue[1:]]
                        cv2.putText(cv_image, f"Next markers: {next_markers}", 
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    self.marker_detected = False
                    cv2.putText(cv_image, "Searching for ArUco markers", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Check if we need to start searching
                if time.time() - self.last_marker_detection_time > self.search_timeout:
                    if not self.search_mode and not self.final_marker_reached:
                        self.enter_search_mode()
                # Add logging for marker tracking status
                if self.marker_detected:
                    self.get_logger().debug(f"Tracking {len(self.marker_queue)} markers")
                elif self.first_marker_seen:
                    time_since_last = time.time() - self.last_marker_detection_time
                    self.get_logger().debug(f"Lost marker tracking for {time_since_last:.1f} seconds")
                
        except Exception as e:
            self.marker_detected = False
            self.get_logger().error(f"Error during ArUco detection: {str(e)}")

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
                            f"final_marker_reached={self.final_marker_reached}, " +
                            f"search_mode={self.search_mode}")
        if self.route_completed:
            # If we've completed the route, just stay stopped
            twist = Twist()
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
            self.search_mode = False
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
               
        # Check if we're at the marker
        if self.marker_queue:
            # print(f'the number of completed markers is: {self.markers_completed}')
            if not self.has_valid_markers():
                # Handle empty queue - enter search mode if we've seen markers before
                if self.first_marker_seen and not self.final_marker_reached:
                    if time.time() - self.last_marker_detection_time > self.search_timeout and not self.search_mode:
                        self.enter_search_mode()
                return            
            goal_x_g = self.marker_queue[0]['x_g']
            goal_y_g = self.marker_queue[0]['y_g']
            dx = goal_x_g - self.current_x
            dy = goal_y_g - self.current_y
            # at_goal = math.hypot(dx, dy) < 0.05
            # dist_err = math.hypot(dx, dy)
            dist_err = math.hypot(dx, dy)
            dist_err = max(0.0, dist_err)
            target_angle_global = math.atan2(dy, dx)
            ang_err = self._normalize_angle(target_angle_global - self.current_yaw)
            self.get_logger().info(f"[Debug] Approaching Marker {self.current_marker_id}: dist_err={dist_err:.3f}, ang_err={math.degrees(ang_err):.1f} deg, markers_completed={self.markers_completed}")
            at_goal = dist_err < 0.3
            # at_goal = dist_err < 0.6
            angle_ok = abs(ang_err) < math.radians(30)
            # self.get_logger().info(f"[Debug Thresholds] dist_err={dist_err:.3f} (Threshold < 0.6), ang_err={math.degrees(ang_err):.1f} (Threshold < 15 deg)") # Log check

            
            # final‑align & queue pop    
            # if at_goal and abs(ang_err) < math.radians(6):
            if at_goal and angle_ok:
                self.get_logger().info(f"Reached marker {self.current_marker_id}")
                # self.get_logger().info(f"[Debug] Inside 'at_goal' for Marker {self.current_marker_id}: Current markers_completed = {self.markers_completed}")
                self.markers_completed += 1
                print(f'the number of completed markers is: {self.markers_completed}')
                # Store the position of this completed marker
                self.last_completed_marker_pos = (goal_x_g, goal_y_g)
                last_marker_direction = self.marker_direction # Save direction before popping
                self.get_logger().info(f"[Debug] Inside 'at_goal' for Marker {self.current_marker_id}: Current markers_completed = {self.markers_completed}")
                # FINAL MARKER CHECK 
                if self.markers_completed >= 8:
                    self.get_logger().info("Final marker reached! Driving forward for 3 seconds...")
                    self.final_marker_reached = True
                    self.final_drive_start_time = time.time()
                    # Store the orientation FROM the last marker for final drive
                    self.final_marker_direction = self._normalize_angle(self.current_yaw + last_marker_direction)
                    # We might still need to pop the marker from the queue if it exists
                    if self.marker_queue and self.marker_queue[0]['id'] == self.current_marker_id: # Defensive check
                        self.marker_queue.pop(0)
                    return # Exit: Final marker logic takes over on next callback

                # === Now pop the marker (if not the final one) ===
                if self.marker_queue and self.marker_queue[0]['id'] == self.current_marker_id: # Defensive check
                    self.marker_queue.pop(0)
                
                # === Original logic for handling non-final markers or empty queue ===
                if self.marker_queue: # queue not empty after popping non-final marker
                    self.marker_direction = self.marker_queue[0]['direction']
                    self.last_marker_detection_time = time.time() # Update time since we see the next one
                    # let the main control loop run for the new target
                else: # Queue is empty AFTER popping a NON-FINAL marker
                    # This case implies we lost sight of subsequent markers OR it was the last one (handled above)
                    self.get_logger().info("Marker queue empty after reaching non-final goal - entering search mode")
                    # Re-check 'final_marker_reached' just in case, though it should be false here
                    if not self.final_marker_reached: 
                        self.enter_search_mode()
                    return # Exit after entering search mode

        else:
            at_goal = False
        
        if self.marker_detected:
            # Only exit search mode if we actually detect a marker
            self.search_mode = False
            self.last_marker_detection_time = time.time()
        
        # Handle search mode
        if self.search_mode:
            # Check if search has timed out
            if time.time() - self.search_mode_start_time > self.search_mode_timeout:
                self.get_logger().info("Search mode timed out, returning to normal operation")
                self.search_mode = False
                return
            if self.search_state == "TURNING_LEFT":
                # Calculate angle from original position
                search_angle_diff = self._normalize_angle(self.current_yaw - self.search_original_angle)
                
                if abs(search_angle_diff) > self.search_angle_limit:
                    # Reached maximum left turn, now turn back to center
                    self.search_state = "TURNING_CENTER"
                    self.get_logger().info("Reached max left turn, returning to center")
                else:
                    # Continue turning left
                    twist.linear.x = 0.0
                    twist.angular.z = 0.5  # Positive for left turn
                    self.cmd_pub.publish(twist)
                    
            elif self.search_state == "TURNING_CENTER":
                # Calculate angle from original position
                center_angle_diff = self._normalize_angle(self.current_yaw - self.search_original_angle)
                
                if abs(center_angle_diff) < math.radians(5):  # Within 5 degrees of original
                    # At center, now turn right
                    self.search_state = "TURNING_RIGHT"
                    self.get_logger().info("Returned to center, starting right turn")
                else:
                    # Continue returning to center
                    twist.linear.x = 0.0
                    twist.angular.z = -0.5 if center_angle_diff > 0 else 0.5  # Turn toward center
                    self.cmd_pub.publish(twist)
                    
            elif self.search_state == "TURNING_RIGHT":
                # Calculate angle from original position
                search_angle_diff = self._normalize_angle(self.current_yaw - self.search_original_angle)
                
                if abs(search_angle_diff) > self.search_angle_limit:
                    # Reached maximum right turn, now turn back to center
                    self.search_state = "TURNING_BACK"
                    self.get_logger().info("Reached max right turn, returning to center")
                else:
                    # Continue turning right
                    twist.linear.x = 0.0
                    twist.angular.z = -0.5  # Negative for right turn
                    self.cmd_pub.publish(twist)
                    
            elif self.search_state == "TURNING_BACK":
                # Calculate angle from original position
                center_angle_diff = self._normalize_angle(self.current_yaw - self.search_original_angle)
                
                if abs(center_angle_diff) < math.radians(5):  # Within 5 degrees of original
                    # At center, start moving forward
                    self.search_state = "MOVING_FORWARD"
                    self.forward_drive_start_time = time.time()
                    self.get_logger().info("Returned to center, moving forward for 2 seconds")
                else:
                    # Continue returning to center
                    twist.linear.x = 0.0
                    twist.angular.z = -0.5 if center_angle_diff < 0 else 0.5  # Turn toward center
                    self.cmd_pub.publish(twist)
                    
            elif self.search_state == "MOVING_FORWARD":
                elapsed = time.time() - self.forward_drive_start_time
                
                if elapsed < self.forward_drive_duration:
                    # Continue moving forward
                    twist.linear.x = 0.08
                    twist.angular.z = 0.0
                    self.cmd_pub.publish(twist)
                    self.get_logger().info(f"Moving forward: {elapsed:.1f}/{self.forward_drive_duration:.1f} seconds")
                else:
                    # Stop and prepare for next search cycle
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_pub.publish(twist)
                    self.search_state = "PAUSED"
                    self.create_pause_timer(1.0)  # Small pause before next cycle
                    self.get_logger().info("Forward movement complete, pausing before next search cycle")
                    
            elif self.search_state == "PAUSED":
                # Just wait until timer callback
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                
            # Always return after handling search mode to avoid other movement logic
            return
        
        # If no markers detected, move forward slowly to look
        if not self.marker_detected and (not hasattr(self, 'marker_queue') or not self.marker_queue):
            self.get_logger().info("NO ARUCO MARKERS DETECTED YET, MOVING SLOW TO LOOK")
            twist.linear.x = 0.05
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            
            if (not self.first_marker_seen):
                self.search_mode = False
                self.sweep_done = False
            else:
                if (time.time() - self.last_marker_detection_time > self.search_timeout and 
                        not self.search_mode and not self.final_marker_reached and not self.route_completed):
                    self.enter_search_mode()

            return
        
        # If no current marker detected but have stored markers
        if not self.marker_detected and hasattr(self, 'marker_queue') and self.marker_queue:
            self.goal_x = self.marker_queue[0]['x_g']
            self.goal_y = self.marker_queue[0]['y_g']
            self.marker_direction = self.marker_queue[0]['direction']
            self.current_marker_id = self.marker_queue[0]['id']
            self.marker_actual_x = self.marker_queue[0]['marker_x']
            self.marker_actual_y = self.marker_queue[0]['marker_y']
            
            self.get_logger().info(f"Using stored marker ID {self.current_marker_id} as target")
        
        # Handle case of empty marker queue, before computing error to goal,
        if not self.has_valid_markers():
            # Handle empty queue - enter search mode if we've seen markers before
            if self.first_marker_seen and not self.final_marker_reached:
                if time.time() - self.last_marker_detection_time > self.search_timeout and not self.search_mode:
                    self.enter_search_mode()
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
        # slow_factor = max(0.1, 1.0 - abs(ang_err))
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
    
    def after_pause_callback(self):
        self.timer.cancel()  # One-time use
        self.search_state = "TURNING_LEFT"  # Restart search cycle
        self.search_original_angle = self.current_yaw  # Reset original angle
        self.get_logger().info("Starting new search cycle")
    
    def create_pause_timer(self, duration):
        """Create a timer with proper cleanup of existing timers."""
        if self.timer:
            self.timer.cancel()
        self.timer = self.create_timer(duration, self.after_pause_callback)
    
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
