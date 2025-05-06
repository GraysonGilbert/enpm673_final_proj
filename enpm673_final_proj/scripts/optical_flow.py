#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class OpticalFlowTracker(Node):
  def __init__(self):
    super().__init__('optical_flow')

    # Subscribe to camera topic
    self._img_sub = self.create_subscription(Image, '/camera/image_raw', self._OF_callback, 5)
    
    # Publish to command velocity topic
    self._cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
    
    # Initialize tracker
    self._tracker = cv.TrackerCSRT_create()
    
    # Convert raw ROS image to OpenCV type
    self._bridge = CvBridge()
    
    # Frame holders
    self._first_frame = None
    self._first_feature = None
    self._prev_frame = None
    self._frame = None
    self._track_mask = None
    
    # Define parameters for conditional logging
    self._detected = False
    self._stopped = False

    # Define the stop command
    self._stop_cmd = Twist()
    self._stop_cmd.linear.x = 0.0
    self._stop_cmd.linear.z = 0.0
    
    # Define the go command
    self._go_cmd = Twist()
    self._go_cmd.linear.z = 0.0
    
    # Initialize region of interest (ROI) for obstacle detection
    self._roi = None
    
    # Initialize parameters for sampling environment baseline
    self._avg = 0
    self._avg_frames = []
    
    #############################
    #### TUNEABLE PARAMETERS ####
    #############################
    
    ### Set the linear speed of the turtlebot ###
    # Going to slow may result in bad optical flow readings
    # Going too fast may result in the floor getting picked up (baseline average and search window should stop this)
    self._go_cmd.linear.x = 0.15
    
    ### Parameters for window of frame to look for obstacle ###
    # Make smaller for looking at a smaller window, larger for wider area
    
    # Look between _w_low % and _w_high % of the frame width (left to right)
    self._w_low = 0.4
    self._w_high = 0.6
    # Look between _h_high % and _h_low % of the frame height (top to bottom)
    self._h_high = 0.2
    self._h_low = 0.8
    
    ### Number of frames to sample for environment baseline ###
    # Ideally this is large to ensure proper readings, but robot frame rate may
    # make waiting for a lot of frames unreasonable
    self._avg_frame_counter = 15
    
    ### Detection threshold  gain ###
    # Used to determine when an average motion score is considered an obstacle
    # Readings above this threshold multiplied by the average are considered obstacles
    self._detection_gain = 1.75
    
    ### Mask threshold gain ###
    # Once an obstacle is detected, this is multipled by the detection threshold value
    # to threshold the magnitude of the optical flow to create a mask (remove low flow areas)
    self._mask_gain = 2.0
    
    #############################
    
    self.get_logger().info('**** OPTICAL FLOW OBSTACLE DETECTION NODE INITIALIZED ****')
    
  # Subscription callback for performing optical flow detection and tracking
  def _OF_callback(self,msg):
    
    # Global variables
    global prev_gray, gray, contours, x, y, w, h, hFrame, wFrame, ret, center, mask
    
    #### GRAB FIRST FRAME TO INITIALIZE OPTICAL FLOW ####
    
    if self._first_frame is None:
      try:
        
        # Begin turtlebot driving
        try:
          self.get_logger().info('**** TURTLEBOT DRIVING ****')
          self._cmd_vel.publish(self._go_cmd)
        except Exception as e:
          self.get_logger().error(f"Error publishing go command: {e}")
          return
        
        # Grab the first frame and intitialize it as prev_gray
        self._first_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._prev_frame = self._first_frame.copy()
        prev_gray = cv.cvtColor(self._prev_frame, cv.COLOR_BGR2GRAY)
        
      except Exception as e:
        self.get_logger().error(f"Error grabbing image: {e}")
        return
      return

    else:
      
      #### GRAB CURRENT FRAME ####
      try:
        # If no obstacle is detected, continue driving
        if not self._detected:
          try:
            self._cmd_vel.publish(self._go_cmd)
          except Exception as e:
            self.get_logger().error(f"Error publishing go command: {e}")
            return
          
        # Grab new frame
        self._frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv.cvtColor(self._frame, cv.COLOR_BGR2GRAY)
      except Exception as e:
        self.get_logger().error(f"Error grabbing image: {e}")
        return
      
      #### DRIVING WITH DENSE OPTICAL FLOW DETECTION ####
      
      # If no obstacle detected ...
      if not self._detected:
        
        # Define reagion of interest to look for obstacle
        hFrame, wFrame = prev_gray.shape
        gray_roi = gray[int(hFrame*self._h_high):int(hFrame*self._h_low), int(wFrame*self._w_low):int(wFrame*self._w_high)]
        prev_gray_roi = prev_gray[int(hFrame*self._h_high):int(hFrame*self._h_low), int(wFrame*self._w_low):int(wFrame*self._w_high)]
        
        # Calculate dense optical flow in ROI using Farneback method
        flow = cv.calcOpticalFlowFarneback(prev=prev_gray_roi, 
                                          next=gray_roi, 
                                          flow=None,
                                          pyr_scale=0.3, 
                                          levels=3, 
                                          winsize=9, 
                                          iterations=3, 
                                          poly_n=5, 
                                          poly_sigma=1.2, 
                                          flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN)
        '''
        These are ideal settings, but computationally expensive
        If robot is doing well, try this ...
        
        flow = cv.calcOpticalFlowFarneback(prev=prev_gray_roi, 
                                          next=gray_roi, 
                                          flow=None,
                                          pyr_scale=0.4, 
                                          levels=3, 
                                          winsize=11, 
                                          iterations=3, 
                                          poly_n=5, 
                                          poly_sigma=1.2, 
                                          flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN)
        ''' 
        
        # Compute magnitude of flow vectors
        mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
        
        '''
        Optional - visualize optical flow for testing
        hsv = np.zeros_like(self._prev_frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        flow_rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        '''

        # Grab average optical flow magnitude in frame
        motion_score = np.mean(mag)

        #### SAMPLING ENVIRONMENT BASELINE FOR FIRST FEW FRAMES ONLY ####
        
        # Compute average motion score over inital frames to get environment baseline
        if len(self._avg_frames) < self._avg_frame_counter:
          # Drive turtlebot and sample environment
          try:
            self._cmd_vel.publish(self._go_cmd)
          except Exception as e:
            self.get_logger().error(f"Error publishing go command: {e}")
            return
          
          # Store motion score and calculate average
          self._avg_frames.append(motion_score)
          self._avg = np.mean(self._avg_frames)
          self.get_logger().info(f'Sampling environment - average motion score after {len(self._avg_frames)} frames: {self._avg:.3f}')

        # If enough frames have been sampled, stop sampling
        else:
          self.get_logger().info(f'Optical flow motion score: {motion_score:.3f}')
          
          # Drive turtlebot and continue calculating motion score
          try:
            self._cmd_vel.publish(self._go_cmd)
          except Exception as e:
            self.get_logger().error(f"Error publishing go command: {e}")
            return

        # Define threshold for detecting obstacles as a function of the average motion score
        threshold = self._avg*self._detection_gain # Tune threshold as needed in __init__

        #### IF OBJECT DETECTED, STOP TURTLEBOT ####
        
        # If motion score threshold is met and enough frames have been sampled (avoids erroneous detection) ...
        if motion_score > threshold and len(self._avg_frames) > int(self._avg_frame_counter * 0.67):  # Tune threshold as needed
          self.get_logger().info('**** OBSTACLE DETECTED - STOPPING TURTLEBOT ****')
          
          # Stop turtlebot
          try:
            self._cmd_vel.publish(self._stop_cmd)
            self._stopped = True
          except Exception as e:
            self.get_logger().error(f"Error publishing stop command: {e}")
            return
          
          # Define parameter as True so algorithm moves to tracking phase
          self._detected = True
          
          # Threshold the motion score again with a higher threshold to create a motion mask
          # This will remove low flow areas and help with contour detection
          maskThreshold = threshold*self._mask_gain # Tune threshold as needed above in __init__
          _, mask = cv.threshold(mag, maskThreshold, 255, cv.THRESH_BINARY)
          
          # Convert mask to an 8-bit single-channel image
          mask = mask.astype(np.uint8)
          
          # Find contours in the mask
          contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
          
          # Assume largest contour is obstacle and get it's bounding box
          try:
            Objcontour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(Objcontour)
            
            # Define the ROI to pass to the tracker
            self._roi = (x + int(wFrame*self._w_low), y + int(hFrame*self._h_high), w, h)

            self.get_logger().info('**** INITIALIZED ROI FOR OBSTACLE TRACKING ****')
            
            # Initialize the CSRT tracker with the first frame and ROI
            ret = self._tracker.init(self._frame, self._roi)
            self.get_logger().info('**** INITIALIZED CSRT TRACKER ****')
            
          except ValueError:
            self.get_logger().error('No contours found in detected obstacle')
            return                      

      #### TRACKING OBSTACLE USING CSRT TRACKER ####
      
      # If an obstacle has been detected and the turtlebot is stopped ...
      elif self._detected and self._stopped:
        
        # Update tracker with the current frame and get the new bounding box
        success, box = self._tracker.update(self._frame)
        
        if success:
          self.get_logger().info('CSRT tracker updated - tracking obstacle ...')
          x,y,w,h = [int(v) for v in box]
          
          # Draw box and center on frame and show image
          cv.rectangle(self._frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
          center = (int(x + w/2), int(y + h/2))
          cv.circle(self._frame, center, 25, (0, 0, 255), -1)
          cv.imshow('Optical Flow', cv.resize(self._frame, (640, 480)))
          cv.waitKey(1)
        else:
          self.get_logger().error('CSRT tracker failed to update')
          return
        
        # If the tracker has the object moving to the outter 10% of the frame, assume it is out of the way
        if center[0] < self._frame.shape[1]*0.1 or center[0] > self._frame.shape[1]*0.9 or center[1] < self._frame.shape[0]*0.1:
          
          # Destroy tracker window
          cv.destroyAllWindows()
          
          self.get_logger().info('**** OBSTACLE REMOVED FROM PATH - RESETTING ALGORITHM ****')
          
          # Reset parameters
          self._detected = False
          self._stopped = False
          self._first_feature = None
          self._roi = None
          self._avg_frames = []
          self._avg = 0
          self._tracker = None
          self._tracker = cv.TrackerCSRT_create()

    # Set current frame to be the previous frame for next iteration        
    self._prev_frame = self._frame


# Spin node
def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowTracker()
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