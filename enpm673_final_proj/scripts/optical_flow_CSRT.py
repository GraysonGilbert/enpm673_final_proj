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
    # self.declare_parameter('use_sim_time', True)
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
    self._mask_init = False

    # Define the stop command
    self._stop_cmd = Twist()
    self._stop_cmd.linear.x = 0.0
    self._stop_cmd.linear.z = 0.0
    
    # Define the go command
    self._go_cmd = Twist()
    self._go_cmd.linear.x = 0.1
    self._go_cmd.linear.z = 0.0
    
    # Parameters for window of frame to look for obstacle
    self._w_low = 0.4
    self._w_high = 0.6
    self._h_low = 0.2
    self._h_high = 0.8
        
    # Initialize region of interest (ROI) for obstacle detection
    self._roi = None
    
    # Initialize parameters for sampling environment baseline
    self._avg = 0
    self._avg_frames = []
    self.avg_frame_counter = 20
    
    self.get_logger().info('**** OPTICAL FLOW OBSTACLE DETECTION NODE INITIALIZED ****')
    
  # Subscription callback for performing optical flow detection and tracking
  def _OF_callback(self,msg):
    
    # Global variables
    global prev_gray, gray, contours, x, y, w, h, hFrame, wFrame, ret, center, mask
    
    #### GRAB FIRST FRAME TO INITIALIZE OPTICAL FLOW ####
    
    if self._first_frame is None:
      try:
        # Grab the first frame
        self._first_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._prev_frame = self._first_frame.copy()
        prev_gray = cv.cvtColor(self._prev_frame, cv.COLOR_BGR2GRAY)
        # cv.imshow('First Frame', cv.resize(self._first_frame, (640, 480)))
        # cv.waitKey(1000)
        
        # Begin turtlebot driving
        try:
          self.get_logger().info('**** TURTLEBOT DRIVING ****')
          # self._cmd_vel.publish(self._go_cmd)
        except Exception as e:
          self.get_logger().error(f"Error publishing go command: {e}")
          return
        
      except Exception as e:
        self.get_logger().error(f"Error grabbing image: {e}")
        return
      return

    else:
      
      # Grab the current frame
      try:
        self._frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv.cvtColor(self._frame, cv.COLOR_BGR2GRAY)
      except Exception as e:
        self.get_logger().error(f"Error grabbing image: {e}")
        return
      
      #### DRIVING USING DENSE OPTICAL FLOW ####
      
      if not self._detected:
        
        # Calculate dense optical flow
        hFrame, wFrame = prev_gray.shape
        gray_roi = gray[int(hFrame*self._h_low):int(hFrame*self._h_high), int(wFrame*self._w_low):int(wFrame*self._w_high)]
        prev_gray_roi = prev_gray[int(hFrame*self._h_low):int(hFrame*self._h_high), int(wFrame*self._w_low):int(wFrame*self._w_high)]
        
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
        # flow = cv.calcOpticalFlowFarneback(prev=prev_gray_roi, 
        #                                   next=gray_roi, 
        #                                   flow=None,
        #                                   pyr_scale=0.4, 
        #                                   levels=3, 
        #                                   winsize=11, 
        #                                   iterations=3, 
        #                                   poly_n=5, 
        #                                   poly_sigma=1.2, 
        #                                   flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN)
        
        # Compute magnitude and angle of flow vectors
        mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Optional: visualize optical flow
        # hsv = np.zeros_like(self._prev_frame)
        # hsv[..., 1] = 255
        # hsv[..., 0] = ang * 180 / np.pi / 2
        # hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        # flow_rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        # Grab average optical flow magnitude in frame
        motion_score = np.mean(mag)

        # Compute average motion score over inital frames to get environment baseline
        if len(self._avg_frames) < self.avg_frame_counter:
          # Continue driving (need?)
          try:
            self._cmd_vel.publish(self._stop_cmd)
          except Exception as e:
            self.get_logger().error(f"Error publishing go command: {e}")
            return
          self._avg_frames.append(motion_score)
          self._avg = np.mean(self._avg_frames)
          self.get_logger().info(f'Sampling environment - average motion score after {len(self._avg_frames)} frames: {self._avg:.3f}')
        
        else:
          self.get_logger().info(f'Optical flow motion score: {motion_score:.3f}')
          
        # else:
        #   self.get_logger().info('Turtlebot driving - Optical flow obstacle detection in progress ...')
        #   # Continue driving (need?)
        #   try:
        #     self._cmd_vel.publish(self._go_cmd)
        #   except Exception as e:
        #     self.get_logger().error(f"Error publishing go command: {e}")
        #     return

        threshold = self._avg*4.0 # Tune threshold as needed

        
        #### IF OBJECT DETECTED, STOP TURTLEBOT ####
        if motion_score > threshold:  # Tune threshold as needed
          self.get_logger().info('**** OBSTACLE DETECTED - STOPPING TURTLEBOT ****')
          
          try:
            self._cmd_vel.publish(self._stop_cmd)
            self._stopped = True
          except Exception as e:
            self.get_logger().error(f"Error publishing stop command: {e}")
            return
          
          self._detected = True
          
          # Threshold the magnitude to create a motion mask
          maskThreshold = threshold*2.0 # Tune threshold as needed
          _, mask = cv.threshold(mag, maskThreshold, 255, cv.THRESH_BINARY)
          
          # Convert mask to an 8-bit single-channel image
          # cv.imshow('mask', mask)
          # cv.waitKey(2000)
          mask = mask.astype(np.uint8)
          # cv.imshow('Mask', cv.resize(mask, (640, 480)))
          # cv.waitKey(4000)
          contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
          
          # Calculate bounding boxes for each contour
          try:
            Objcontour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(Objcontour)
            # self.get_logger().info(f'Obstacle detected at (x,y): {x}, {y}')
            # draw = self._frame.copy()
            # cv.rectangle(draw, (x + int(wFrame*0.45), y + int(hFrame*0.3)), (x + int(wFrame*0.45) + w, y + int(hFrame*0.3) + h), (0, 0, 255), 10)
            # cv.imshow('ROI', cv.resize(draw, (640, 480)))
            # cv.waitKey(4000)
            self._roi = (x + int(wFrame*self._w_low), y + int(hFrame*self._h_low), w, h)

            self.get_logger().info('**** INITIALIZED ROI FOR OBSTACLE TRACKING ****')
            ret = self._tracker.init(self._frame, self._roi)
            self.get_logger().info('**** INITIALIZED CSRT TRACKER ****')
            
          except ValueError:
            self.get_logger().error('No contours found in detected obstacle')
            return
          
          # ObjFrame = frame.copy()
          # cv.rectangle(ObjFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
          # cv.imshow('Optical Flow', ObjFrame)
          # cv.waitKey(2000)
          
        # cv.imshow('Dense Optical Fl
            

      #### TRACKING OBSTACLE USING LUCAS KANADE OPTICAL FLOW ####
      
      elif self._detected and self._stopped:
        
        success, box = self._tracker.update(self._frame)
        
        if success:
          self.get_logger().info('CSRT tracker updated - tracking obstacle ...')
          x,y,w,h = [int(v) for v in box]
          cv.rectangle(self._frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
          center = (int(x + w/2), int(y + h/2))
          cv.circle(self._frame, center, 25, (0, 0, 255), -1)
          cv.imshow('Optical Flow', cv.resize(self._frame, (640, 480)))
          cv.waitKey(1)
        else:
          self.get_logger().error('CSRT tracker failed to update')
          return
        
        if center[0] < self._frame.shape[1]*0.1 or center[0] > self._frame.shape[1]*0.9 or center[1] < self._frame.shape[0]*0.1:
          cv.destroyAllWindows()
          
          self.get_logger().info('**** OBSTACLE REMOVED FROM PATH - RESETTING ALGORITHM ****')
          
          self._detected = False
          self._stopped = False
          self._first_feature = None
          self._mask_init = False
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