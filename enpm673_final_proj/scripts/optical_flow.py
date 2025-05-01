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
    self._img_sub = self.create_subscription(Image, '/camera/image_raw', self._OF_callback, 10)
    
    # Publish to command velocity topic
    self._cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
    
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
    
    # Initialize center of Lucas-Kanade optical flow tracker
    self._sum_LK_x = 0
    self._sum_LK_y = 0
    self._avg_LK_x = 0
    self._avg_LK_y = 0
    self._LK_center = [int(self._avg_LK_x), int(self._avg_LK_y)]
    
    # Define the stop command
    self._stop_cmd = Twist()
    self._stop_cmd.linear.x = 0.0
    self._stop_cmd.linear.z = 0.0
    
    # Define the go command
    self._go_cmd = Twist()
    self._go_cmd.linear.x = 0.1
    self._go_cmd.linear.z = 0.0
    
    # Initialize region of interest (ROI) for obstacle detection
    self._roi = None
    
    # Initialize list to store first frames for sampling environment baseline
    self._avg = 0
    self._avg_frames = []
    
    self.get_logger().info('*'*50)
    self.get_logger().info('Optical Flow Obstacle Detection Node Initialized')
    self.get_logger().info('*'*50)
    
  # Subscription callback for performing optical flow detection and tracking
  def _OF_callback(self,msg):
    
    #### GRAB FIRST FRAME TO INITIALIZE OPTICAL FLOW ####
    
    if self._first_frame is None:
      try:
        # Grab the first frame
        self._first_frame = self._bridge.imgmsg_to_cv(msg, desired_encoding='bgr8')
        self._prev_frame = self._first_frame.copy()
        prev_gray = cv.cvtColor(self._prev_frame, cv.COLOR_BGR2GRAY)
        
        # Initialize the mask for drawing LK optical flow
        self._track_mask = np.zeros_like(self._first_frame)
        
        # Begin turtlebot driving
        try:
          self.get_logger().info('*'*50)
          self.get_logger().info('\nTurtlebot Beginning to Drive')
          self.get_logger().info('*'*50)
          self._cmd_vel.publish(self._go_cmd)
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
        self._frame = self._bridge.imgmsg_to_cv(msg, desired_encoding='bgr8')
        gray = cv.cvtColor(self._frame, cv.COLOR_BGR2GRAY)
      except Exception as e:
        self.get_logger().error(f"Error grabbing image: {e}")
        return
      
      #### DRIVING USING DENSE OPTICAL FLOW ####
      
      if not self._detected:
        
        # Continue driving (need?)
        try:
          self._cmd_vel.publish(self._go_cmd)
        except Exception as e:
          self.get_logger().error(f"Error publishing go command: {e}")
          return
        
        self.get_logger().info('\nTurtlebot driving - Optical flow obstacle detection in progress ...')
        
        # Calculate dense optical flow
        flow = cv.calcOpticalFlowFarneback(prev=prev_gray, 
                                          next=gray, 
                                          flow=None,
                                          pyr_scale=0.5, 
                                          levels=3, 
                                          winsize=15, 
                                          iterations=3, 
                                          poly_n=5, 
                                          poly_sigma=1.2, 
                                          flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN)
        
        # Compute magnitude and angle of flow vectors
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Optional: visualize optical flow
        hsv = np.zeros_like(self._prev_frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        flow_rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        # Obstacle detection: check for high motion in bottom center
        h, w = mag.shape
        roi = mag[int(h*0.25):, int(w*0.45):int(w*0.55)]
        motion_score = np.mean(roi)

        # Compute average motion score over inital frames to get environment baseline
        if len(self._avg_frames) < 30:
          self._avg_frames.append(motion_score)
          self._avg = np.mean(self._avg_frames)

        threshold = self._avg*2.15
  
        if motion_score > threshold:  # Tune threshold as needed
          self._detected = True
          cv.putText(flow_rgb, 'Obstacle Detected!', (50, 50),
                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
          
          # Threshold the magnitude to create a motion mask
          maskThreshold = threshold*2.0
          _, mask = cv.threshold(mag, maskThreshold, 255, cv.THRESH_BINARY)
          
          # Convert mask to an 8-bit single-channel image
          # cv.imshow('mask', mask)
          # cv.waitKey(2000)
          mask = mask.astype(np.uint8)
          contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
          
          # Calculate bounding boxes for each contour
          try:
            Objcontour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(Objcontour)
            self._roi = (x, y, w, h)
            self.get_logger().info('\nInitlaized ROI for obstacle detection')
          except ValueError:
            self.get_logger().error('No contours found in detected obstacle')
            return
          
          # ObjFrame = frame.copy()
          # cv.rectangle(ObjFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
          # cv.imshow('Optical Flow', ObjFrame)
          # cv.waitKey(2000)
          
        cv.imshow('Dense Optical Flow', cv.resize(flow_rgb, (int(self._frame.shape[1] / 2), int(self._frame.shape[0] / 2))))
            

      #### TRACKING OBSTACLE USING LUCAS KANADE OPTICAL FLOW ####
      
      elif self._detected:
        
        if not self._stopped:
          self.get_logger().info('*'*50)
          self.get_logger().info('Obstacle detected! Stopping turtlebot ...')
          self.get_logger().info('*'*50)
          
          try:
            self._cmd_vel.publish(self._stop_cmd)
          except Exception as e:
            self.get_logger().error(f"Error publishing stop command: {e}")
            return
        
        elif self._stopped:
          mask = np.zeros_like(prev_gray)
          mask[y:y+h, x:x+w] = 255
          
          if self._first_feature is None:
            # Select good starting features to track in the ROI using Shi-Tomasi corner detection
            self._first_feature = cv.goodFeaturesToTrack(gray, 
                                                         mask=mask, 
                                                         maxCorners=100, 
                                                         qualityLevel=0.3, 
                                                         minDistance=7, 
                                                         blockSize=7)
            p0 = self._first_feature
            
            # Get center of detected features and plot
            for i in range(len(p0)):
              x, y = p0[i].ravel()
              self._sum_LK_x += x
              self._sum_LK_y += y
              
            self._avg_LK_x = self._sum_LK_x / len(p0)
            self._avg_LK_y = self._sum_LK_y / len(p0)  
            
            cv.circle(self._track_mask, (self._LK_center[0], self._LK_center[1]), 5, (0, 0, 255), -1)
            track = cv.add(self._frame, self._track_mask)
            cv.imshow('Optical Flow', cv.resize(track, (int(self._frame.shape[1] / 2), int(self._frame.shape[0] / 2))))
                   
          elif self._first_feature is not None:
            try:
              # Calculate optical flow using Lucas-Kanade method
              p1, st, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
            except Exception as e:
              self.get_logger().error(f"Error grabbing good features in p0: {e}")
              
            # Select good points
            good = p1[st == 1]
            
            # Get center of detected features and plot
            for i in range(len(good)):
              x, y = good[i].ravel()
              self._sum_LK_x += x
              self._sum_LK_y += y
              
            self._avg_LK_x = self._sum_LK_x / len(p0)
            self._avg_LK_y = self._sum_LK_y / len(p0)
            
            # self.get_logger().info(f'Obstacle center (x,y): {self._avg_LK_x}, {self._avg_LK_y}')
            
            cv.circle(self._track_mask, (self._LK_center[0], self._LK_center[1]), 5, (0, 0, 255), -1)
            track = cv.add(self._frame, self._track_mask)
            cv.imshow('Optical Flow', cv.resize(track, (int(self._frame.shape[1] / 2), int(self._frame.shape[0] / 2))))
            
            # Set current frame points to be the previous frame points for nexr iteration
            p0 = good.reshape(-1, 1, 2)
            
            if self._LK_center[0] < self._frame.shape[1]*0.1 or self._LK_center[0] > self._frame.shape[1]*0.9 or self._LK_center[1] < self._frame.shape[0]*0.1:
              self.get_logger().info('*'*50)
              self.get_logger().info('Obstacle Removed From Path - Turtlebot Resuming')
              self.get_logger().info('*'*50)
              
              self._detected = False
              self._stopped = False
              self._first_feature = None
              self._sum_LK_x = 0
              self._sum_LK_y = 0
              self._avg_LK_x = 0
              self._avg_LK_y = 0
              
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