#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class StopSignDetector(Node):
    def __init__(self):
        super().__init__('Stop_Sign_Detector')

        # subscribe to camera image
        self.subscription = self.create_subscription(Image,'/camera/image_raw',self.find_stop_sign,10)
        
        # publish to command velocity topic
        self.cmd_vel = self.create_publisher(Twist,'/cmd_vel',10)
        
        #load in cascade classifier initialization
        self.cascade_stop_sign = cv.CascadeClassifier('/home/ahall113/enpm673_final_proj/stop_sign_sample/stop_data.xml')

        #conver ros image to open cv type
        self.bridge = CvBridge()
    
    def find_stop_sign(self,msg):
        
        # convert ros image to open cv type
        convert_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # convert to gray imaage for feature detection
        frame_gray = cv.cvtColor(convert_image, cv.COLOR_BGR2GRAY)
        
        # Run Haar Cascade over image to find stop sign
        stop_sign_found = self.cascade_stop_sign.detectMultiScale(frame_gray, minSize=(10,10))
        
        # if stop sign is found draw box around detected stop sign and display 
        if len(stop_sign_found) > 0:
            
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.linear.z = 0.0
            self.cmd_vel.publish(stop_cmd)
            print("TURTLE BOT STOPPING")
            
            x = stop_sign_found[0][0]
            y = stop_sign_found[0][1]
            w = stop_sign_found[0][2]
            h = stop_sign_found[0][3]
            cv.rectangle(convert_image, (x,y), (x+w,y+h), (255,0,0),3)

            #Display Image with Stop sign
            # cv.imshow("frames",convert_image)
            # cv.waitKey(1)
        # else:
            # cv.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = StopSignDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
    
    
# frame_count = 4
# for frames in range(frame_count):
#     frame_path = f"/home/ahall113/enpm673_final_proj/stop_sign_sample/stop_sign_{frames}.jpg"
#     current_frame = cv.imread(frame_path)
#     if current_frame is None:
#         print(f"Failed to load {frame_path}")
#     else:
#         size_width = current_frame.shape[1]
#         size_height = current_frame.shape[0]
#         size = [size_width,size_height]
#         print("width", size_width)
#         print("height", size_height)
        
#         frame_rgb = cv.cvtColor(current_frame, cv.COLOR_BGR2RGB)
#         frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
#         cascade_stop_sign = cv.CascadeClassifier('/home/ahall113/enpm673_final_proj/stop_sign_sample/stop_data.xml')
#         stop_sign_found = cascade_stop_sign.detectMultiScale(frame_gray, minSize=(10,10))
#         print(stop_sign_found)
#         x = stop_sign_found[0][0]
#         y = stop_sign_found[0][1]
#         w = stop_sign_found[0][2]
#         h = stop_sign_found[0][3]
#         cv.rectangle(current_frame, (x,y), (x+w,y+h), (0,0,255),3)
#         cv.imshow("frames",current_frame)
#         cv.waitKey(0)
#         cv.destroyAllWindows()
# cascade_stop_sign = cv.CascadeClassifier('/home/ahall113/enpm673_final_proj/stop_sign_sample/stop_data.xml')
