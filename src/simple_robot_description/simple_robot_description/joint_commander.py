joint_commader.py file

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

# Line following parameters
TURN_LEFT_SPD = 0.15
TURN_RIGHT_SPD = 0.125
STRAIGHT_SPD = 0.25
ANGULAR_VEL = 0.4
OFFSET_Y = 500
TIMER_PERIOD = 0.05  # Faster response time

# Object detection parameters
OBJECT_MIN_AREA = 1500       # Reduced to detect objects sooner
SCAN_HEIGHT_START = 100      # Focus higher in the image to detect earlier
SCAN_HEIGHT_END = 400        
OBJECT_INTENSITY_THRESHOLD = 100  
STOP_DISTANCE_PX = 200       # Distance in pixels when we should stop

prev = (400, 400)
img_grayscale = None
img_color = None
object_detected_flag = False  # Persistent flag for object detection

def image_callback(msg):
    global img_grayscale, img_color
    img_color = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    img_grayscale = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

def detect_object(image):
    """
    Improved object detection with color filtering for red objects
    and better distance estimation
    """
    global object_detected_flag
    
    if image is None:
        return False
        
    height, width = image.shape[:2]
    
    # Focus on upper part of image to detect objects earlier
    roi_start = height - SCAN_HEIGHT_END
    roi_end = height - SCAN_HEIGHT_START
    roi = image[roi_start:roi_end, :]
    
    # Convert to HSV color space for better color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define range for red color (adjust these values as needed)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contour is large enough and close enough
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > OBJECT_MIN_AREA:
            # Get bounding box and check distance
            x, y, w, h = cv2.boundingRect(cnt)
            object_center = x + w//2
            object_bottom = y + h
            
            # If object is large and near the bottom of ROI, it's close
            if object_bottom > (roi_end - roi_start) - STOP_DISTANCE_PX:
                object_detected_flag = True
                return True
    
    # If no objects found, reset the flag
    object_detected_flag = False
    return False

def callback():
    global prev, img_grayscale, img_color, object_detected_flag

    try:
        if img_grayscale is None or img_color is None:
            return
            
        # Check for objects
        detect_object(img_color)
        
        if object_detected_flag:
            print("OBJECT DETECTED! STOPPING!")
            # Stop the robot completely
            move = Twist()
            move.linear.x = 0.0
            move.angular.z = 0.0
            pub.publish(move)
            return
            
        # Only proceed with line following if no object is detected
        _, img_bin = cv2.threshold(img_grayscale, 128, 1, cv2.THRESH_BINARY_INV)

        coords_bin = center_of_mass(img_bin[-300:])
        y = coords_bin[0] + OFFSET_Y
        x = coords_bin[1]

        if np.isnan(x) or np.isnan(y):
            x = prev[0]
            y = prev[1]
        else:
            prev = (x, y)
    
        print(f"Line position: ({x:.1f},{y:.1f})")

        move = Twist()
        if x < 350:
            move.linear.x = TURN_LEFT_SPD
            move.angular.z = ANGULAR_VEL
        elif 350 <= x <= 450:
            move.linear.x = STRAIGHT_SPD
            move.angular.z = 0.0
        else:
            move.linear.x = TURN_RIGHT_SPD
            move.angular.z = -ANGULAR_VEL
        
        pub.publish(move)

    except Exception as e:
        print(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)

    global bridge, pub
    bridge = CvBridge()
    node = rclpy.create_node('move_robot')
    
    sub = node.create_subscription(Image, '/camera1/image_raw', image_callback, rclpy.qos.qos_profile_sensor_data)
    pub = node.create_publisher(Twist, '/cmd_vel', rclpy.qos.qos_profile_system_default)
    timer = node.create_timer(TIMER_PERIOD, callback=callback)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot before shutting down
        stop_cmd = Twist()
        pub.publish(stop_cmd)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
