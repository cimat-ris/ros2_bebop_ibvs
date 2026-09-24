#   Ros based packages for Image processing and basic IBVS controllers for single robot and multiagent.



#!/usr/bin/env python3

# import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool, Int32
from tf_transformations import quaternion_matrix, euler_from_matrix
from cv_bridge import CvBridge
# from PyQt5.QtGui import QImage
import cv2
import numpy as np
# import struct
# import os

# STATE = ["IDLE",
#           "CONTROL",
#           "TAKEOFF",
#           "LAND",
#           "STOP",
#           "INITCOND",
#           "REFERENCE",
#           "RESETVIS",
#           ]
IDLE = 0
CONTROL = 1
TAKEOFF = 2
LAND = 3
STOP = 4
INITCOND = 5
REFERENCE = 6
RESETVIS = 7

markers_list = ["4X4_50" ,
        "4X4_100" ,
        "4X4_250" ,
        "4X4_1000" ,
        "5X5_50" ,
        "5X5_100" ,
        "5X5_250" ,
        "5X5_1000" ,
        "6X6_50" ,
        "6X6_100" ,
        "6X6_250" ,
        "6X6_1000" ,
        "7X7_50" ,
        "7X7_100" ,
        "7X7_250" ,
        "7X7_1000" ,
        "ARUCO_ORIGINAL" ,
        "APRILTAG_16h5" ,
        "APRILTAG_25h9" ,
        "APRILTAG_36h10" ,
        "APRILTAG_36h11" ,
        "ARUCO_MIP_36h12"]

def get_yaw(orientation):
    a = 2* (orientation.w * orientation.z + orientation.x * orientation.y)
    b = 1 - 2 *(orientation.y**2 + orientation.z** 2)
    return np.arctan2(a,b)



def interaction_matrix_xyz(points,Z):

    n = points.shape[1]
    L = np.zeros((n,12), dtype= np.float64)
    L[:,0]  =   L[:,7] = -1/Z
    L[:,2]  =   points[0,:]/Z
    L[:,3]  =   points[0,:]*points[1,:]
    L[:,4]  =   -(1+points[0,:]**2)
    L[:,5]  =   points[1,:]
    L[:,8]  =   points[1,:]/Z
    L[:,9]  =   1+points[1,:]**2
    L[:,10] =   -points[0,:]*points[1,:]
    L[:,11] =   -points[0,:]

    return L.reshape((-1,6))

def interaction_matrix_y(points,Z):

    n = points.shape[1]
    L = np.zeros((n,8))
    L[:,0]  =   L[:,5] = -1/Z
    L[:,2]  =   points[0,:]/Z
    L[:,3]  =  -(1+points[0,:]**2)
    L[:,6]  =   points[1,:]/Z
    L[:,7]  =  -points[0,:]*points[1,:]

    return L.reshape((-1,4))

def interaction_matrix_z(points,Z):
    n = points.shape[1]
    L = np.zeros((n,8))
    L[:,0]  =   L[:,5] = -1/Z
    L[:,2]  =   points[0,:]/Z
    L[:,3]  =   points[1,:]
    L[:,6]  =   points[1,:]/Z
    L[:,7] =   -points[0,:]
    return L.reshape((-1,4))

def interaction_matrix_t(points,Z):
    n = points.shape[1]
    L = np.zeros((n,6))
    L[:,0]  =   L[:,4] = -1/Z
    L[:,2]  =   points[0,:]/Z
    L[:,5]  =   points[1,:]/Z


    return L.reshape((-1,3))

def interaction_matrix_polar(points, Z):

    n = points.shape[1]
    L = np.zeros((n,12))

    c = np.cos(points[1,:])
    s = np.sin(points[1,:])

    L[:,0]  =   -c/Z
    L[:,1]  =   -s/Z
    L[:,2]  =   points[0,:]/Z
    L[:,3]  =   (1+points[0,:]**2)*s
    L[:,4]  =   -(1+points[0,:]**2)*c
    L[:,6]  =   s/(points[0,:]*Z)
    L[:,7]  =   -c/(points[0,:]*Z)
    L[:,9] =    c/points[0,:]
    L[:,10] =   s/points[0,:]
    L[:,11] =   -1.

    return L.reshape((-1,6))

def Inv_Moore_Penrose(L):
    A = L.T@L
    if np.linalg.det(A) == 0:
        return None
    return np.linalg.inv(A) @ L.T

# def custom_draw_matching(image1, image2, points1, points2,
#                          color1=(0, 0, 255), color2=(0, 255, 0),
#                          point_radius=3, line_thickness = 1):
#
#     _shape = list(image1.shape)
#     W = _shape[1]
#     _shape[1] *= 2
#     _shape = tuple(_shape)
#     output_image = np.zeros(_shape, dtype = image1.dtype)
#     output_image[:,:W,:] = image1.copy()
#     output_image[:,W:,:] = image2.copy()
#
#     # Draw points from the first array
#     _points2 = points2.copy()
#     _points2 += np.array([W,0.])
#     for i in range(points1.shape[0]):
#         # Draw the line
#         cv2.line(output_image, points1[i,:].astype(int), _points2[i,:].astype(int), color1, line_thickness)
#
#         # Draw points
#         cv2.circle(output_image, points1[i,:].astype(int), point_radius, color1, -1)
#         cv2.circle(output_image, _points2[i,:].astype(int), point_radius, color2, -1)
#
#     return output_image

#   ---------------------------------------------------
#   ---------------------------------------------------
#   IMAGE PROC CLASS : Feature matcher
#   ---------------------------------------------------
#   ---------------------------------------------------

# Assumtions:
#     This will be used instead of Node class

class FeatureMatcher(Node):

    def __init__(self, name):

        super().__init__(name)
        self.track_conf = False
        super().declare_parameter('nfeatures', 100)
        super().declare_parameter('scaleFactor', 1.2)
        super().declare_parameter('nlevels', 8)
        super().declare_parameter('edgeThreshold', 15)
        super().declare_parameter('patchSize', 30)
        super().declare_parameter('fastThreshold', 20)
        super().declare_parameter('flann_ratio', 0.7)
        super().declare_parameter('matcher_threshold', 12)
        super().declare_parameter('K', [1.]*9)


        #   TODO: matriz de intrinsecos

        self.nfeatures = super().get_parameter('nfeatures').value
        self.scaleFactor = super().get_parameter('scaleFactor').value
        self.nlevels = super().get_parameter('nlevels').value
        self.edgeThreshold = super().get_parameter('edgeThreshold').value
        self.patchSize = super().get_parameter('patchSize').value
        self.fastThreshold = super().get_parameter('fastThreshold').value
        self.flann_ratio = super().get_parameter('flann_ratio').value
        self.matcher_threshold = super().get_parameter('matcher_threshold').value
        self.K = super().get_parameter('K').value

        self.f = [self.K[0], self.K[4]]
        self.pPrinc = [self.K[2],self.K[5]]
        self.K = np.array(self.K).reshape((3,3))

    def config_reference(self, ref_name):

        # self.lk_params = dict(winSize=(15, 15),
        #                     maxLevel=2,
        #                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.orb = cv2.ORB_create(
            nfeatures    = int( self.nfeatures ),
            scaleFactor  = self.scaleFactor,
            nlevels      = int( self.nlevels ),
            edgeThreshold= int( self.edgeThreshold ),
            patchSize    = int( self.patchSize ),
            fastThreshold= int( self.fastThreshold )
            )

        #   Matcher
        index_params = {
            "algorithm": 6,
            "table_number": 20,
            "key_size": 10,
            "multi_probe_level": 2,
        }

        self.flann = cv2.FlannBasedMatcher(index_params)

        self.image_ref = cv2.imread(ref_name)
        if  self.image_ref is None :
            return False
        gray_image = cv2.cvtColor(self.image_ref, cv2.COLOR_BGR2GRAY)
        self.kp_ref, self.desc_ref = self.orb.detectAndCompute(gray_image, None)
        if self.desc_ref is None:
            return False

        self.prev_image = np.zeros((gray_image.shape),
                                    dtype = gray_image.dtype)
        self.match_threshold = self.matcher_threshold
        self.track_conf = True

        return True

    def normalize(self, p):
        _p = p.copy()
        _p[0,:] -= self.pPrinc[0]#cu
        _p[1,:] -= self.pPrinc[1]#cv
        _p[0,:] /= self.f[0]
        _p[1,:] /= self.f[1]
        return _p

    def match(self, desc, deltas):

        #   match
        knn_matches = self.flann.knnMatch(desc,
                                self.desc_self, k=2)
        # Lowe ratio test
        good_matches = []
        for matches in knn_matches:
            if len(matches) == 2:
                m, n = matches
                if m.distance < self.flann_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) <= 4:
            super().get_logger().warning("No Neighboring Matches available")
            return None

        _delta_i = np.float32([
            self.points[:,m.trainIdx]
            for m in good_matches
        ])
        _delta_j = np.float32([
            deltas[:,m.queryIdx]
            for m in good_matches
        ])


        _, _mask = cv2.findHomography(_delta_i, _delta_j, cv2.RANSAC)
        _mask = _mask.reshape(-1)

        if _mask.sum() <= 4:
            super().get_logger().warning("No Neighboring Matches available (RANSAC)")
            return None

        _delta_i = _delta_i[_mask == 1,:].T
        _delta_j = _delta_j[_mask == 1,:].T
        _delta_i = _delta_i.reshape((2,-1))
        _delta_j = _delta_j.reshape((2,-1))

        return _delta_i, _delta_j

    def img_proc(self):
        gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

        #   Detect
        # Extract Matches if not enough
        kp, self.desc_self = self.orb.detectAndCompute(gray_image, None)

        if len(kp) <4:
            return

        #  Kp conversion
        self.p = np.float32([k.pt for k in kp  ])
        self.points = self.normalize(self.p.astype(float).T)


    #   Takes a list of points and overlaps the matches in the same picture
    def custom_draw_matching(self, m_image, points1, points2,
                         color1=(0, 0, 255), color2=(0, 255, 0),
                         point_radius=3, line_thickness = 1,
                         reproject = False):

        if reproject:
            _points1 =  np.vstack([points1, np.ones(points1.shape[1])])
            _points1 = self.K @ _points1
            _points1 = _points1[:2,:] / _points1[2,:]
            _points2 =  np.vstack([points2, np.ones(points2.shape[1])])
            _points2 = self.K @ _points2
            _points2 = _points2[:2,:] / _points2[2,:]
            _points1 =  _points1.T
            _points2 =  _points2.T
        else:
            _points1 =  points1.T
            _points2 =  points2.T

        for i in range(points1.shape[0]):
            # Draw the line
            cv2.line(m_image, _points1[i,:].astype(int), _points2[i,:].astype(int), color2, line_thickness)

            # Draw points
            cv2.circle(m_image, _points1[i,:].astype(int), point_radius, color1, -1)
            cv2.circle(m_image, _points2[i,:].astype(int), int(0.5*point_radius), color2, -1)

#   ---------------------------------------------------
#   ---------------------------------------------------
#   STATE MACHINE CLASS BASE
#   ---------------------------------------------------
#   ---------------------------------------------------

# Asumtions:
#     The following are defined:
#         self.cmd_pub
#         self.cmd_enable
#         self.pos_sub
#         self.initial_cond
#         self.reference_pose
#         self.gain_takeoff

class State(Node):

    def __init__(self,name):
        super().__init__(name)
        self.state = self.s_idle
        self.new_state = IDLE
        self.current_pose = Pose()
        self.m_vel = Twist()
        self.takeoff_complete = False

        super().declare_parameter('takeoff_threshold', 0.04)
        super().declare_parameter('landing_threshold', 0.08)
        super().declare_parameter('takeoff_height', 1.0)
        super().declare_parameter('gain_takeoff', 1.)


        self.takeoff_threshold = super().get_parameter('takeoff_threshold').value
        self.landing_threshold = super().get_parameter('landing_threshold').value
        self.takeoff_height = super().get_parameter('takeoff_height').value
        self.gain_takeoff = super().get_parameter('gain_takeoff').value


    def create_publishers(self, qos):
        # qos = QoSProfile(depth=2)
        self.cmd_pub = self.create_publisher(Twist,
                                             f"/{self.robot_name}_{self.label}/cmd_vel",
                                             qos)

        print(f"/{self.robot_name}_{self.label}/cmd_vel" )
        self.cmd_enable = self.create_publisher(Bool,
                                                f"/{self.robot_name}_{self.label}/enable",
                                                qos)

        #   Subscriptions
        self.pos_sub = self.create_subscription(Pose,
                                                f"/{self.robot_name}_{self.label}/pose",
                                                self.pos_changed,
                                                qos)
    def pos_changed(self, msg):
        self.current_pose = msg

    def s_idle(self):

        if self.new_state == IDLE:
            return
        if self.new_state == TAKEOFF:
            self.get_logger().info("State change: TAKEOFF")
            self.state = self.s_takeoff
            self.takeoff_complete = False
            return
        if self.new_state == INITCOND:
            self.get_logger().info("State change: INITCOND")
            self.state = self.s_init_cond
            self.init_complete = False
            return
        if self.new_state == REFERENCE:
            self.get_logger().info("State change: REFERENCE")
            self.state = self.s_reference
            self.init_complete = False

    def s_takeoff(self):

        current_z = self.current_pose.position.z
        delta = current_z- self.takeoff_height

        if abs(delta) < self.takeoff_threshold and not self.takeoff_complete:
            #   Proportional control iniside takeoff_threshold
            self.get_logger().info(f"Takeoff completed: {current_z:.2f}m")
            self.takeoff_complete = True

        msg = Twist()
        msg.linear.z = -self.gain_takeoff*float(delta)
        self.cmd_pub.publish(msg)

        self.get_logger().debug(f"Control input: {msg.linear.z}")

        if self.new_state == CONTROL and self.takeoff_complete:
            self.get_logger().info("State change: CONTROL")
            self.state = self.s_control
            return
        if self.new_state == CONTROL and  not self.takeoff_complete:
            self.get_logger().info("Waiting for TAKEOFF to finish, can not change to CONTROL")
            self.new_state = TAKEOFF
            return
        if self.new_state == LAND:
            self.get_logger().info("State change: LAND")
            self.state = self.s_land
            return
        if self.new_state == STOP:
            self.get_logger().info("State change: STOP")
            self.state = self.s_stop
            return
        if self.new_state == INITCOND:
            self.get_logger().info("State change: INITCOND")
            self.state = self.s_init_cond
            self.init_complete = False
            return
        if self.new_state == REFERENCE:
            self.get_logger().info("State change: REFERENCE")
            self.state = self.s_reference
            self.init_complete = False
            return

    def s_init_cond(self):

        _my_position = [self.current_pose.position.x,
                        self.current_pose.position.y,
                        self.current_pose.position.z]
        my_position = np.array(_my_position)
        _orientation = [self.current_pose.orientation.x,
                        self.current_pose.orientation.y,
                        self.current_pose.orientation.z,
                        self.current_pose.orientation.w]

        _delta = my_position- self.initial_cond[:3]
        if np.linalg.norm(_delta) < self.takeoff_threshold and not self.init_complete:
            #   Proportional control iniside takeoff_threshold
            self.get_logger().info(f"Initial condition reached")
            self.init_complete = True

        msg = Twist()
        _u = -self.gain_takeoff * _delta
        _R = quaternion_matrix(_orientation)
        _R = _R[:3,:]
        _R = _R[:,:3]
        _u = _R.T @ _u

        _, _, _yaw = euler_from_matrix(_R)

        _yaw = _yaw - self.initial_cond[3]
        _yaw = _yaw + 2*np.pi if _yaw < np.pi else _yaw
        _yaw = _yaw - 2*np.pi if _yaw > np.pi else _yaw

        msg.linear.x = float(_u[0])
        msg.linear.y = float(_u[1])
        msg.linear.z = float(_u[2])
        msg.angular.z = float(-self.gain_takeoff* _yaw)
        self.cmd_pub.publish(msg)

        self.get_logger().debug(f"Control input: {_u}")

        #   Change state
        if self.new_state == CONTROL and self.init_complete:
            self.get_logger().info("State change: CONTROL")
            self.state = self.s_control
            return
        if self.new_state == CONTROL and  not self.init_complete:
            self.get_logger().info("Waiting for INITIAL CONDITION to finish, can not change to CONTROL")
            self.new_state = INITCOND
            return
        if self.new_state == LAND:
            self.get_logger().info("State change: LAND")
            self.state = self.s_land
            return
        if self.new_state == STOP:
            self.get_logger().info("State change: STOP")
            self.state = self.s_stop
            return
        if self.new_state == REFERENCE:
            self.get_logger().info("State change: REFERENCE")
            self.state = self.s_reference
            self.init_complete = False

    def s_reference(self):

        _my_position = [self.current_pose.position.x,
                        self.current_pose.position.y,
                        self.current_pose.position.z]
        my_position = np.array(_my_position)
        _orientation = [self.current_pose.orientation.x,
                        self.current_pose.orientation.y,
                        self.current_pose.orientation.z,
                        self.current_pose.orientation.w]

        _delta = my_position- self.reference_pose[:3]
        if np.linalg.norm(_delta) < self.takeoff_threshold and not self.init_complete:
            #   Proportional control iniside takeoff_threshold
            self.get_logger().info(f"Reference pose reached")
            self.init_complete = True

        msg = Twist()
        _u = -self.gain_takeoff * _delta
        _R = quaternion_matrix(_orientation)
        _R = _R[:3,:]
        _R = _R[:,:3]
        _u = _R.T @ _u

        _, _, _yaw = euler_from_matrix(_R)

        _yaw = _yaw - self.reference_pose[3]
        _yaw = _yaw + 2*np.pi if _yaw < np.pi else _yaw
        _yaw = _yaw - 2*np.pi if _yaw > np.pi else _yaw

        msg.linear.x = float(_u[0])
        msg.linear.y = float(_u[1])
        msg.linear.z = float(_u[2])
        msg.angular.z = float(-self.gain_takeoff* _yaw)
        self.cmd_pub.publish(msg)

        self.get_logger().debug(f"Control input: {_u}")

        #   Change state
        if self.new_state == CONTROL and self.init_complete:
            self.get_logger().info("State change: CONTROL")
            self.state = self.s_control
            return
        if self.new_state == CONTROL and  not self.init_complete:
            self.get_logger().info("Waiting for REFERENCE CONDITION to finish, can not change to CONTROL")
            self.new_state = REFERENCE
            return
        if self.new_state == LAND:
            self.get_logger().info("State change: LAND")
            self.state = self.s_land
            return
        if self.new_state == STOP:
            self.get_logger().info("State change: STOP")
            self.state = self.s_stop
            return
        if self.new_state == INITCOND:
            self.get_logger().info("State change: INITCOND")
            self.state = self.s_init_cond
            self.init_complete = False

    def s_land(self):
        current_z = self.current_pose.position.z
        msg = Twist()

        # Descender controladamente
        msg.linear.z = self.gain_takeoff* float(- current_z)
        self.cmd_pub.publish(msg)


        if current_z <= self.landing_threshold:
            #   Landing finished
            self.get_logger().info("¡Landing complete!")
            self.state = self.s_idle
            # self.enable = False
            self.cmd_enable.publish(Bool(data=False))
            self.cmd_pub.publish(Twist())
            return
        #   Change state
        if self.new_state == IDLE or  abs(current_z-.1) < self.takeoff_threshold:
            self.get_logger().info("State change: IDLE")
            self.state = self.s_idle
            return
        if self.new_state == STOP:
            self.get_logger().info("State change: STOP")
            self.state = self.s_stop

    def s_control(self, u):

        self.m_vel.linear.x = float(u[0])
        self.m_vel.linear.y = float(u[1])
        self.m_vel.linear.z = float(u[2])
        self.m_vel.angular.z = float(u[5])
        # self.get_logger().info( f"Control_cmd_vel: {self.m_vel.angular.z}")
        # self.cmd_pub.publish(self.m_vel)

        try:
            self.cmd_pub.publish(self.m_vel)
        except Exception as e:
            self.get_logger().error(f"Error with IBFC control: {str(e)}")
            self.enable = False
            self.cmd_enable.publish(Bool(data=self.enable))


        #   Change state
        if self.new_state == LAND:
            self.get_logger().info("State change: LAND")
            self.state = self.s_land
            return
        if self.new_state == STOP:
            self.get_logger().info("State change: STOP")
            self.state = self.s_stop
            return
        if self.new_state == INITCOND:
            self.get_logger().info("State change: INITCOND")
            self.state = self.s_init_cond
            self.init_complete = False
            return
        if self.new_state == REFERENCE:
            self.get_logger().info("State change: REFERENCE")
            self.state = self.s_reference
            self.init_complete = False

    def s_stop(self):
        self.cmd_pub.publish(Twist())
        self.cmd_pub.publish(Twist())
        self.cmd_pub.publish(Twist())
        # self.enable = False
        self.cmd_enable.publish(Bool(data=False))
        self.get_logger().info("State change: IDLE")
        self.state = self.s_idle



