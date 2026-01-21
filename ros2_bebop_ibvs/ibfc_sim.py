#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Empty
from sensor_msgs.msg import Image

from tf_transformations import quaternion_matrix, euler_from_matrix
from cv_bridge import CvBridge
# from PyQt5.QtGui import QImage
import cv2
import numpy as np
import struct
import os

IDLE = 0
IBVS = 1
TAKEOFF = 2
LANDING = 3
STOP = 4
INITCOND = 5

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

class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        
        # Parameters
        self.declare_parameter('frequency', 50.0)
        self.declare_parameter('robot_name', 'bebop')
        self.declare_parameter('takeoff_threshold', 0.04)
        self.declare_parameter('landing_threshold', 0.08)
        self.declare_parameter('takeoff_height', 1.0)
        self.declare_parameter('label', 1)
        self.declare_parameter('ref_image', "reference.png")
        self.declare_parameter('output', "output")
        self.declare_parameter('img_depth', 1.)
        self.declare_parameter('gain', 1.)
        self.declare_parameter('gain_w', 1.)
        self.declare_parameter('gain_takeoff', 1.)
        self.declare_parameter('K', [1.]*9)
        self.declare_parameter('p0', [1.]*4)
        self.declare_parameter('polar', False)
        self.declare_parameter('save_log', False)
        
        self.frequency = self.get_parameter('frequency').value
        self.robot_name = self.get_parameter('robot_name').value.strip()
        self.takeoff_threshold = self.get_parameter('takeoff_threshold').value
        self.landing_threshold = self.get_parameter('landing_threshold').value
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.label = self.get_parameter('label').value
        self.ref_image = self.get_parameter('ref_image').value
        self.output = self.get_parameter('output').value
        self.img_depth = self.get_parameter('img_depth').value
        self.gain = self.get_parameter('gain').value
        self.kw = self.get_parameter('gain_w').value
        self.gain_takeoff = self.get_parameter('gain_takeoff').value
        self.K = self.get_parameter('K').value
        self.initial_cond = self.get_parameter('p0').value
        self.enable_polar = self.get_parameter('polar').value
        self.enable_log = self.get_parameter('save_log').value


        self.initial_cond =  np.array(self.initial_cond[self.label*4: (self.label+1)*4])

        #   Camera calibration data
        self.f = [self.K[0], self.K[4]]
        self.pPrinc = [self.K[2],self.K[5]]
        self.K = np.array(self.K).reshape((3,3))
        print(self.K)

        if not self.robot_name:
            self.get_logger().info('Empty "robot_name": Setting "bebop" as default.')
            self.robot_name = 'bebop'
        self.get_logger().info(f"Robot Name: {self.robot_name}")

        #   Reference image
        image_ref = cv2.imread(self.ref_image)
        if  image_ref is None :
            self.get_logger.error(f"Image {self.ref_image} could not be read ")
            return
        gray_image = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self.corners_ref, self.ids_ref, rejected = self.detector.detectMarkers(gray_image)
        if self.ids_ref is None:
            self.get_logger.error(f"No detected Markers")
            return
        self._ids_ref = self.ids_ref.tolist()
        cv2.aruco.drawDetectedMarkers(image_ref, self.corners_ref, self.ids_ref,
                                        borderColor = (100,1.,0.) )
        cv2.imwrite("reference_proc.png", image_ref)
        self.points = None

        #   Camera and robot transformations
        self.R_cam = np.array([[0.,  0., 1.],
                               [-1., 0., 0.],
                               [0., -1., 0.]])
        self.t_cam = np.array([0.12, 0., 0.])   #   Different in real Bebop

        #   Publishers
        qos = QoSProfile(depth=2)
        self.cmd_pub = self.create_publisher(Twist,
                                             f"/{self.robot_name}/cmd_vel",
                                             qos)
        self.cmd_enable = self.create_publisher(Bool,
                                                f"/{self.robot_name}/enable",
                                                qos)

        #   Image bridge
        img_qos = QoSProfile(depth=2)
        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(
            Image, '/camera/image_raw',
            self.image_recv,
            img_qos)
        self.image_pub = self.create_publisher(Image,
                                               '/matching',
                                               img_qos)

        #   Subscriptions
        self.pos_sub = self.create_subscription(Pose,
                                                f"/parrot_bebop_2/pose",
                                                self.pos_changed,
                                                qos)
        self.state_sub = self.create_subscription(Int32,
                                                  "/state",
                                                  self.state_changed,
                                                  qos)
        
        #   output files for data storage:
        self.position_d = os.path.join(self.output, "position.dat")
        with open(self.position_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.vel_d = os.path.join(self.output, "velocities.dat")
        with open(self.vel_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.norm_e_d = os.path.join(self.output, "norm_error.dat")
        with open(self.norm_e_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.arucos_d = os.path.join(self.output, "arUcos.dat")
        with open(self.arucos_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.error_d = os.path.join(self.output, "error.dat")
        with open(self.error_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.log_d = os.path.join(self.output, "log.dat")
        with open(self.log_d, 'w') as file:
            pass  # 'w' mode clears the file's contents

        #   State
        self.state = IDLE
        self.u = np.zeros(6)
        self._u = np.zeros(6)
        self.new_state = IDLE
        self.current_pose = Pose()
        self.data2save = False
        self.enable = False
        self.found_arucos_w = False
        self.takeoff_complete = False  # Nuevo flag para controlar despegue completado
        self.m_vel = Twist()

        # INIT control loop
        self.timer = self.create_timer(1.0 / self.frequency, self.control_loop)



    def state_changed(self, msg):
        self.new_state = msg.data
        
    def pos_changed(self, msg):
        self.current_pose = msg

    def normalize(self, p):
        _p = p.copy()
        _p[0,:] -= self.pPrinc[0]#cu
        _p[1,:] -= self.pPrinc[1]#cv
        _p[0,:] /= self.f[0]
        _p[1,:] /= self.f[1]
        return _p

    def image_recv(self, msg):

        # self.get_logger().info("Image received")

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
        except KeyError as e:
            self.get_logger().error(f"Robot name not found in topic: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")

        # Extract ArUcos
        gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray_image)

        if ids is None:
            if self.found_arucos_w:
                self.get_logger().warning("No ArUcos found in received image")
                self.found_arucos_w = False
            self.points = None
            self.points_ref = None
            return
        if not self.found_arucos_w:
            self.get_logger().warning("ArUcos found in received image")
            self.found_arucos_w = True

        #   Pairing
        _p = np.array([])
        _p_ref = np.array([])
        self.ids = []
        # self.get_logger().info(str(ids.shape[0]))
        for i in range(ids.shape[0]):
            if ids[i,0] in self._ids_ref:
                idx = self._ids_ref.index(ids[i,0])
                _p = np.concatenate ((_p, corners[i].reshape(-1 )))
                _p_ref = np.concatenate ((_p_ref, self.corners_ref[idx].reshape(-1) ))
                self.ids.append(ids[i,0])

        if _p.shape[0] == 0:
            self.get_logger().warning(f"Matching Failed {self._ids_ref} {ids.tolist()}")
            return
        self.p = _p.reshape((-1,2))
        _p_ref = _p_ref.reshape((-1,2))
        # self.ids = np.array(self.ids, dtype = int)

        #   Normalize
        self.points = self.normalize(self.p.astype(float).T)
        self.points_ref = self.normalize(_p_ref.astype(float).T)

        if self.enable_polar:
            _r = np.linalg.norm(self.points, axis = 0)
            _t = np.arctan2(self.points[1,:], self.points[0,:])
            self.points = np.c_[_r, _t].T
            _r = np.linalg.norm(self.points_ref, axis = 0)
            _t = np.arctan2(self.points_ref[1,:], self.points_ref[0,:])
            self.points_ref = np.c_[_r, _t].T


        #   Publish detection
        _image = self.cv_image.copy()
        # print(ids)
        cv2.aruco.drawDetectedMarkers(_image, corners, ids,
                                        borderColor = (0,100,0.) )
        cv2.aruco.drawDetectedMarkers(_image, self.corners_ref, self.ids_ref,
                                        borderColor = (100,1.,0.) )

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(_image, "bgr8"))

    def save_data(self):

        t = self.get_clock().now().nanoseconds * 1e-9
        orientation_q = self.current_pose.orientation
        ang = get_yaw(orientation_q)
        with open(self.position_d, 'ab') as f:
            data = (t, self.current_pose.position.x,
                    self.current_pose.position.y,
                    self.current_pose.position.z,
                    ang)
            binary = struct.pack('ddddd', *data)
            f.write(binary)

        with open(self.vel_d, 'ab') as f:
            data = (t,) + tuple(self.u[[0,1,2,5]].reshape(-1))
            # data = (t,) + tuple(self.u[[0,1,2,3]].reshape(-1))
            # data = (t,) + tuple(self.u[[0,1,2,2]].reshape(-1))
            binary = struct.pack('ddddd', *data)
            f.write(binary)


        with open(self.norm_e_d, 'ab') as f:
            data = (t,np.linalg.norm(self.error))
            binary = struct.pack('dd', *data)
            f.write(binary)

        with open(self.arucos_d, 'ab') as f:
            for i in range(len(self.ids)):

                data = (t, self.ids[i])
                data += tuple(self.p[4*i:4*(i+1), :].reshape(-1))
                binary = struct.pack('didddddddd', *data)
                f.write(binary)

        with open(self.error_d, 'ab') as f:
            for i in range(len(self.ids)):

                data = (t, self.ids[i])
                data += tuple(self.error[:,4*i:4*(i+1)].T.reshape(-1))
                binary = struct.pack('didddddddd', *data)
                f.write(binary)

        # save log
        if not self.enable_log:
            return
        with open(self.log_d, 'ab') as f:
            data = (t,)
            # data += tuple(self.L.reshape(-1))
            data += tuple(self.svd.reshape(-1))
            # binary = struct.pack('d'*(1+8*6+6), *data) ## 6 dof y un aruco
            binary = struct.pack('d'*(1+6), *data) ## 6 dof only singular values
            # binary = struct.pack('d'*(1+8*4+4), *data) ## 4 dof
            f.write(binary)

    def control_loop(self):

        if self.state == IDLE:
            #   Change state
            if self.new_state == TAKEOFF:
                self.get_logger().info("State change: TAKEOFF")
                self.state = TAKEOFF
                self.takeoff_complete = False
            if self.new_state == INITCOND:
                self.get_logger().info("State change: INITCOND")
                self.state = INITCOND
                self.init_complete = False

        elif self.state == TAKEOFF:
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
            #   Change state
            if self.new_state == IBVS and self.takeoff_complete:
                self.get_logger().info("State change: IBVS")
                self.state = IBVS
            elif self.new_state == IBVS and  not self.takeoff_complete:
                self.get_logger().info("Waiting for TAKEOFF to finish, can not change to IBVS")
                self.new_state == TAKEOFF
            elif self.new_state == LANDING:
                self.get_logger().info("State change: LANDING")
                self.state = LANDING
            elif self.new_state == STOP:
                self.get_logger().info("State change: STOP")
                self.state = STOP
            elif self.new_state == INITCOND:
                self.get_logger().info("State change: INITCOND")
                self.state = INITCOND
                self.init_complete = False

        elif self.state == INITCOND:
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
            if self.new_state == IBVS and self.init_complete:
                self.get_logger().info("State change: IBVS")
                self.state = IBVS
            elif self.new_state == IBVS and  not self.init_complete:
                self.get_logger().info("Waiting for INITIAL CONDITION to finish, can not change to IBVS")
            elif self.new_state == LANDING:
                self.get_logger().info("State change: LANDING")
                self.state = LANDING
            elif self.new_state == STOP:
                self.get_logger().info("State change: STOP")
                self.state = STOP

        elif self.state == LANDING:

            current_z = self.current_pose.position.z
            msg = Twist()
            
            if current_z > self.landing_threshold:
                # Descender controladamente
                msg.linear.z = self.gain_takeoff* float(- current_z)
                self.cmd_pub.publish(msg)
            else:
                #   Landing finished
                self.get_logger().info("¡Landing complete!")
                self.state = IDLE
                self.enable = False
                self.cmd_enable.publish(Bool(data=self.enable))
                self.cmd_pub.publish(Twist())

            #   Change state
            if self.new_state == IDLE or  abs(current_z-.1) < self.takeoff_threshold:
                self.get_logger().info("State change: IDLE")
                self.state = IDLE
            elif self.new_state == STOP:
                self.get_logger().info("State change: STOP")
                self.state = STOP


        elif self.state == IBVS and self.points is None:

            self.get_logger().error("Image error can not be computed")

            if self.data2save:
                self.save_data()

            try:
                self.cmd_pub.publish(self.m_vel)

            except Exception as e:
                self.get_logger().error(f"Error with IBVS control: {str(e)}")
                self.enable = False
                self.cmd_enable.publish(Bool(data=self.enable))
            #   Change state
            if self.new_state == LANDING:
                self.get_logger().info("State change: LANDING")
                self.state = LANDING
            elif self.new_state == STOP:
                self.get_logger().info("State change: STOP")
                self.state = STOP
            elif self.new_state == INITCOND:
                self.get_logger().info("State change: INITCOND")
                self.state = INITCOND
                self.init_complete = False

        elif self.state == IBVS:
            # #   IBVS
            # self.error = self.points - self.points_ref
            # if self.enable_polar:
            #     select = (abs(self.error[1,:]) > np.pi)
            #     # self.get_logger().info(f"{select}")
            #     self.error[1,select] = -1.* np.sign(self.error[1,select]) * (2*np.pi - abs(self.error[1,select]))
            #
            #     # self.error[1,self.error[1,:] < np.pi] += 2*np.pi
            #     # self.error[1,self.error[1,:] > np.pi] -= 2*np.pi
            #
            #
            # # self.get_logger().info(f"Norm(s^*) = \n{self.points_ref}")
            # # self.get_logger().info(f"Norm(s) = \n{self.points}")
            # # self.get_logger().info(f"s^* = \n{self.corners_ref[0]}")
            # #   TODO: depth?
            # if self.enable_polar:
            #     self.L = interaction_matrix_polar(self.points, self.img_depth)
            #     # self.L = interaction_matrix_polar(self.points_ref, self.img_depth)
            # else:
            #     self.L = interaction_matrix_xyz(self.points_ref, self.img_depth)
            #     # self.L = interaction_matrix_y(self.points, self.img_depth)
            #     # self.L = interaction_matrix_t(self.points_ref, self.img_depth)
            # # self.get_logger().info( f"L: : {self.L}")
            # L_inv = Inv_Moore_Penrose(self.L)
            # if self.enable_log:
            #     _, self.svd, _ = np.linalg.svd(self.L.T @ self.L)
            #
            # if L_inv is None:
            #     self.get_logger().error("Invalid Ls matrix")
            #     # self.u =  np.zeros(6)
            #     self.cmd_pub.publish(self.m_vel)
            #
            # # self.u = - self.gain * L_inv @ self.error.T.reshape((-1,1))
            # self._u = - self.gain * L_inv @ self.error.T.reshape((-1,1))
            #
            #
            # # if abs(self.u[1].T) > 0.05 :
            # # self.get_logger().info( f"L: : {L_inv}")
            # # self.get_logger().info( f"Control_c y : {self._u.T}")
            # # self.get_logger().info( f"Error: {self.error}")
            # # self.get_logger().info( f"Error: {self.error.T.reshape((-1,1)).T}")
            #
            # #   Transformation camera -> robot
            #
            # #   6DOF
            # _w = self.R_cam @ self._u[3:]
            # _v = (self.R_cam @ self._u[:3]).reshape(-1)
            # _v += np.cross( self.t_cam , _w.reshape(-1) )
            # _w *= self.kw
            # self.u[:3] = _v.copy()
            # self.u[3:] = _w.reshape(-1)
            # #   4DOF
            # # _w = self.R_cam @ np.array([0.,self._u[3],0.])
            # # _v = (self.R_cam @ self._u[:3]).reshape(-1)
            # # _v += np.cross( self.t_cam , _w.reshape(-1) )
            # # _w *= self.kw
            # # self.u[:3] = _v.copy()
            # # self.u[3:] = _w.copy()
            #
            # # self.get_logger().info( f"Control_d: {self.u}")
            #
            # #   Send message
            # # msg = Twist()
            # # _norm = np.linalg.norm(self.u)
            # # if  _norm > .2:
            # #     self.u = .2 * self.u  / _norm
            #
            # self.m_vel.linear.x = float(self.u[0])
            # self.m_vel.linear.y = float(self.u[1])
            # self.m_vel.linear.z = float(self.u[2])
            # self.m_vel.angular.z = float(self.u[5])
            # # self.get_logger().info( f"Control_cmd_vel: {self.m_vel.angular.z}")
            # self.cmd_pub.publish(self.m_vel)
            #
            # #   BEGIN
            # # self.m_vel = Twist()
            # # # self.m_vel.linear.x = float(.1)
            # # # self.m_vel.linear.x = float(self.u[0])
            # # # self.m_vel.linear.y = float(-.1)
            # # # self.m_vel.linear.y = float(self.u[1])
            # # self.m_vel.linear.z = float(self.u[2])
            # # # self.m_vel.angular.z = float(self.u[5])
            # # self.cmd_pub.publish(self.m_vel)
            #
            # #   END
            #
            # #   Save data
            # self.save_data()
            # self.data2save = True

            #   Change state
            if self.new_state == LANDING:
                self.get_logger().info("State change: LANDING")
                self.state = LANDING
            elif self.new_state == STOP:
                self.get_logger().info("State change: STOP")
                self.state = STOP
            elif self.new_state == INITCOND:
                self.get_logger().info("State change: INITCOND")
                self.state = INITCOND
                self.init_complete = False

        elif self.state == STOP:
            self.cmd_pub.publish(Twist())
            self.cmd_pub.publish(Twist())
            self.cmd_pub.publish(Twist())
            self.enable = False
            self.cmd_enable.publish(Bool(data=self.enable))
            self.get_logger().info("State change: IDLE")
            self.state = IDLE

def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
