#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Empty
from sensor_msgs.msg import Image

from tf_transformations import quaternion_matrix, euler_from_matrix
from cv_bridge import CvBridge
# from PyQt5.QtGui import QImage
import cv2
import numpy as np
import struct
import os



class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        
        # Parameters
        self.declare_parameter('frequency', 50.0)
        self.declare_parameter('time', 10.)
        self.declare_parameter('robot_name', 'bebop')
        self.declare_parameter('ref_image', "reference.png")
        self.declare_parameter('output', "output")
        self.declare_parameter('gains', [0.]*4)
        self.declare_parameter('K', [1.]*9)
        self.declare_parameter('p0', [1.]*4)
        
        self.frequency = self.get_parameter('frequency').value
        self.time = self.get_parameter('time').value
        self.name = self.get_parameter('robot_name').value.strip()
        self.ref_image = self.get_parameter('ref_image').value
        self.output = self.get_parameter('output').value
        self.gains = self.get_parameter('gains').value
        self.K = self.get_parameter('K').value
        self.initial_cond = self.get_parameter('p0').value

        #   Camera calibration data
        self.f = [self.K[0], self.K[4]]
        self.pPrinc = [self.K[2],self.K[5]]
        self.K = np.array(self.K).reshape((3,3))
        print(self.K)

        if not self.name:
            self.get_logger().info('Empty "robot_name": Setting "bebop" as default.')
            self.name = 'bebop'
        self.get_logger().info(f"Robot Name: {self.name}")

        #   Reference image
        image_ref = cv2.imread(self.ref_image)
        if  image_ref is None :
            self.get_logger.error(f"Image {self.ref_image} could not be read ")
            return
        gray_image = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self.corners_ref, self.ids_ref, rejected = self.detector.detectMarkers(gray_image)
        if self.ids_ref is None:
            self.get_logger().error(f"No detected Markers")
            return
        self._ids_ref = self.ids_ref.tolist()
        cv2.aruco.drawDetectedMarkers(image_ref, self.corners_ref, self.ids_ref,
                                        borderColor = (100,1.,0.) )
        cv2.imwrite("reference_proc.png", image_ref)

        #   Camera and robot transformations
        self.R_cam = np.array([[0.,  0., 1.],
                               [-1., 0., 0.],
                               [0., -1., 0.]])
        self.t_cam = np.array([0.12, 0., 0.])   #   Different in real Bebop

        #   Subs and Pubs
        qos = QoSProfile(depth=2)
        self.cmd_pub = self.create_publisher(Twist,
                                             f"/{self.name}/cmd_vel",
                                             qos)
        # camera_tilt = self.create_publisher(Twist,
        self.camera_tilt = self.create_publisher(Vector3,
                                             f"{self.name}/move_camera",
                                             qos)

        self.takeoff_pub = self.create_publisher(Empty,
                                             f"/{self.name}/takeoff",
                                             qos)
        self.land_pub = self.create_publisher(Empty,
                                             f"/{self.name}/land",
                                             qos)

        self.stop_sub = self.create_subscription(Empty,
                                             "/stop",
                                                  self.stop,
                                                  qos)
        self.start_sub = self.create_subscription(Empty,
                                             "/start",
                                                  self.start,
                                                  qos)

        #   Image bridge
        img_qos = QoSProfile(depth=2)
        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(
            Image, f"/{self.name}/camera/image_raw",
            self.image_recv,
            img_qos)
        self.image_pub = self.create_publisher(Image,
                                               '/matching',
                                               img_qos)

        
        #   output files for data storage:
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

        #   State
        self.error = np.zeros(6)
        self.points_ref = None
        self.points = None
        self.u = np.zeros(6)
        self._u = np.zeros(6)
        self.wait = int(self.time * self.frequency)
        self.counter = 0
        self.idle = True
        self.found_arucos_w = False
        self.m_vel = Twist()
        self.cv_image = None

        # INIT control loop
        self.not_init = True
        self.timer = self.create_timer(1.0 / self.frequency, self.control_loop)

        self.idle_counter = 0.
        self.idle_lim = .5 / self.frequency

    def start(self,msg):
        self.takeoff_pub.publish(Empty())
        self.counter = 0
        self.idle = False

    def stop(self, msg):
        self.land_pub.publish(Empty())
        self.land_pub.publish(Empty())
        self.counter = 0
        self.idle = True
        self.idle_counter = 0.
        self.get_logger().info(f"Stopping..")

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
            data = (t,)
            data += tuple(self.error.reshape(-1))
            binary = struct.pack('ddddddd', *data)
            f.write(binary)

    def control_loop(self):

        if self.not_init:
            self.camera_tilt.publish(Vector3())
            self.camera_tilt.publish(Vector3())
            self.not_init = False

        if self.idle:
            if self.idle_counter <= 0:
                self.land_pub.publish(Empty())
                self.idle_counter = self.idle_lim
            else:
                self.idle_counter -= 1
            return

        if self.counter< self.wait/2.:
            self.get_logger().info("Waiting...")
            self.counter += 1
            return

        msg = Twist();
        if self.counter < self.wait:
            self.get_logger().info(f"V = 0")
            self.cmd_pub.publish( msg )
            self.counter += 1
            return

        if self.counter == self.wait:
            self.get_logger().info("Start")
            self.counter += 1

        #   IBVS
        if self.points_ref is None:
            self.cmd_pub.publish( msg )
            return

        #   BEGIN Control
        H, _ = cv2.findHomography(self.points_ref.T,
                                self.points.T,
                                cv2.RANSAC)

        if H is None:
            H = np.eye(3)
        else:
            H /= H[1,1]

        #   As stated in formulation
        _v =  np.concatenate(( self.points_ref[:,0], [1.]))
        self.error[:3] = (H-np.eye(3)) @ _v


        ##   used in paper
        #_v1 = self.s_obj_n.mean(axis = 1)
        #_v2 = self.s_n.mean(axis = 1)
        #_v1 =  np.concatenate(( _v1, [1.]))
        #_v2 =  np.concatenate(( _v2, [1.]))
        #_v = _v2.T @ H @ _v1
        #_v /= np.dot( _v2, _v2)
        #self.error[:3] = _v* _v2 - _v1


        _H = H - H.T
        #print(H)
        self.error[3] = - _H[1,2]
        self.error[4] = _H[0,2]
        self.error[5] = - _H[0, 1]
        #print(self.error)

        self._u[:3] = self.error[:3]
        self._u[3:] = self.error[3:]

        #   END Control

        #   Transformation camera -> robot

        #   6DOF
        _w = (self.R_cam @ self._u[3:]).reshape(-1)
        _v = (self.R_cam @ self._u[:3]).reshape(-1)
        # print(_v.shape, self.t_cam.shape, _w.shape)
        _v += np.cross( self.t_cam , _w )
        self.u[:3] = self.gains[:3] * _v.copy()
        self.u[3:] = self.gains[3] * _w.copy()

        #   4DOF
        # self.u[:3] = self.gains[:3] * (self.R_cam @ self._u[:3]).reshape(-1)
        # self.u[5] = -self.gains[3] * self._u[4]
        # self.u[:3] += np.cross(self.t_cam, self.u[3:])
        # self.u[:3] += - self.t_cam[0]* self.u[5] # TODO simplify cross product

        msg.linear.x = float(self.u[0])
        msg.linear.y = float(self.u[1])
        msg.linear.z = float(self.u[2])
        msg.angular.z = float(self.u[5])
        # self.get_logger().info( f"Control_cmd_vel: {self._u}")
        # self.get_logger().info( f"Control_cmd_vel: {self._u[3:]}")
        self.get_logger().info( f"Control_cmd_vel: {self.u[[0,1,2,5]]}")
        # self.get_logger().info( f"Control_cmd_vel: {msg.angular.z}")
        self.cmd_pub.publish(msg)
        self.save_data()


def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
