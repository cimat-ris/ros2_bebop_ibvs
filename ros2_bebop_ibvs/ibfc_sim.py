#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from formation_interfaces.msg import Corners, ArUco

from tf_transformations import quaternion_matrix, euler_from_matrix
from cv_bridge import CvBridge
# from PyQt5.QtGui import QImage
import cv2
import numpy as np
import struct
import os

IDLE = 0
IBFC = 1
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
        self.declare_parameter('robot_name_prefix', 'bebop')
        self.declare_parameter('takeoff_threshold', 0.04)
        self.declare_parameter('landing_threshold', 0.08)
        self.declare_parameter('takeoff_height', 1.0)
        self.declare_parameter('label', 1)
        self.declare_parameter('n_agents', 1)
        self.declare_parameter('reference_image_prefix', "reference_f")
        self.declare_parameter('output', "output")
        self.declare_parameter('img_depth', 1.)
        self.declare_parameter('gain', 1.)
        self.declare_parameter('gain_int', 0.)
        self.declare_parameter('gain_w', 1.)
        self.declare_parameter('gain_takeoff', 1.)
        self.declare_parameter('K', [1.]*9)
        self.declare_parameter('p0', [1.]*4)
        self.declare_parameter('polar', False)
        self.declare_parameter('save_log', False)
        
        self.frequency = self.get_parameter('frequency').value
        self.robot_name = self.get_parameter('robot_name_prefix').value.strip()
        self.takeoff_threshold = self.get_parameter('takeoff_threshold').value
        self.landing_threshold = self.get_parameter('landing_threshold').value
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.label = self.get_parameter('label').value
        self.n_agents = self.get_parameter('n_agents').value
        self.reference_image_prefix = self.get_parameter('reference_image_prefix').value
        self.output = self.get_parameter('output').value
        self.img_depth = self.get_parameter('img_depth').value
        self.gain = self.get_parameter('gain').value
        self.kw = self.get_parameter('gain_w').value
        self.k_int = self.get_parameter('gain_int').value
        self.gain_takeoff = self.get_parameter('gain_takeoff').value
        self.K = self.get_parameter('K').value
        self.initial_cond = self.get_parameter('p0').value
        self.enable_polar = self.get_parameter('polar').value
        self.enable_log = self.get_parameter('save_log').value

        #   inital conditions
        self.initial_cond =  np.array(self.initial_cond)
        self.initial_cond = self.initial_cond.reshape((-1,4))
        self.initial_cond = self.initial_cond[self.label].reshape(-1)

        #   Camera calibration data
        self.f = [self.K[0], self.K[4]]
        self.pPrinc = [self.K[2],self.K[5]]
        self.K = np.array(self.K).reshape((3,3))
        print(self.K)

        if not self.robot_name:
            self.get_logger().info('Empty "robot_name": Setting "bebop" as default.')
            self.robot_name = 'bebop'
        # self.get_logger().info(f"Robot Name: {self.robot_name}_{self.label}")

        #   Detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        #   Reference image
        self.ids_ref = [None]* self.n_agents
        self._ids_ref = [None]* self.n_agents
        self.points_ref = [None]* self.n_agents
        self.corners_ref = [None]* self.n_agents
        for i in range(self.n_agents):
            image_ref = cv2.imread(f"{self.reference_image_prefix}_{i}.png")
            if  image_ref is None :
                self.get_logger().error(f"Image {self.reference_image_prefix}_{i}.png could not be read ")
                return
            gray_image = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)

            corners_ref, ids_ref, rejected = self.detector.detectMarkers(gray_image)
            if ids_ref is None:
                self.get_logger().error(f"No detected Markers")
                return
            self.corners_ref[i] = corners_ref
            _corners_ref = np.array( corners_ref)
            _corners_ref = _corners_ref.astype(float).reshape((-1,2)).T
            self.points_ref[i] = self.normalize(_corners_ref)
            self._ids_ref[i]  = ids_ref
            self.ids_ref[i]  = [ j[0] for j in  ids_ref.tolist()]
            cv2.aruco.drawDetectedMarkers(image_ref, corners_ref, ids_ref,
                                         borderColor = (100,1.,0.) )
            _name = os.path.join(self.output, f"reference_proc_{self.label}_{i}.png")
            cv2.imwrite(_name, image_ref)
        self.ids = [None]*self.n_agents
        self.points = [None]*self.n_agents
        self.p = None

        #   Camera and robot transformations
        self.R_cam = np.array([[0.,  0., 1.],
                               [-1., 0., 0.],
                               [0., -1., 0.]])
        self.t_cam = np.array([0.12, 0., 0.])   #   Different in real Bebop

        #   Publishers
        qos = QoSProfile(depth=2)
        self.cmd_pub = self.create_publisher(Twist,
                                             f"/{self.robot_name}_{self.label}/cmd_vel",
                                             qos)
        self.cmd_enable = self.create_publisher(Bool,
                                                f"/{self.robot_name}_{self.label}/enable",
                                                qos)
        # self.get_logger().info(f"control: /{self.robot_name}_{self.label}/cmd_vel")
        #   Image bridge
        img_qos = QoSProfile(depth=2)
        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(
            Image, f"/{self.robot_name}_{self.label}/image",
            self.image_recv,
            img_qos)
        self.image_pub = self.create_publisher(Image,
                                                f"/{self.robot_name}_{self.label}/matching",
                                               img_qos)

        #   Subscriptions
        self.pos_sub = self.create_subscription(Pose,
                                                f"/{self.robot_name}_{self.label}/pose",
                                                self.pos_changed,
                                                qos)
        self.state_sub = self.create_subscription(Int32,
                                                  "/state",
                                                  self.state_changed,
                                                  qos)

        #   Network
        #   TODO: graph
        self.features_sub  = []
        self.features_pub  = []
        for i in range(self.n_agents):
            if i != self.label:
                _pub = self.create_publisher(Corners,
                                f"/{self.robot_name}_{self.label}_{i}/ArUcos",
                                qos)
                self.features_pub.append(_pub)
                _sub = self.create_subscription(Corners,
                                f"/{self.robot_name}_{i}_{self.label}/ArUcos",
                                self.feature_receiver,
                                qos)
                self.features_sub.append(_sub)
        
        #   output files for data storage:
        self.position_d = os.path.join(self.output, f"position_{self.label}.dat")
        with open(self.position_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.vel_d = os.path.join(self.output, f"velocities_{self.label}.dat")
        with open(self.vel_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.norm_e_d = os.path.join(self.output, f"norm_error_{self.label}.dat")
        with open(self.norm_e_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.arucos_d = os.path.join(self.output, f"arUcos_{self.label}.dat")
        with open(self.arucos_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.error_d = [None]*self.n_agents
        self.log_d = [None]*self.n_agents
        for j in range(self.n_agents):
            if j != self.label:
                self.error_d[j] = os.path.join(self.output, f"error_{self.label}_{j}.dat")
                with open(self.error_d[j], 'w') as file:
                    pass  # 'w' mode clears the file's contents
                self.log_d[j] = os.path.join(self.output, f"log_{self.label}.dat")
                with open(self.log_d[j], 'w') as file:
                    pass  # 'w' mode clears the file's contents
        if self.k_int != 0.:
            self.error_int_d = [None]*self.n_agents
            for j in range(self.n_agents):
                if j != self.label:
                    self.error_int_d[j] = os.path.join(self.output, f"error_int_{self.label}_{j}.dat")
                    with open(self.error_int_d[j], 'w') as file:
                        pass  # 'w' mode clears the file's contents

        # output_filename = os.path.join(self.output, f"video_{self.label}.mp4")
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
        # fps = int(self.frequency)
        # self.frame_shape = (480, 856)
        # self.video_writer = cv2.VideoWriter(output_filename, fourcc, fps, self.frame_shape)


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
        self.cv_image = None
        self.error = [None]*self.n_agents
        self._err_int = [None]*self.n_agents
        self.norm = -1.
        if self.k_int != 0:
            self.ids_int = [[] for i in range(self.n_agents)]
            self.err_int = [[] for i in range(self.n_agents)]
            # self.err_int = [np.array([[],[]])]*self.n_agents
            self.control = self.control_int
            self.tick = -1.
            self.tock = -1.

        else:
            self.control = self.control_p
        self.svd = [None]*self.n_agents

        # INIT control loop
        self.timer = self.create_timer(1.0 / self.frequency, self.control_loop)

    # def __exit__(self):
    # def __del__(self):
    #     self.video_writer.release()

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
            return
        if not self.found_arucos_w:
            self.get_logger().warning("ArUcos found in received image")
            self.found_arucos_w = True

        self.ids[self.label] = [ i[0] for i in  ids.tolist()]
        self.p = np.array(corners).reshape((-1,2))
        #   Normalize
        self.points[self.label] = self.normalize(self.p.astype(float).T)

        self.view_corners = corners
        self.view_ids = ids

        # if self.enable_polar:
        #     _r = np.linalg.norm(self.points, axis = 0)
        #     _t = np.arctan2(self.points[1,:], self.points[0,:])
        #     self.points = np.c_[_r, _t].T
        #     _r = np.linalg.norm(self.points_ref, axis = 0)
        #     _t = np.arctan2(self.points_ref[1,:], self.points_ref[0,:])
        #     self.points_ref = np.c_[_r, _t].T


        # #   Publish detection
        # _image = self.cv_image.copy()
        # # print(ids)
        # cv2.aruco.drawDetectedMarkers(_image, corners, ids,
        #                                 borderColor = (0,100,0.) )
        # cv2.aruco.drawDetectedMarkers(_image, self.corners_ref[self.label], self._ids_ref[self.label],
        #                                 borderColor = (100,1.,0.) )
        #
        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(_image, "bgr8"))

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

        if self.norm >= .0:
            with open(self.norm_e_d, 'ab') as f:
                data = (t,self.norm)
                binary = struct.pack('dd', *data)
                f.write(binary)

        with open(self.arucos_d, 'ab') as f:
            for i in range(len(self.ids[self.label])):

                data = (t, self.ids[self.label][i])
                data += tuple(self.p[4*i:4*(i+1), :].reshape(-1))
                binary = struct.pack('didddddddd', *data)
                f.write(binary)

        for j in range(self.n_agents):
            if j != self.label:
                with open(self.error_d[j], 'ab') as f:
                    for i in range(len(self.ids[j])):

                        data = (t, self.ids[j][i])
                        data += tuple(self.error[j][:,4*i:4*(i+1)].T.reshape(-1))
                        diff = 10 - len(data)
                        if diff !=0:
                            data += tuple(np.zeros(diff))
                        binary = struct.pack('didddddddd', *data)
                        f.write(binary)

        #   Integral error
        if self.k_int != 0.:
            for j in range(self.n_agents):
                if  not self._err_int[j] is None:
                    with open(self.error_int_d[j], 'ab') as f:
                        for i in range(len(self.ids[j])):

                            data = (t, self.ids[j][i])
                            data += tuple(self._err_int[j][:,4*i:4*(i+1)].T.reshape(-1))
                            diff = 10 - len(data)
                            if diff !=0:
                                data += tuple(np.zeros(diff))
                            binary = struct.pack('didddddddd', *data)
                            f.write(binary)

        # save log
        if not self.enable_log:
            return
        for j in range(self.n_agents):
            if j != self.label:
                with open(self.log_d[j], 'ab') as f:
                    data = (t,)
                    # data += tuple(self.L.reshape(-1))
                    data += tuple(self.svd[j].reshape(-1))
                    # binary = struct.pack('d'*(1+8*6+6), *data) ## 6 dof y un aruco
                    binary = struct.pack('d'*(1+6), *data) ## 6 dof only singular values
                    # binary = struct.pack('d'*(1+8*4+4), *data) ## 4 dof
                    f.write(binary)

    def feature_receiver(self, msg):

        #   TODO: include depth

        j = msg.j
        _size = msg.size
        _depth = msg.depth
        _points = []
        _ids = []
        for i in range(_size):
            _ids.append(msg.arucos[i].id)
            _points.append(msg.arucos[i].points[0].x)
            _points.append(msg.arucos[i].points[0].y)
            _points.append(msg.arucos[i].points[1].x)
            _points.append(msg.arucos[i].points[1].y)
            _points.append(msg.arucos[i].points[2].x)
            _points.append(msg.arucos[i].points[2].y)
            _points.append(msg.arucos[i].points[3].x)
            _points.append(msg.arucos[i].points[3].y)
        self.ids[j] = _ids

        _points = np.array(_points)
        _points = _points.reshape((-1,2)).astype(float).T
        self.points[j] = self.normalize(_points)

    def get_mathing(self, j):

        # matching_elements = list(set(array1).intersection(set(array2)))
        _query = set(self.ids[self.label])
        _query = _query.intersection(set(self.ids_ref[self.label]))
        _query = _query.intersection(set(self.ids[j]))
        _query = _query.intersection(set(self.ids_ref[j]))
        _query = list(set(_query))

        # self.get_logger().info(f"idx = {_query}")

        if len(_query) == 0:
            return [], [], [], [], []

        match_1 = []
        for q in _query:
            k =  self.ids[self.label].index(q)
            match_1 = match_1 + list(range(k*4, k*4 +4))

        match_2 = []
        for q in _query:
            k =  self.ids[j].index(q)
            match_2 = match_2 + list(range(k*4, k*4 +4))

        match_3 = []
        for q in _query:
            k =  self.ids_ref[self.label].index(q)
            match_3 = match_3 + list(range(k*4, k*4 +4))

        match_4 = []
        for q in _query:
            k =  self.ids_ref[j].index(q)
            match_4 = match_4 + list(range(k*4, k*4 +4))

        return _query, match_1, match_2, match_3, match_4

    def control_p(self, _image = None):
        for j in range(self.n_agents):
            if j != self. label and (not  self.points[j] is None):

                #   mask
                ids, idi, idj, idir, idjr = self.get_mathing(j)

                # self.get_logger().info(f"self.points[{self.label}] ")
                # self.get_logger().info(f"{self.points[self.label]} ")
                points_i = self.points[self.label][:,idi]
                points_j = self.points[j][:,idj]
                points_ref = self.points_ref[self.label][:,idir]
                points_ref_j = self.points_ref[j][:,idjr]

                complement = points_j  + (points_ref - points_ref_j)
                # complement = points_ref
                # complement[1,:] += 0.5
                self.error[j] = points_i - complement
                self.L = interaction_matrix_xyz(complement, self.img_depth)
                # self.L = interaction_matrix_xyz(points_ref, self.img_depth)
                # self.L = interaction_matrix_xyz(points_ref, self.img_depth)
                L_inv = Inv_Moore_Penrose(self.L)

                if self.enable_log:
                    _, self.svd[j], _ = np.linalg.svd(self.L.T @ self.L)

                if L_inv is None:
                    self.get_logger().error("Invalid Ls matrix")
                    continue

                self._u += - self.gain * L_inv @ self.error[j].T.reshape(-1)

                if not _image is None:
                    complement[0,:] = complement[0,:]*self.f[0] + self.pPrinc[0]
                    complement[1,:] = complement[1,:]*self.f[1] + self.pPrinc[1]
                    complement = complement.T.reshape((len(ids), 4,2)).astype(float)
                    complement = tuple(complement[i].reshape((1,4,2)) for i in range(len(ids)))
                    view_ids = np.array(ids)
                    cv2.aruco.drawDetectedMarkers(_image,
                                complement,
                                view_ids,
                                borderColor = (50,1.,0.) )

        #   6DOF
        _w = self.R_cam @ self._u[3:]
        _v = (self.R_cam @ self._u[:3]).reshape(-1)
        _v += np.cross( self.t_cam , _w.reshape(-1) )
        _w *= self.kw
        self.u[:3] = _v.copy()
        self.u[3:] = _w.reshape(-1)
        #   4DOF
        # _w = self.R_cam @ np.array([0.,self._u[3],0.])
        # _v = (self.R_cam @ self._u[:3]).reshape(-1)
        # _v += np.cross( self.t_cam , _w.reshape(-1) )
        # _w *= self.kw
        # self.u[:3] = _v.copy()
        # self.u[3:] = _w.copy()

        return _image

    def get_mathing_int(self, query, j):

        _err = np.zeros((2,4*len(query)))
        for i in range(len(query)):
            q = query[i]
            if q in self.ids_int[j]:
                k =  self.ids_int[j].index(q)
                _err[:,4*i:4*i+4] = self.err_int[j][k]

        return _err

    def error_int(self, query, j):


        self.tick = self.tock
        self.tock = self.get_clock().now().nanoseconds * 1e-9
        dt = self.tock - self.tick

        if self.tick < 0. or self.tock < 0.:
            return

        for i in range(len(query)):
            q = query[i]
            _err = self.error[j][:,i*4: i*4 +4]
            if q in self.ids_int[j]:
                k =  self.ids_int[j].index(q)
                self.err_int[j][k] += _err*dt
                # print(self.err_int[j][k])
                # print(f"Int err from {j} to {self.label} = {_err.reshape(-1)*dt}")
            else:
                self.ids_int[j].append(q)
                self.err_int[j].append(_err)

    # def get_mathing_int(self, query, j):
    #
    #     _err = np.zeros((2,len(query)))
    #     for q in query:
    #         if q in self.ids_int[j]:
    #             k =  self.ids_int[j].index(q)
    #             _err = self.err_int[j][:,k*4: k*4 +4]
    #
    #     return _err
    #
    # def error_int(self, query, dt, j):
    #
    #     for i in range(len(query)):
    #         q = query[i]
    #         _err = self.error[j][:,i*4: i*4 +4]
    #         if q in self.ids_int[j]:
    #             k =  self.ids_int[j].index(q)
    #             self.err_int[j][:,k*4: k*4 +4] += _err*dt
    #         else:
    #             self.ids_int[j].append(q)
    #             self.err_int[j] = np.concatenate((self.err_int[j], _err), axis = 1 )

    def control_int(self, _image = None):
        for j in range(self.n_agents):
            if j != self. label and (not  self.points[j] is None):

                #   mask
                ids, idi, idj, idir, idjr = self.get_mathing(j)
                self._err_int[j] = self.get_mathing_int(ids, j)


                # self.get_logger().info(f"self.points[{self.label}] ")
                # self.get_logger().info(f"{self.points[self.label]} ")
                points_i = self.points[self.label][:,idi]
                points_j = self.points[j][:,idj]
                points_ref = self.points_ref[self.label][:,idir]
                points_ref_j = self.points_ref[j][:,idjr]

                complement = points_j  + (points_ref - points_ref_j)
                # complement = points_ref
                # complement[1,:] += 0.5
                self.error[j] = points_i - complement

                # print(ids, self._err_int[j], self.error[j])
                self.L = interaction_matrix_xyz(complement, self.img_depth)
                # self.L = interaction_matrix_xyz(points_ref, self.img_depth)
                # self.L = interaction_matrix_xyz(points_ref, self.img_depth)
                L_inv = Inv_Moore_Penrose(self.L)

                if self.enable_log:
                    _, self.svd[j], _ = np.linalg.svd(self.L.T @ self.L)

                if L_inv is None:
                    self.get_logger().error("Invalid Ls matrix")
                    continue

                _arg = self.error[j].T.reshape(-1) + self.k_int * self._err_int[j].T.reshape(-1)
                self._u += - self.gain * L_inv @ _arg

                if not _image is None:
                    complement[0,:] = complement[0,:]*self.f[0] + self.pPrinc[0]
                    complement[1,:] = complement[1,:]*self.f[1] + self.pPrinc[1]
                    complement = complement.T.reshape((len(ids), 4,2)).astype(float)
                    complement = tuple(complement[i].reshape((1,4,2)) for i in range(len(ids)))
                    view_ids = np.array(ids)
                    cv2.aruco.drawDetectedMarkers(_image,
                                complement,
                                view_ids,
                                borderColor = (50,1.,0.) )
                self.error_int(ids, j )


        # print(self.err_int)
        #   6DOF
        _w = self.R_cam @ self._u[3:]
        _v = (self.R_cam @ self._u[:3]).reshape(-1)
        _v += np.cross( self.t_cam , _w.reshape(-1) )
        _w *= self.kw
        self.u[:3] = _v.copy()
        self.u[3:] = _w.reshape(-1)
        #   4DOF
        # _w = self.R_cam @ np.array([0.,self._u[3],0.])
        # _v = (self.R_cam @ self._u[:3]).reshape(-1)
        # _v += np.cross( self.t_cam , _w.reshape(-1) )
        # _w *= self.kw
        # self.u[:3] = _v.copy()
        # self.u[3:] = _w.copy()

        return _image

    def control_loop(self):

        if not self.p is None:

            msg = Corners()
            msg.j = int(self.label)
            _arucos = self.p.reshape((-1,8))
            msg.size = int(_arucos.shape[0])
            msg.depth = float(1.)
            # _msg = []
            for i in range(_arucos.shape[0]):
                __msg = ArUco()
                __msg.id = int( self.ids[self.label][i])
                __msg.points[0].x = float(_arucos[i,0])
                __msg.points[0].y = float(_arucos[i,1])
                __msg.points[1].x = float(_arucos[i,2])
                __msg.points[1].y = float(_arucos[i,3])
                __msg.points[2].x = float(_arucos[i,4])
                __msg.points[2].y = float(_arucos[i,5])
                __msg.points[3].x = float(_arucos[i,6])
                __msg.points[3].y = float(_arucos[i,7])
                msg.arucos.append(__msg)
            for i in range(len(self.features_pub)):
                self.features_pub[i].publish(msg)

        _image = None
        if not self.cv_image is None:
            #   Publish detection
            _image = self.cv_image.copy()
            # print(ids)
            cv2.aruco.drawDetectedMarkers(_image, self.view_corners, self.view_ids,
                                            borderColor = (0,100,0.) )
            cv2.aruco.drawDetectedMarkers(_image, self.corners_ref[self.label],
                                          self._ids_ref[self.label],
                                            borderColor = (0,0., 100.) )




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
            if self.new_state == IBFC and self.takeoff_complete:
                self.get_logger().info("State change: IBFC")
                self.state = IBFC
            elif self.new_state == IBFC and  not self.takeoff_complete:
                self.get_logger().info("Waiting for TAKEOFF to finish, can not change to IBFC")
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
            if self.new_state == IBFC and self.init_complete:
                self.get_logger().info("State change: IBFC")
                self.state = IBFC
            elif self.new_state == IBFC and  not self.init_complete:
                self.get_logger().info("Waiting for INITIAL CONDITION to finish, can not change to IBFC")
                self.new_state = INITCOND
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


        elif self.state == IBFC and self.points is None:

            self.get_logger().error("Image error can not be computed")

            if self.data2save:
                self.save_data()

            try:
                self.cmd_pub.publish(self.m_vel)

            except Exception as e:
                self.get_logger().error(f"Error with IBFC control: {str(e)}")
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

        elif self.state == IBFC:

            #   IBFC
            self._u = np.zeros(6)
            if self.k_int != 0. and self.norm < 0.4 and self.norm > 0.:
                _image = self.control_int(_image)

            else:
                _image = self.control_p(_image)
            # _image = self.control(_image)

            _norm = 0.
            for j in range(self.n_agents):
                if j != self.label:
                    _v = self.error[j].reshape(-1)
                    _norm += np.dot(_v,_v)
            self.norm = np.sqrt(_norm)

            self.m_vel.linear.x = float(self.u[0])
            self.m_vel.linear.y = float(self.u[1])
            self.m_vel.linear.z = float(self.u[2])
            self.m_vel.angular.z = float(self.u[5])
            # self.get_logger().info( f"Control_cmd_vel: {self.m_vel.angular.z}")
            self.cmd_pub.publish(self.m_vel)


            #   Save data
            self.save_data()
            self.data2save = True
            # frame = np.full(self.frame_shape+(3,), fill_value=255, dtype=np.uint8)
            # self.video_writer.write(frame)
            # self.video_writer.write(_image)

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

        if not _image is None:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(_image, "bgr8"))
            # self.video_writer.write(_image)


def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
