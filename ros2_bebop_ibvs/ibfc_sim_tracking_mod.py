#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from formation_interfaces.msg import ArUco, Corners
from formation_interfaces.msg import DeltaS

from tf_transformations import quaternion_matrix, euler_from_matrix
from cv_bridge import CvBridge
# from PyQt5.QtGui import QImage
import cv2
import numpy as np
import struct
import os
import yaml

IDLE = 0
IBFC = 1
TAKEOFF = 2
LANDING = 3
STOP = 4
INITCOND = 5
REFERENCE = 6
RESET_TRACKING = 7

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

def custom_draw_matching(image1, image2, points1, points2,
                         color1=(0, 0, 255), color2=(0, 255, 0),
                         point_radius=3, line_thickness = 1):

    _shape = list(image1.shape)
    W = _shape[1]
    _shape[1] *= 2
    _shape = tuple(_shape)
    output_image = np.zeros(_shape, dtype = image1.dtype)
    output_image[:,:W,:] = image1.copy()
    output_image[:,W:,:] = image2.copy()

    # Draw points from the first array
    _points2 = points2.copy()
    _points2 += np.array([W,0.])
    for i in range(points1.shape[0]):
        # Draw the line
        cv2.line(output_image, points1[i,:].astype(int), _points2[i,:].astype(int), color1, line_thickness)

        # Draw points
        cv2.circle(output_image, points1[i,:].astype(int), point_radius, color1, -1)
        cv2.circle(output_image, _points2[i,:].astype(int), point_radius, color2, -1)

    return output_image

class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        
        #   Save data
        self.proc_paramaters()

        #   Load references
        enable_IBVS = self.config_reference()

        #   State
        self.state = IDLE
        self.u = np.zeros(6)
        self._u = np.zeros(6)
        self.new_state = IDLE
        self.current_pose = Pose()
        self.data2save = False
        self.enable = False
        self.takeoff_complete = False  # Nuevo flag para controlar despegue completado
        self.m_vel = Twist()
        self.cv_image = None
        self.m_image = None
        self.error = [None]*self.n_agents
        self._err_int = [None]*self.n_agents
        self.norm = -1.
        self.deltas_self = None
        self.lost_features = False
        self.reset_flag = False
        self.desc_self = None
        self.points = None

        if self.enable_log:
            self.svd = [None]*self.n_agents
            if  self.k_int != 0:
                self.u_log = [np.zeros(6), np.zeros(6)]

        #   Publishers
        qos = QoSProfile(depth=2)
        self.cmd_pub = self.create_publisher(Twist,
                                             f"/{self.robot_name}_{self.label}/cmd_vel",
                                             qos)
        self.cmd_enable = self.create_publisher(Bool,
                                                f"/{self.robot_name}_{self.label}/enable",
                                                qos)

        #   Subscriptions
        self.pos_sub = self.create_subscription(Pose,
                                                f"/{self.robot_name}_{self.label}/pose",
                                                self.pos_changed,
                                                qos)
        # self.state_sub = self.create_subscription(Int32,
        #                                           f"/state_{self.label}",
        #                                           self.state_changed,
        #                                           qos)


        if enable_IBVS:

            #   config control
            if self.k_int != 0:
                self.ids_int = [[] for i in range(self.n_agents)]
                self.err_int = [[] for i in range(self.n_agents)]
                # self.err_int = [np.array([[],[]])]*self.n_agents
                self.tick = -1.
                self.tock = -1.

            #   Config data storage
            self.config_data_storage()

            #   Image bridge
            self.features_sub  = []
            self.features_pub  = []
            img_qos = QoSProfile(depth=2)
            self.bridge = CvBridge()

            self.image_subscription = self.create_subscription(
                Image, f"/{self.robot_name}_{self.label}/image",
                self.image_recv,
                img_qos)
            for i in self.out_neighbors:
                _pub = self.create_publisher(DeltaS,
                                f"/{self.robot_name}_{self.label}_{i}/desc",
                                qos)
                self.features_pub.append(_pub)
            for i in self.in_neighbors:
                _sub = self.create_subscription(DeltaS,
                                f"/{self.robot_name}_{i}_{self.label}/desc",
                                self.delta_receiver,
                                qos)
                self.features_sub.append(_sub)
            # INIT control loop
            self.image_pub = self.create_publisher(Image,
                                                    f"/{self.robot_name}_{self.label}/matching",
                                                   img_qos)
            self.state_sub = self.create_subscription(Int32,
                                                  f"/state_{self.label}",
                                                  self.state_changed_ibvs,
                                                  qos)
        else:
            self.state_sub = self.create_subscription(Int32,
                                                  f"/state_{self.label}",
                                                  self.state_changed_simple,
                                                  qos)

            self.get_logger().warning(f"{self.label}: Control configuration incomplete, simple control enabled")
            # self.timer = self.create_timer(1.0 / self.frequency, self.open_loop)

        self.state = self.s_idle
        self.timer = self.create_timer(1.0 / self.frequency, self.control_loop)








    def proc_paramaters(self):

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
        self.declare_parameter('L', [0])
        self.declare_parameter('CamR', [1.]*9)
        self.declare_parameter('CamT', [1.]*9)
        self.declare_parameter('tracker', ["NAN"])
        self.declare_parameter('tracker_vals', [0.])
        self.declare_parameter('aruco_dictionary', "")
        self.declare_parameter('p0', [1.]*4)
        self.declare_parameter('pd', [1.]*4)
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
        self.L = self.get_parameter('L').value
        self.camR = self.get_parameter('CamR').value
        self.camT = self.get_parameter('CamT').value
        tracker = self.get_parameter('tracker').value
        tracker_vals = self.get_parameter('tracker_vals').value
        self.aruco_dictionary = self.get_parameter('aruco_dictionary').value
        self.initial_cond = self.get_parameter('p0').value
        self.reference_pose = self.get_parameter('pd').value
        self.enable_polar = self.get_parameter('polar').value
        self.enable_log = self.get_parameter('save_log').value

        if not self.robot_name:
            self.get_logger().info('Empty "robot_name": Setting "bebop" as default.')
            self.robot_name = 'bebop'
        # self.get_logger().info(f"Robot Name: {self.robot_name}_{self.label}")

        if tracker[0] == "NAN" :
            self.tracker = {}
        else:
            self.tracker = {i:j for i, j in zip(tracker,tracker_vals)}

        # Convert parameters to a dictionary
        param_dict = {
            "frequency" : self.frequency,
            "robot_name" : self.robot_name,
            "takeoff_threshold" : self.takeoff_threshold,
            "landing_threshold" : self.landing_threshold,
            "takeoff_height" : self.takeoff_height,
            "label" : self.label,
            "n_agents" : self.n_agents,
            "reference_image_prefix" : self.reference_image_prefix,
            "output" : self.output,
            "img_depth" : self.img_depth,
            "gain" : self.gain,
            "kw" : self.kw,
            "k_int" : self.k_int,
            "gain_takeoff" : self.gain_takeoff,
            "K" : self.K,
            "L" : self.L,
            "initial_cond" : self.initial_cond,
            "reperence_pose" : self.reference_pose,
            "enable_polar" : self.enable_polar,
            "enable_log" : self.enable_log,
            }


        # Save to YAML file
        _name = os.path.join(self.output, f"params.yaml")
        with open(_name, 'w') as yaml_file:
            yaml.dump(param_dict, yaml_file)



        #   inital conditions
        #   TODO: simplify
        self.initial_cond =  np.array(self.initial_cond)
        self.initial_cond = self.initial_cond.reshape((-1,4))
        self.initial_cond = self.initial_cond[self.label].reshape(-1)
        self.reference_pose =  np.array(self.reference_pose)
        self.reference_pose = self.reference_pose.reshape((-1,4))

        #   TODO: generalizar
        _R = np.array([[0., -1., 0.],
                       [-1., 0., 0.],
                       [0., 0., -1.]])
        self.pref = self.reference_pose - self.reference_pose[self.label,:]

        self.pref[:,:3] = -(_R @ self.pref[:,:3].T).T
        self.pref[:,:3] /= self.img_depth
        self.pref[self.pref[:,3] < -np.pi, 3] += np.pi
        self.pref[self.pref[:,3] >  np.pi, 3] -= np.pi
        print(self.pref)

        self.reference_pose = self.reference_pose[self.label]


        #   Camera calibration data
        self.f = [self.K[0], self.K[4]]
        self.pPrinc = [self.K[2],self.K[5]]
        self.K = np.array(self.K).reshape((3,3))
        self.R_cam = np.array(self.camR).reshape((3,3))
        self.t_cam = np.array(self.camT)

        #   Graph Laplacian
        if len(self.L) != self.n_agents**2 :
            self.get_logger().info('Empty "robot_name": Setting "bebop" as default.')
            self.L = np.ones((self.n_agents,self.n_agents)) - np.eye(self.n_agents)
        else:
            self.L = np.array(self.L).reshape((-1,self.n_agents))
        _neighbors = self.L[self.label,:].tolist()
        self.in_neighbors = [i for i in range(len(_neighbors)) if _neighbors[i]]
        _neighbors = self.L[:,self.label].tolist()
        self.out_neighbors = [i for i in range(len(_neighbors)) if _neighbors[i]]



    def config_data_storage(self):

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
        self.error_d = [None]*self.n_agents
        for j in self.in_neighbors:
            self.error_d[j] = os.path.join(self.output, f"error_{self.label}_{j}.dat")
            with open(self.error_d[j], 'w') as file:
                pass  # 'w' mode clears the file's contents

        self.features_d = os.path.join(self.output, f"features_{self.label}.dat")
        with open(self.features_d, 'w') as file:
            pass  # 'w' mode clears the file's contents

        if self.enable_log:
            self.log_d = [None]*self.n_agents
            for j in self.in_neighbors:
                self.log_d[j] = os.path.join(self.output, f"log_{self.label}.dat")
                with open(self.log_d[j], 'w') as file:
                    pass  # 'w' mode clears the file's contents
            if self.k_int != 0.:
                self.vel_log_d_0 = os.path.join(self.output, f"log_vel_prop_{self.label}.dat")
                with open(self.vel_log_d_0, 'w') as file:
                    pass  # 'w' mode clears the file's contents
                self.vel_log_d_1 = os.path.join(self.output, f"log_vel_int_{self.label}.dat")
                with open(self.vel_log_d_1, 'w') as file:
                    pass  # 'w' mode clears the file's contents

        if self.k_int != 0.:
            self.error_int_d = [None]*self.n_agents
            for j in self.in_neighbors:
                self.error_int_d[j] = os.path.join(self.output, f"error_int_{self.label}_{j}.dat")
                with open(self.error_int_d[j], 'w') as file:
                    pass  # 'w' mode clears the file's contents

    def config_reference(self):

        if len(self.tracker) == 0:
            return False

        self.orb = cv2.ORB_create(
            nfeatures    = int( self.tracker["nfeatures"] ),
            scaleFactor  = self.tracker["scaleFactor"],
            nlevels      = int( self.tracker["nlevels"] ),
            edgeThreshold= int( self.tracker["edgeThreshold"] ),
            patchSize    = int( self.tracker["patchSize"] ),
            fastThreshold= int( self.tracker["fastThreshold"] )
            )

        #   Matcher
        index_params = {
            "algorithm": 6,
            "table_number": 20,
            "key_size": 10,
            "multi_probe_level": 2,
        }

        self.flann = cv2.FlannBasedMatcher(index_params)

        self.lk_params = dict(winSize=(15, 15),
                            maxLevel=2,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # self.ids_ref = [None]* self.n_agents
        # self._ids_ref = [None]* self.n_agents
        # self.points_ref = [None]* self.n_agents
        # self.corners_ref = [None]* self.n_agents
        # self.ids = [None]*self.n_agents
        self.desc = [None]*self.n_agents
        self.deltas = [None]*self.n_agents
        self.ids_save = [None]*self.n_agents
        # self.p = [None]*self.n_agents
        self.get_logger().info(f"{self.label}: Reading Image")
        self.image_ref = cv2.imread(f"{self.reference_image_prefix}_{self.label}.png")
        if  self.image_ref is None :
            self.get_logger().error(f"Image {self.reference_image_prefix}_{self.label}.png could not be read ")
            return False
        gray_image = cv2.cvtColor(self.image_ref, cv2.COLOR_BGR2GRAY)
        self.kp_ref, self.desc_ref = self.orb.detectAndCompute(gray_image, None)
        if self.desc_ref is None:
            self.get_logger().error(f"No detected Features ")
            return False

        self.get_logger().info(str(self.desc_ref.shape))

        self.prev_image = np.zeros((gray_image.shape),
                                    dtype = gray_image.dtype)
        self.p = np.zeros((2,2), dtype = np.float32)
        self.deltas_self = np.zeros((2,2), dtype = np.float32)
        self.ids = np.zeros(2, dtype = np.float32)
        self.desc_masked = np.zeros((2,2), dtype = np.int8)
        self.match_threshold = self.tracker["matcher_threshold"]


        return True


    def state_changed_ibvs(self, msg):
        if msg.data == RESET_TRACKING:
            self.reset_flag = True
            return
        self.new_state = msg.data
    def state_changed_simple(self, msg):
        if msg.data == IBFC:
            return
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

        gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

        #   Detect
        # Extract Matches if not enough
        kp, self.desc_self = self.orb.detectAndCompute(gray_image, None)

        if len(kp) <4:
            return

        #  Kp conversion
        self.p = np.float32([k.pt for k in kp  ])
        self.points = self.normalize(self.p.astype(float).T)

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

        if not self.deltas is None:

            with open(self.features_d, 'ab') as f:
                for i, m in enumerate(self.ids):

                    data = (t, m)
                    data += tuple(self.p[i, :].reshape(-1))
                    binary = struct.pack('didd', *data)
                    f.write(binary)

            for j in self.in_neighbors:
                if self.ids_save[j] is None:
                    continue
                with open(self.error_d[j], 'ab') as f:
                    for i, m in enumerate( self.ids_save[j]):

                        data = (t, m)
                        data += tuple(self.error[j][:,i].T.reshape(-1))
                        binary = struct.pack('didd', *data)
                        f.write(binary)



    def delta_receiver(self, msg):

        #   TODO: include depth

        j = msg.j
        _depth = msg.depth

        _desc = np.array(msg.desc.data, dtype= self.desc_ref.dtype )
        _desc = _desc.reshape((msg.desc.rows, msg.desc.cols))

        _deltas = np.array(msg.deltas.data, dtype= np.float32 )
        _deltas = _deltas.reshape((msg.deltas.rows, msg.deltas.cols))

        self.desc[j] = _desc
        self.deltas[j] = _deltas




    def control(self):
        _n = 0.
        for j in self.in_neighbors:
            if self.deltas[j] is None:
                continue

            #   match
            knn_matches = self.flann.knnMatch(self.desc[j],
                                              self.desc_self, k=2)
            # Lowe ratio test
            good_matches = []
            for matches in knn_matches:
                if len(matches) == 2:
                    m, n = matches
                    if m.distance < self.tracker["flann_ratio"] * n.distance:
                        good_matches.append(m)

            if len(good_matches) <= 4:
                self.get_logger().warning("No Neighboring Matches available")
                continue


            # else:
            # self.get_logger().info("Neighboring Matches available")

            # _p_i = np.float32([
            #     self.points[:,m.trainIdx]
            #     for m in good_matches
            # ]).T
            # _pr_i = np.float32([
            #     self.points_ref[:,m.trainIdx]
            #     for m in good_matches
            # ]).T

            _delta_i = np.float32([
                self.points[:,m.trainIdx]
                for m in good_matches
            ])
            _delta_j = np.float32([
                self.deltas[j][:,m.queryIdx]
                for m in good_matches
            ])


            _, _mask = cv2.findHomography(_delta_i, _delta_j, cv2.RANSAC)
            _mask = _mask.reshape(-1)

            if _mask.sum() <= 4:
                self.get_logger().warning("No Neighboring Matches available (RANSAC)")
                continue

            _delta_i = _delta_i[_mask == 1,:].T
            _delta_j = _delta_j[_mask == 1,:].T
            _delta_i = _delta_i.reshape((2,-1))
            _delta_j = _delta_j.reshape((2,-1))

            # self.ids_save[j] = np.int8([
            #     self.ids[m.trainIdx]
            #     for m in good_matches
            # ]).T

            # complement =  _delta_i + 2.*self.pref[j,:2].reshape((2,1))
            # print(_delta_j.shape, _delta_i.shape, self.pref[j,:2].reshape((2,1)).shape)
            complement =  _delta_j - 1.*self.pref[j,:2].reshape((2,1))
            self.error[j] = complement - _delta_i
            # self.error[j] = _delta_j - _delta_i - .1*self.pref[j,:2].reshape((2,1))
            self.L = interaction_matrix_xyz(_delta_i, self.img_depth)
            # self.L = interaction_matrix_xyz(complement, self.img_depth)
            # self.L = interaction_matrix_xyz(_p_i, self.img_depth)
            # self.L = interaction_matrix_xyz(_p_i, self.img_depth)

            #   BEGIN TEST
            # self.error[j] = _delta_i
            # self.L = interaction_matrix_xyz(_pr_i, self.img_depth)
            # self.error[j] = _delta_i
            # self.L = interaction_matrix_xyz(_p_i, self.img_depth)
            # self.error[j] = self.deltas_self
            # self.L = interaction_matrix_xyz(self.points_ref, self.img_depth)
            #   END TEST

            L_inv = Inv_Moore_Penrose(self.L)

            if self.enable_log:
                _, self.svd[j], _ = np.linalg.svd(self.L.T @ self.L)

            if L_inv is None:
                self.get_logger().error("Invalid Ls matrix")
                continue

            self._u +=  self.gain * L_inv @ self.error[j].T.reshape(-1)
            _n += 1.

            # TODO image draw
            if self.m_image is None:
                continue


            complement =  np.vstack([complement, np.ones(_delta_i.shape[1])])
            complement = self.K @ complement
            complement = complement[:2,:] / complement[2,:]
            m_delta_i =  np.vstack([_delta_i, np.ones(_delta_i.shape[1])])
            m_delta_i = self.K @ m_delta_i
            m_delta_i = m_delta_i[:2,:] / m_delta_i[2,:]
            # print(m_delta_i, complement)
            self.custom_draw_matching(m_delta_i.T,
                        complement.T,
                        color2 = (0,124,int(255*j / self.n_agents))) # TODO: color i

        # # BEGIN DEBUG
        # if self.label != 0:
        #     self.u = np.zeros(6)
        #     return
        # # END DEBUG

        if _n != 0:
            self.get_logger().info(f"Neig:{_n}")
            self._u /= _n
        # self._u = np.zeros(6)
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

        return



    def send_points(self):
        if self.desc_self is None:
            return
        if self.points is None:
            return

        msg = DeltaS()
        msg.j = int(self.label)
        msg.depth = float(1.)
        # _msg = []

        msg.desc.rows = int(self.desc_self.shape[0])
        msg.desc.cols = int(self.desc_self.shape[1])
        msg.desc.data = self.desc_self.ravel().tolist()

        msg.deltas.rows = int(self.points.shape[0])
        msg.deltas.cols = int(self.points.shape[1])
        msg.deltas.data = self.points.ravel().tolist()

        for i in range(len(self.features_pub)):
            self.features_pub[i].publish(msg)


    def preproc_image(self):
        if  self.cv_image is None:
            return None
        self.m_image = self.cv_image.copy()

    def custom_draw_matching(self, points1, points2,
                         color1=(0, 0, 255), color2=(0, 255, 0),
                         point_radius=3, line_thickness = 1):

        for i in range(points1.shape[0]):
            # Draw the line
            cv2.line(self.m_image, points1[i,:].astype(int), points2[i,:].astype(int), color2, line_thickness)

            # Draw points
            cv2.circle(self.m_image, points1[i,:].astype(int), point_radius, color1, -1)
            cv2.circle(self.m_image, points2[i,:].astype(int), int(0.5*point_radius), color2, -1)


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

        if self.new_state == IBFC and self.takeoff_complete:
            self.get_logger().info("State change: IBFC")
            self.state = self.s_ibvs
            return
        if self.new_state == IBFC and  not self.takeoff_complete:
            self.get_logger().info("Waiting for TAKEOFF to finish, can not change to IBFC")
            self.new_state = TAKEOFF
            return
        if self.new_state == LANDING:
            self.get_logger().info("State change: LANDING")
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
        if self.new_state == IBFC and self.init_complete:
            self.get_logger().info("State change: IBFC")
            self.state = self.s_ibvs
            return
        if self.new_state == IBFC and  not self.init_complete:
            self.get_logger().info("Waiting for INITIAL CONDITION to finish, can not change to IBFC")
            self.new_state = INITCOND
            return
        if self.new_state == LANDING:
            self.get_logger().info("State change: LANDING")
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
        if self.new_state == IBFC and self.init_complete:
            self.get_logger().info("State change: IBFC")
            self.state = self.s_ibvs
            return
        if self.new_state == IBFC and  not self.init_complete:
            self.get_logger().info("Waiting for REFERENCE CONDITION to finish, can not change to IBFC")
            self.new_state = REFERENCE
            return
        if self.new_state == LANDING:
            self.get_logger().info("State change: LANDING")
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
            self.enable = False
            self.cmd_enable.publish(Bool(data=self.enable))
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

    def s_ibvs(self):

        if self.points is None:
            self.get_logger().error("Image error can not be computed")
            # if self.data2save:
            #     self.save_data()
        else:
            #   IBFC
            self._u = np.zeros(6)
            # if self.k_int != 0. and self.norm < 0.4 and self.norm > 0.:
            #     _image = self.control_int(_image)
            #
            # else:
            #     _image = self.control_p(_image)
            self.control()

            _norm = 0.
            for j in self.in_neighbors:
                if self.error[j] is None:
                    continue
                _v = self.error[j].reshape(-1)
                _norm += np.dot(_v,_v)
            self.norm = np.sqrt(_norm)

            self.m_vel.linear.x = float(self.u[0])
            self.m_vel.linear.y = float(self.u[1])
            self.m_vel.linear.z = float(self.u[2])
            self.m_vel.angular.z = float(self.u[5])
            # self.get_logger().info( f"Control_cmd_vel: {self.m_vel.angular.z}")
            # self.cmd_pub.publish(self.m_vel)
            self.data2save = True
            # self.save_data()

        try:
            self.cmd_pub.publish(self.m_vel)
        except Exception as e:
            self.get_logger().error(f"Error with IBFC control: {str(e)}")
            self.enable = False
            self.cmd_enable.publish(Bool(data=self.enable))
            #   Save data
            self.save_data()
            # self.data2save = True
            # frame = np.full(self.frame_shape+(3,), fill_value=255, dtype=np.uint8)
            # self.video_writer.write(frame)
            # self.video_writer.write(_image)

        #   Change state
        if self.new_state == LANDING:
            self.get_logger().info("State change: LANDING")
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
        self.enable = False
        self.cmd_enable.publish(Bool(data=self.enable))
        self.get_logger().info("State change: IDLE")
        self.state = self.s_idle

    def control_loop(self):

        #   Publish descriptor messages
        self.send_points()
        #   Preprocess matching points image
        self.preproc_image()
        #   Exec state
        self.state()
        #   Publish matching image
        if self.m_image is None:
            return

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.m_image, "bgr8"))
        # self.video_writer.write(self.m_image)


def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
