#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
# from formation_interfaces.msg import ArUco, Corners
from formation_interfaces.msg import DeltaS

from tf_transformations import quaternion_matrix, euler_from_matrix
from cv_bridge import CvBridge
# from PyQt5.QtGui import QImage
import cv2
import numpy as np
import struct
import os
import yaml

#   Custom
from .my_classes import *



class Controller(State, FeatureTracker):

    def __init__(self):
        super().__init__('Controller')
        
        #   Save data
        enable_IBVS = self.proc_paramaters()
        self.config_reference()


        #   State
        self.u = np.zeros(6)
        self._u = np.zeros(6)
        self.data2save = False
        self.cv_image = None
        self.m_image = None
        self.error = [None]*self.n_agents
        self._err_int = [None]*self.n_agents
        self.norm = -1.
        self.lost_features = False
        self.desc = [None]*self.n_agents
        self.deltas = [None]*self.n_agents
        self.ids_save = [None]*self.n_agents

        if self.enable_log:
            self.svd = [None]*self.n_agents
            if  self.k_int != 0:
                self.u_log = [np.zeros(6), np.zeros(6)]

        #   Publishers
        qos = QoSProfile(depth=2)
        self.create_publishers(qos)


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

        self.timer = self.create_timer(1.0 / self.frequency, self.control_loop)








    def proc_paramaters(self):

        self.declare_parameter('frequency', 50.0)
        self.declare_parameter('robot_name', 'bebop')

        self.declare_parameter('label', 1)
        self.declare_parameter('n_agents', 1)
        self.declare_parameter('reference_image_prefix', "reference_f")
        self.declare_parameter('output', "output")
        self.declare_parameter('img_depth', 1.)
        self.declare_parameter('gain', 1.)
        self.declare_parameter('gain_int', 0.)
        self.declare_parameter('gain_w', 1.)

        self.declare_parameter('L', [0])
        self.declare_parameter('CamR', [1.]*9)
        self.declare_parameter('CamT', [1.]*9)
        self.declare_parameter('p0', [1.]*4)
        self.declare_parameter('pd', [1.]*4)
        self.declare_parameter('polar', False)
        self.declare_parameter('save_log', False)

        #   Config tracker


        self.frequency = self.get_parameter('frequency').value
        self.robot_name = self.get_parameter('robot_name').value.strip()

        self.label = self.get_parameter('label').value
        self.n_agents = self.get_parameter('n_agents').value
        self.reference_image_prefix = self.get_parameter('reference_image_prefix').value
        self.output = self.get_parameter('output').value
        self.img_depth = self.get_parameter('img_depth').value
        self.gain = self.get_parameter('gain').value
        self.kw = self.get_parameter('gain_w').value
        self.k_int = self.get_parameter('gain_int').value

        # self.K = self.get_parameter('K').value
        self.L = self.get_parameter('L').value
        self.camR = self.get_parameter('CamR').value
        self.camT = self.get_parameter('CamT').value
        self.initial_cond = self.get_parameter('p0').value
        self.reference_pose = self.get_parameter('pd').value
        self.enable_polar = self.get_parameter('polar').value
        self.enable_log = self.get_parameter('save_log').value

        self.get_logger().info(f"robot_name: {self.robot_name}_{self.label}")

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

        #   Frame transformation robot-camera
        self.R_cam = np.array(self.camR).reshape((3,3))
        self.t_cam = np.array(self.camT)

        #   inital conditions
        if len(self.reference_pose) != 4*self.n_agents:
            return False
        if len(self.initial_cond) != 4*self.n_agents:
            return False
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

        return True

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
        self.error_d[self.label] = os.path.join(self.output, f"error_{self.label}.dat")
        with open(self.error_d[self.label], 'w') as file:
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
        #     if self.k_int != 0.:
        #         self.vel_log_d_0 = os.path.join(self.output, f"log_vel_prop_{self.label}.dat")
        #         with open(self.vel_log_d_0, 'w') as file:
        #             pass  # 'w' mode clears the file's contents
        #         self.vel_log_d_1 = os.path.join(self.output, f"log_vel_int_{self.label}.dat")
        #         with open(self.vel_log_d_1, 'w') as file:
        #             pass  # 'w' mode clears the file's contents
        #
        # if self.k_int != 0.:
        #     self.error_int_d = [None]*self.n_agents
        #     for j in self.in_neighbors:
        #         self.error_int_d[j] = os.path.join(self.output, f"error_int_{self.label}_{j}.dat")
        #         with open(self.error_int_d[j], 'w') as file:
        #             pass  # 'w' mode clears the file's contents


    def state_changed_ibvs(self, msg):
        if msg.data == RESETVIS:
            self.reset_tracking = True
            return
        self.new_state = msg.data

    def state_changed_simple(self, msg):
        if msg.data == CONTROL:
            return
        self.new_state = msg.data
        


    def image_recv(self, msg):

        # self.get_logger().info("Image received")

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
            return
        except KeyError as e:
            self.get_logger().error(f"Robot name not found in topic: {e}")
            return
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")
            return

        self.img_proc(self.cv_image)


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
            if not self.ids is None:
                with open(self.error_d[self.label], 'ab') as f:
                    for i, m in enumerate( self.ids):

                        data = (t, m)
                        data += tuple(self.error[self.label][:,i].T.reshape(-1))
                        binary = struct.pack('didd', *data)
                        f.write(binary)
        #


    def delta_receiver(self, msg):

        #   TODO: include depth
        if self.desc_self is None:
            return

        j = msg.j
        _depth = msg.depth

        _desc = np.array(msg.desc.data, dtype= self.desc_self.dtype )
        _desc = _desc.reshape((msg.desc.rows, msg.desc.cols))

        _deltas = np.array(msg.deltas.data, dtype= np.float32 )
        _deltas = _deltas.reshape((msg.deltas.rows, msg.deltas.cols))

        self.desc[j] = _desc
        self.deltas[j] = _deltas


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

    def control(self):
        _n = 0.
        self._u = np.zeros(6)
        # self._u = np.zeros(4)
        mismatch = len(self.in_neighbors)
        for j in self.in_neighbors:
            if self.deltas[j] is None:
                continue

            _ret = self.match(self.desc[j], self.deltas[j])
            if _ret is None:
                mismatch -= 1
                self.get_logger().warning(f"Not enough matchings in neighbors ({self.label}-{j})")
                continue

            _delta_i, _delta_j, idx = _ret
            complement =  _delta_j - 1.*self.pref[j,:2].reshape((2,1))
            self.error[j] = complement - _delta_i
            # self.error[j] = _delta_j - _delta_i - .1*self.pref[j,:2].reshape((2,1))
            # self.L = interaction_matrix_z(_delta_i, self.img_depth)
            self.L = interaction_matrix_xyz(_delta_i, self.img_depth)
            # self.L = interaction_matrix_xyz(complement, self.img_depth)
            # self.L = interaction_matrix_xyz(_p_i, self.img_depth)
            # self.L = interaction_matrix_xyz(_p_i, self.img_depth)

            self.ids_save[j] = [self.ids[k] for k in idx]

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

            if self.label == 0:
                self.get_logger().info(str(self.error[j]))

            # TODO image draw
            if self.m_image is None:
                continue

            self.custom_draw_matching(self.m_image,
                        # m_delta_i.T,
                        # complement.T,
                        _delta_i,
                        complement,
                        color2 = (0,124,int(255*j / self.n_agents)),
                        reproject = True)

        if mismatch == 0:
            self.reset_tracking = True
            self.u = np.zeros(6)
            return

        # # BEGIN DEBUG
        # if self.label != 0:
        #     self.u = np.zeros(6)
        #     return

        # if _n != 0:
        #     self.get_logger().info(f"Neig:{_n}")
        #     self._u /= _n
        # self._u = np.zeros(6)
        # # END DEBUG
        #   6DOF
        _w = self.R_cam @ self._u[3:]
        _v = (self.R_cam @ self._u[:3]).reshape(-1)
        _v += np.cross( self.t_cam , _w.reshape(-1) )
        _w *= self.kw
        self.u[:3] = _v.copy()
        self.u[3:] = _w.reshape(-1)
        #   4DOF
        # _w = self.R_cam @ np.array([0.,self._u[3],0.])
        # _w = self.R_cam @ np.array([0.,0.,self._u[3]])
        # _v = (self.R_cam @ self._u[:3]).reshape(-1)
        # _v += np.cross( self.t_cam , _w.reshape(-1) )
        # _w *= self.kw
        # self.u[:3] = _v.copy()
        # self.u[3:] = _w.copy()

        return

    def control2(self):
        _n = 0.
        self.error[self.label] = np.zeros(self.points.shape)
        mismatch = len(self.in_neighbors)
        # self.L = interaction_matrix_xyz(self.points, self.img_depth)
        # L_inv = Inv_Moore_Penrose(self.L)
        # if L_inv is None:
        #     self.get_logger().error("Invalid Ls matrix")
        #     self._u = np.zeros(6)
        #     self.u = np.zeros(6)
        #     return

        if self.enable_log:
            _, self.svd[j], _ = np.linalg.svd(self.L.T @ self.L)

        for j in self.in_neighbors:
            if self.deltas[j] is None:
                continue

            _ret = self.match(self.desc[j], self.deltas[j])
            if _ret is None:
                mismatch -= 1
                self.get_logger().warning(f"Not enough matchings in neighbors ({self.label}<-{j})")
                continue

            _delta_i, _delta_j, idx = _ret
            complement =  _delta_j - 1.*self.pref[j,:2].reshape((2,1))
            self.error[j] = complement - _delta_i
            self.error[self.label][:,idx] += self.error[j]
            self.ids_save[j] = [self.ids[k] for k in idx]

            # # BEGIN debug
            # if self.label == 0:
            #     self.get_logger().info(str(self.error[j]))
            #     self.get_logger().info(str(idx))
            #     self.get_logger().info(str(self.error[self.label]))
            #
            # # END debug

            _n += 1.

            # TODO image draw
            if self.m_image is None:
                continue

            self.custom_draw_matching(self.m_image,
                        # m_delta_i.T,
                        # complement.T,
                        _delta_i,
                        complement,
                        color1 = (0,200,0),
                        color2 = (0,124,int(255*j / self.n_agents)),
                        reproject = True)

        if mismatch == 0:
            self.reset_tracking = True
            self._u = np.zeros(6)
            self.u = np.zeros(6)
            return

        mask = np.logical_or(self.error[self.label][0,:] !=0.,
                             self.error[self.label][1,:] !=0.)

        self.L = interaction_matrix_xyz(self.points[:,mask], self.img_depth)
        L_inv = Inv_Moore_Penrose(self.L)
        if L_inv is None:
            self.get_logger().error("Invalid Ls matrix")
            self._u = np.zeros(6)
            self.u = np.zeros(6)
            return
        self._u = self.gain * L_inv @ self.error[self.label][:,mask].T.reshape(-1)
        # BEGIN debug
        # if self.label == 0:
        #     self.get_logger().info(str(self.error[self.label][:,mask]))

            # END debug

        # # # BEGIN DEBUG
        # if self.label != 0:
        #     self.u = np.zeros(6)
        #     return
        #
        # if _n != 0:
        #     self.get_logger().info(f"Neig:{_n}")
        #     self._u /= _n
        # self._u = np.zeros(6)
        # # # END DEBUG
        # 6DOF
        _w = self.R_cam @ self._u[3:]
        _v = (self.R_cam @ self._u[:3]).reshape(-1)
        _v += np.cross( self.t_cam , _w.reshape(-1) )
        _w *= self.kw
        self.u[:3] = _v.copy()
        self.u[3:] = _w.reshape(-1)
        #   4DOF
        # # _w = self.R_cam @ np.array([0.,self._u[3],0.])
        # _w = self.R_cam @ np.array([0.,0.,self._u[3]])
        # _v = (self.R_cam @ self._u[:3]).reshape(-1)
        # _v += np.cross( self.t_cam , _w.reshape(-1) )
        # _w *= self.kw
        # self.u[:3] = _v.copy()
        # self.u[3:] = _w.copy()

        return

    def s_control(self):

        if self.points is None:
            self.get_logger().error("Image error can not be computed")
            if self.data2save:
                self.save_data()
        else:
            #   IBFC
            # self._u = np.zeros(6)
            # if self.k_int != 0. and self.norm < 0.4 and self.norm > 0.:
            #     _image = self.control_int(_image)
            #
            # else:
            #     _image = self.control_p(_image)
            # self.control()
            self.control2()

            _norm = 0.
            for j in self.in_neighbors:
                if self.error[j] is None:
                    continue
                _v = self.error[j].reshape(-1)
                _norm += np.dot(_v,_v) / float(_v.shape[0])
            self.norm = np.sqrt(_norm)
            self.data2save = True
            self.save_data()

        super().s_control()


    def control_loop(self):

        #   Publish descriptor messages
        self.send_points()
        #   Preprocess matching points image
        self.preproc_image()
        #   Exec state
        if len(self.p) > 2:
            self.custom_draw(self.m_image, self.p.T)
            # self.get_logger().info(f"IBVS tracking {len(self.p)}")
            # self.get_logger().info(f"IBVS tracking {str(self.p)}")
        self.state()
        #   Publish matching image
        if self.m_image is None:
            return
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.m_image, "bgr8"))
        #   Save video frame
        # self.video_writer.write(self.m_image)


def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
