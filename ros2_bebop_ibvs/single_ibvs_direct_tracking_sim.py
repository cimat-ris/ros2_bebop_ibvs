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
import cv2
import numpy as np
import struct
import os
import yaml

#   Custom
from .my_classes import *



class Controller( FeatureTracker):

    def __init__(self):
        super().__init__('Controller')
        
        #   Save data
        enable_IBVS = self.proc_paramaters()

        #   State
        self.u = np.zeros(6)
        _u = np.zeros(6)
        self.data2save = False
        self.cv_image = None
        self.m_image = None
        self.error = None
        self.norm = 200.
        self.lost_features = False
        self.ids_save = None
        self.int_act = False
        self.error_counter = 0
        self.error_counter_th = 200
        self.err_int = None

        if self.enable_log:
            self.svd = [None]*self.n_agents
            if  self.k_int != 0:
                self.u_log = [np.zeros(6), np.zeros(6)]

        #   Video
        v_name = os.path.join(self.output,"matching")
        self.video_config(v_name)

        #   Publishers
        qos = QoSProfile(depth=2)
        self.create_publishers(qos, self.robot_name)


        if enable_IBVS:

            #   config control
            if self.k_int != 0:
                self.control_ = self.control_int
            else:
                self.control_ = self.control

            #   Config data storage
            self.config_data_storage()

            #   Image bridge
            img_qos = QoSProfile(depth=2)
            self.bridge = CvBridge()

            self.image_subscription = self.create_subscription(
                Image, f"/{self.robot_name}/image",
                self.image_recv,
                img_qos)

            # INIT control loop
            self.image_pub = self.create_publisher(Image,
                                                    f"/{self.robot_name}/matching",
                                                   img_qos)
            self.state_sub = self.create_subscription(Int32,
                                                  f"/state",
                                                  self.state_changed_ibvs,
                                                  qos)
        else:
            self.state_sub = self.create_subscription(Int32,
                                                  f"/state",
                                                  self.state_changed_simple,
                                                  qos)

            self.get_logger().warning(f"Control configuration incomplete, simple control enabled")

        self.timer = self.create_timer(1.0 / self.frequency, self.control_loop)








    def proc_paramaters(self):

        self.declare_parameter('frequency', 50.0)
        self.declare_parameter('robot_name', 'bebop')

        self.declare_parameter('ref_image', "ref_image")
        self.declare_parameter('output', "output")
        self.declare_parameter('img_depth', 1.)
        self.declare_parameter('gain', 1.)
        self.declare_parameter('gain_integral', 0.)
        self.declare_parameter('th_integral', 0.1)
        self.declare_parameter('gain_w', 1.)

        self.declare_parameter('p0', [1.]*4)
        self.declare_parameter('pd', [1.]*4)
        self.declare_parameter('polar', False)
        self.declare_parameter('save_log', False)
        self.declare_parameter('enable_video', True)
        self.declare_parameter('tanh', False)

        #   Config tracker


        self.frequency = self.get_parameter('frequency').value
        self.robot_name = self.get_parameter('robot_name').value.strip()

        self.ref_name = self.get_parameter('ref_image').value
        self.output = self.get_parameter('output').value
        self.img_depth = self.get_parameter('img_depth').value
        self.gain = self.get_parameter('gain').value
        self.kw = self.get_parameter('gain_w').value
        self.k_int = self.get_parameter('gain_integral').value
        self.th_int = self.get_parameter('th_integral').value

        self.enable_tanh = self.get_parameter('tanh').value

        self.initial_cond = self.get_parameter('p0').value
        self.reference_pose = self.get_parameter('pd').value
        self.enable_polar = self.get_parameter('polar').value
        self.enable_log = self.get_parameter('save_log').value
        self.enable_video = self.get_parameter('enable_video').value

        self.get_logger().info(f"robot_name: {self.robot_name}")


        #   inital conditions
        if len(self.reference_pose) != 4:
            return False
        if len(self.initial_cond) != 4:
            return False
        #   TODO: simplify
        self.initial_cond =  np.array(self.initial_cond)
        self.reference_pose =  np.array(self.reference_pose)

        return self.config_reference(self.ref_name)


    def config_data_storage(self):


        #   output files for data storage:
        self.position_d = os.path.join(self.output, f"position.dat")
        with open(self.position_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.vel_d = os.path.join(self.output, f"velocities.dat")
        with open(self.vel_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.norm_e_d = os.path.join(self.output, f"norm_error.dat")
        with open(self.norm_e_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.error_d = os.path.join(self.output, f"error.dat")
        with open(self.error_d, 'w') as file:
            pass  # 'w' mode clears the file's contents

        self.features_d = os.path.join(self.output, f"features.dat")
        with open(self.features_d, 'w') as file:
            pass  # 'w' mode clears the file's contents


        if self.k_int != 0.:
            self.error_int_d = os.path.join(self.output, f"error_int.dat")
            with open(self.error_int_d, 'w') as file:
                pass  # 'w' mode clears the file's contents


    def state_changed_ibvs(self, msg):
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

    def save_error(self,t):

        if self.error is None:
            return
        if self.ids_save is None:
            return
        if self.first_track:
            return
        with open(self.error_d, 'ab') as f:
            for i, m in enumerate( self.ids_save):

                data = (t, m)
                data += tuple(self.error[:,i].reshape(-1))
                binary = struct.pack('didd', *data)
                f.write(binary)

    def save_integral(self,t):
        if self.err_int is None:
            return
        with open(self.error_int_d, 'ab') as f:
            for i, m in enumerate( self.ids_int):

                data = (t, m)
                data += tuple(self.err_int[:,i].reshape(-1))
                binary = struct.pack('didd', *data)
                f.write(binary)

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

        with open(self.features_d, 'ab') as f:
            for i, m in enumerate(self.ids):

                data = (t, m)
                data += tuple(self.p[i, :].reshape(-1))
                binary = struct.pack('didd', *data)
                f.write(binary)

        self.save_error(t)


        if self.k_int != 0:
            self.save_integral(t)




    def preproc_image(self):
        if not self.enable_video :
            return
        if  self.cv_image is None:
            return
        self.m_image = self.cv_image.copy()
        if len(self.p) > 2:
            self.custom_draw(self.m_image, self.p.T)



    def control(self):

        if self.first_track:
            return

        _ret = self.match_ref()
        if _ret is None:
            self.get_logger().warning(f"Not enough matchings")
            self.u *= 0.9
            return
        p_ref, p, idx = _ret
        self.error = p -p_ref
        self.ids_save = [self.ids[k] for k in idx]

        # # BEGIN debug
        # if self.label == 0:
        #     # self.get_logger().info(str(self.error[j]))
        #     # self.get_logger().info(str(idx))
        #     self.get_logger().info(str(self.error[self.label]))
        #
        # # END debug


        # TODO image draw
        if not self.m_image is None:
            self.custom_draw_matching(self.m_image,
                        p,
                        p_ref,
                        color1 = (0,200,0),
                        color2 = (0,124,255),
                        reproject = True)

        # mask = np.logical_or(self.error[self.label][0,:] !=0.,
        #                      self.error[self.label][1,:] !=0.)

        # self.L = interaction_matrix_xyz(self.points[:,mask], self.img_depth)
        self.L = interaction_matrix_xyz(p, self.img_depth)
        # self.L = interaction_matrix_xyz(self.points, self.current_pose.position.z)
        L_inv = Inv_Moore_Penrose(self.L)
        if L_inv is None:
            self.get_logger().error("Invalid Ls matrix")
            self.u *= 0.9
            return
        # _u = self.gain * L_inv @ self.error[self.label][:,mask].T.reshape(-1)
        # _u = self.gain * L_inv @ self.error[self.label].T.reshape(-1)  / float(mask.sum())
        _u =  L_inv @ self.error.T.reshape(-1)
        _u *= self.gain
        _u /=  float(self.error.shape[1])
        if self.enable_tanh:
            t = self.get_clock().now().nanoseconds * 1e-9
            _u *= np.tanh(0.5*(t-self.t0))
        # BEGIN debug
        # if self.label == 0:
        #     _u = np.zeros(6)
        #     self.get_logger().info(str(self.error[self.label][:,mask]))

            # END debug

        # # # BEGIN DEBUG
        # if self.label != 0:
        #     self.u = np.zeros(6)
        #     return
        #
        # if _n != 0:
        #     self.get_logger().info(f"Neig:{_n}")
        #     _u /= _n
        # _u = np.zeros(6)
        # # # END DEBUG
        # 6DOF
        _w = self.R_cam @ _u[3:]
        _v = (self.R_cam @ _u[:3]).reshape(-1)
        _v += np.cross( self.t_cam , _w.reshape(-1) )
        _w *= self.kw
        self.u[:3] = _v.copy()
        self.u[3:] = _w.reshape(-1)
        #   4DOF
        # # _w = self.R_cam @ np.array([0.,_u[3],0.])
        # _w = self.R_cam @ np.array([0.,0.,_u[3]])
        # _v = (self.R_cam @ _u[:3]).reshape(-1)
        # _v += np.cross( self.t_cam , _w.reshape(-1) )
        # _w *= self.kw
        # self.u[:3] = _v.copy()
        # self.u[3:] = _w.copy()

        return

    def control_int(self):
        if self.first_track:
            return

        _ret = self.match_ref()
        if _ret is None:
            self.get_logger().warning(f"Not enough matchings")
            self.u *= 0.9
            return
        p_ref, p, idx = _ret

        self.error = p -p_ref
        self.ids_save = [self.ids[k] for k in idx]


        # # BEGIN debug
        # if self.label == 0:
        #     # self.get_logger().info(str(self.error[j]))
        #     # self.get_logger().info(str(idx))
        #     self.get_logger().info(str(self.error[self.label]))
        #
        # # END debug

        _n += 1.

        # TODO image draw
        if not self.m_image is None:
            self.custom_draw_matching(self.m_image,
                        p,
                        p_ref,
                        color1 = (0,200,0),
                        color2 = (0,124,255),
                        reproject = True)



        # mask = np.logical_or(self.error[self.label][0,:] !=0.,
        #                      self.error[self.label][1,:] !=0.)

        # self.L = interaction_matrix_xyz(self.points[:,mask], self.img_depth)
        self.L = interaction_matrix_xyz(self.points, self.img_depth)
        # self.L = interaction_matrix_xyz(self.points, self.current_pose.position.z)
        L_inv = Inv_Moore_Penrose(self.L)
        if L_inv is None:
            self.get_logger().error("Invalid Ls matrix")
            _u = np.zeros(6)
            # self.u = np.zeros(6)
            self.u *= 0.9
            return
        # _u = self.gain * L_inv @ self.error[self.label][:,mask].T.reshape(-1)


        #   Integral filter to index
        filt = [a in self.ids for a in self.ids_int]
        self.err_int = self.err_int[:,filt]
        self.ids_int = [i for i in self.ids]

        #   Control
        if self.error.shape[1] != self.err_int.shape[1]:
            self.get_logger().info(str(self.ids_int))
            self.get_logger().info(str(self.ids))
            self.ids_int = [i for i in self.ids]
            self.err_int = np.zeros(self.points.shape)
            self.get_logger().warning("Integral misshape")

        _error  =  self.errorT.reshape(-1)
        _error +=  self.k_int * self.err_int.T.reshape(-1)
        _u = self.gain * L_inv @ _error
        _u /= float(_error.shape[0]/2)
        # _u /= float(mask.sum())
        if self.enable_tanh:
            t = self.get_clock().now().nanoseconds * 1e-9
            _u *= np.tanh(0.5*(t-self.t0))

        #   Integration step
        self.tick = self.tock
        self.tock = self.get_clock().now().nanoseconds * 1e-9
        if self.int_act:
            dt = self.tock - self.tick
            # self.err_int *= 0.8    #   Filter
            self.err_int += dt * self.error
        else:
            if self.norm < self.th_int:
                self.get_logger().info(f"PI activated")
                self.int_act = True


        # BEGIN debug
        # if self.label == 0:
        #     _u = np.zeros(6)
        #     self.get_logger().info(str(self.error[self.label][:,mask]))

            # END debug

        # # # BEGIN DEBUG
        # if self.label != 0:
        #     self.u = np.zeros(6)
        #     return
        #
        # if _n != 0:
        #     self.get_logger().info(f"Neig:{_n}")
        #     _u /= _n
        # _u = np.zeros(6)
        # # # END DEBUG
        # 6DOF
        _w = self.R_cam @ _u[3:]
        _v = (self.R_cam @ _u[:3]).reshape(-1)
        _v += np.cross( self.t_cam , _w.reshape(-1) )
        _w *= self.kw
        self.u[:3] = _v.copy()
        self.u[3:] = _w.reshape(-1)
        #   4DOF
        # # _w = self.R_cam @ np.array([0.,_u[3],0.])
        # _w = self.R_cam @ np.array([0.,0.,_u[3]])
        # _v = (self.R_cam @ _u[:3]).reshape(-1)
        # _v += np.cross( self.t_cam , _w.reshape(-1) )
        # _w *= self.kw
        # self.u[:3] = _v.copy()
        # self.u[3:] = _w.copy()

        return

    def s_resetvis(self):

        self.reset_tracking = True
        self.img_proc(self.cv_image)

        # Control interemdiate state
        self.tock = self.get_clock().now().nanoseconds * 1e-9
        self.tick = self.tock

        self.t0 = self.get_clock().now().nanoseconds * 1e-9

        self.ids_int = [i for i in self.ids]
        self.err_int = np.zeros(self.points.shape)


        super().s_resetvis()


    def s_control(self):

        # self.reset_tracking = True
        # self.img_proc(self.cv_image)

        # Control interemdiate state
        self.tock = self.get_clock().now().nanoseconds * 1e-9
        self.tick = self.tock

        self.t0 = self.get_clock().now().nanoseconds * 1e-9

        self.ids_int = [i for i in self.ids]
        self.err_int = np.zeros(self.points.shape)

        self.int_act = False

        self.state = self.s_control_

    def s_control_(self):

        if self.points is None:
            self.get_logger().error("Image error can not be computed")
            if self.data2save:
                self.save_data()
        else:
            self.control_()

            self.norm = np.linalg.norm(self.error)/ float(self.error.shape[0])

            #   Print error each error_counter_th steps
            self.error_counter += 1
            if self.error_counter >= self.error_counter_th:
                self.get_logger().info(str(self.norm))
                self.error_counter = 0
            self.data2save = True
            self.save_data()

        # self.u = np.zeros(6)
        super().s_control()


    def control_loop(self):

        #   Preprocess matching points image
        self.preproc_image()
        #   Exec state
        self.state()
        #   Video
        self.video_frame(self.m_image)

        #   Publish matching image
        if self.m_image is None:
            return
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.m_image, "bgr8"))


def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
