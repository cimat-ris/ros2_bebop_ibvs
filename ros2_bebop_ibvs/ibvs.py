#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Empty
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
# from PyQt5.QtGui import QImage
import cv2
import numpy as np

IDLE = 0
AUTOMATIC = 1
TAKEOFF = 2
LANDING = 3
STOP = 4


def Interaction_Matrix(points,Z,gdl = 1):

    n = points.shape[0]
    if gdl == 1:
        m = 6
    if gdl == 2:
        m = 4
    if gdl == 3:
        m = 3
    L = np.zeros((n,2*m))

    if gdl == 1:
        L[:,0]  =   L[:,7] = -1/Z
        L[:,2]  =   points[:,0]/Z
        L[:,3]  =   points[:,0]*points[:,1]
        L[:,4]  =   -(1+points[:,0]**2)
        L[:,5]  =   points[:,1]
        L[:,8]  =   points[:,1]/Z
        L[:,9]  =   1+points[:,1]**2
        L[:,10] =   -points[:,0]*points[:,1]
        L[:,11] =   -points[:,0]
    if gdl == 2:
        L[:,0]  =   L[:,5] = -1/Z
        L[:,2]  =   points[:,0]/Z
        L[:,3]  =   points[:,1]
        L[:,6]  =   points[:,1]/Z
        L[:,7] =   -points[:,0]
    if gdl == 3:
        L[:,0]  =   L[:,4] = -1/Z
        L[:,2]  =   points[:,0]/Z
        L[:,5]  =   points[:,1]/Z


    return L.reshape((2*n,m))

def Inv_Moore_Penrose(L):
    A = L.T@L
    if np.linalg.det(A) == 0:
        return None
    return np.linalg.inv(A) @ L.T

class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        
        # Parámetros
        self.declare_parameter('frequency', 50.0)
        self.declare_parameter('robot_name', 'bebop')
        self.declare_parameter('takeoff_threshold', 0.04)
        self.declare_parameter('landing_threshold', 0.08)
        self.declare_parameter('takeoff_height', 1.0)  # Nueva altura de despegue
        self.declare_parameter('ref_image', "reference.png")  # Nueva altura de despegue
        self.declare_parameter('img_depth', 1.)  # Nueva altura de despegue
        self.declare_parameter('gain', 1.)  # Nueva altura de despegue
        self.declare_parameter('gain_takeoff', 1.)  # Nueva altura de despegue
        self.declare_parameter('K', [1.]*9)  # Nueva altura de despegue
        
        self.frequency = self.get_parameter('frequency').value
        self.robot_name = self.get_parameter('robot_name').value.strip()
        # self.goal_name = self.get_parameter("goal_name").value.strip()
        self.goal_name = "goal"
        self.takeoff_threshold = self.get_parameter('takeoff_threshold').value
        self.landing_threshold = self.get_parameter('landing_threshold').value
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.ref_image = self.get_parameter('ref_image').value
        self.img_depth = self.get_parameter('img_depth').value
        self.gain = self.get_parameter('gain').value
        self.gain_takeoff = self.get_parameter('gain_takeoff').value
        self.K = self.get_parameter('K').value

        self.f = [self.K[0], self.K[4]]
        self.pPrinc = [self.K[2],self.K[5]]
        self.K = np.array(self.K).reshape((3,3))

        if not self.robot_name:
            self.get_logger().info('Empty "robot_name": Setting "bebop" as default.')
            self.robot_name = 'bebop'
        self.get_logger().info(f"Robot Name: {self.robot_name}")

        #   Reference image
        #   TODO: Cargar matriz de camara
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
        self.t_cam = np.array([0.09, 0., 0.])

        # Publicadores
        qos = QoSProfile(depth=10)
        self.cmd_pub = self.create_publisher(Twist, f"/{self.robot_name}/cmd_vel", qos)
        self.cmd_enable = self.create_publisher(Bool, f"/{self.robot_name}/enable", qos)

        #   Image bridge
        img_qos = QoSProfile(depth=2)
        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(
            Image, '/camera/image_raw',
            self.image_recv, img_qos)
        # self.image_subscription = self.create_subscription(
        #     Image, '/world/default/model/parrot_bebop_2/link/body/sensor/rgb_camera_sensor/image',
        #     self.image_recv, qos)
        self.image_pub = self.create_publisher(Image, '/matching', img_qos)
        # self.image_subscription = self.create_subscription(
            # Image, '/world/bebop/model/bebop1/link/body/sensor/rgb_camera_sensor/image',
            # self.image_callback, qos)

        # Suscriptores
        self.pos_sub = self.create_subscription(Pose, f"/parrot_bebop_2/pose", self.pos_changed, qos)
        self.state_sub = self.create_subscription(Int32, "/state", self.state_changed, qos)
        
        # Estado
        self.state = IDLE
        self.goal = Pose()
        self.current_pose = Pose()
        self.enable = False
        self.takeoff_complete = False  # Nuevo flag para controlar despegue completado
        


        # Timer
        self.timer = self.create_timer(1.0 / self.frequency, self.control_loop)

    # def image_callback(self, msg, drone_name):
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #     except Exception as e:
    #         self.get_logger().error(f"Error en image_callback para {drone_name}: {str(e)}")
    #         return
    #
    #     if cv_image is None:
    #         self.get_logger().error("Received empty image")
    #         return
    #
    #     h, w = cv_image.shape[:2]
    #     ratio = min(320/w, 240/h)
    #     cv_image = cv2.resize(cv_image, (int(w*ratio), int(h*ratio)))
    #
    #     h, w, ch = cv_image.shape
    #     bytes_per_line = ch * w
    #     cv2.imshow(window_name, image)
    #     cv2.waitKey(1)

    def get_param_or(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).get_parameter_value().double_value

    def state_changed(self, msg):
        """Maneja cambios en el estado desde el tópico /state"""
        new_state = msg.data
        
        if new_state == IDLE:
            if self.state != IDLE:
                self.get_logger().info("Cambiando a estado IDLE")
                self.state = IDLE
                self.enable = False
                
        elif new_state == AUTOMATIC:
            if self.state == TAKEOFF and self.takeoff_complete:
                self.get_logger().info("Cambiando a estado AUTOMATIC")
                self.state = AUTOMATIC
                self.enable = True
            elif self.state != TAKEOFF:
                self.get_logger().info("No se puede cambiar a AUTOMATIC sin completar el despegue primero")
                
        elif new_state == TAKEOFF:
            if self.state != TAKEOFF:
                self.get_logger().info("Cambiando a estado TAKING_OFF")
                self.state = TAKEOFF
                self.takeoff_complete = False
                self.enable = True
                self.cmd_enable.publish(Bool(data=self.enable))
                
        elif new_state == LANDING:
            if self.state != LANDING:
                self.get_logger().info("Cambiando a estado LANDING")
                self.state = LANDING
                self.enable = True
                
        elif new_state == STOP:
            if self.state != STOP:
                self.get_logger().info("¡EMERGENCY STOP activado!")
                self.state = STOP
                self.enable = False
        

    def goal_changed(self, msg):
        self.goal = msg

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
            self.get_logger().error("No ArUcos found in received image")
            self.points = None
            self.points_ref = None
            return

        #   Pairing
        _p = np.array([])
        _p_ref = np.array([])
        for i in range(ids.shape[1]):
            if ids[0,i] in self._ids_ref:
                idx = self._ids_ref.index(ids[0,i])
                _p = np.concatenate ((_p, corners[i].reshape(-1 )))
                _p_ref = np.concatenate ((_p_ref, self.corners_ref[idx].reshape(-1) ))

        if _p.shape[0] == 0:
            self.get_logger().error(f"Matching Failed {self._ids_ref} {ids.tolist()}")
        #   Normalzie
        _p = _p.reshape((-1,2)).T
        _p_ref = _p_ref.reshape((-1,2)).T
        _p = self.normalize(_p)
        _p_ref = self.normalize(_p_ref)

        self.points = _p.T.reshape((-1,2))
        self.points_ref = _p_ref.T.reshape((-1,2))

        #   Publish detection
        _image = self.cv_image.copy()
        # print(ids)
        cv2.aruco.drawDetectedMarkers(_image, corners, ids,
                                        borderColor = (0,100,0.) )
        cv2.aruco.drawDetectedMarkers(_image, self.corners_ref, self.ids_ref,
                                        borderColor = (100,1.,0.) )

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(_image, "bgr8"))

    def control_loop(self):
        # Altura alcanzada
        if self.state == TAKEOFF:
            current_z = self.current_pose.position.z

            if abs(current_z- self.takeoff_height) < self.takeoff_threshold:
                self.get_logger().info(f"Altura de despegue alcanzada: {current_z:.2f}m")
                self.takeoff_complete = True
                msg = Twist()
                msg.linear.z = -0.1*self.gain_takeoff*float(current_z- self.takeoff_height)
                self.cmd_pub.publish(msg)

            else:

                msg = Twist()
                msg.linear.z = -self.gain_takeoff*float(current_z- self.takeoff_height)
                self.get_logger().debug(f"Control input: {msg.linear.z}")
                self.cmd_pub.publish(msg)

        elif self.state == LANDING:
            current_z = self.current_pose.position.z
            msg = Twist()
            
            if current_z > self.landing_threshold:
                # Descender controladamente
                msg.linear.z = self.gain_takeoff* float(- current_z)
                self.cmd_pub.publish(msg)
            else:
                # Aterrizaje completado
                self.get_logger().info("¡Aterrizaje completado!")
                self.state = IDLE
                self.enable = False
                self.cmd_enable.publish(Bool(data=self.enable))
                self.cmd_pub.publish(Twist())

        elif self.state == AUTOMATIC and self.enable:

            if self.points is None:
                self.get_logger().error("Image error can not be computed")
                msg = Twist()
                msg.linear.x = 0.
                msg.linear.y = 0.
                msg.linear.z = 0.
                msg.angular.z = 0.

                try:
                    self.cmd_pub.publish(msg)

                except Exception as e:
                    self.get_logger().error(f"Error en control automático: {str(e)}")
                    self.enable = False
                    self.cmd_enable.publish(Bool(data=self.enable))
                return
            #   IBVS
            self.error = self.points - self.points_ref

            L = Interaction_Matrix(self.points, self.img_depth) #   TODO: profundidad
            L = Inv_Moore_Penrose(L)
            if L is None:
                print("Invalid Ls matrix")
                self.u =  np.zeros(6)

            self.u = - self.gain * L @ self.error.reshape((-1,1))


            #   Transformation camera -> robot

            _w = self.R_cam @ self.u[3:]
            _v = (self.R_cam @ self.u[:3]).reshape(-1)
            _v += np.cross( self.t_cam , _w.reshape(-1) )

            self.u[:3] = _v.reshape((3,1))
            self.u[3:] = _w.reshape((3,1))

            #   Send message
            msg = Twist()
            msg.linear.x = float(self.u[0])
            msg.linear.y = float(self.u[1])
            msg.linear.z = float(self.u[2])
            msg.angular.z = float(self.u[5])

            try:
                self.cmd_pub.publish(msg)

            except Exception as e:
                self.get_logger().error(f"Error en control automático: {str(e)}")
                self.enable = False
                self.cmd_enable.publish(Bool(data=self.enable))

        elif self.state in [IDLE, STOP]:
            self.cmd_pub.publish(Twist())
            self.enable = False
            self.cmd_enable.publish(Bool(data=self.enable))

def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
