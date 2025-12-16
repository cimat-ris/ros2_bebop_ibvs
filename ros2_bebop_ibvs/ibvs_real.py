
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from numpy import sin, cos
import numpy as np
import cv2
import struct
import os


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

def Inv_Moore_Penrose(L):
    A = L.T@L
    if np.linalg.det(A) == 0:
        return None
    return np.linalg.inv(A) @ L.T

class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        # Parameters
        self.declare_parameter('frequency', 30.0)
        self.declare_parameter('offset', [0.]*2)
        self.declare_parameter('gains', [0.]*4)
        self.declare_parameter('name', 'bebop')
        self.declare_parameter('ref_image', 'real_1.png')
        self.declare_parameter('time', 1.)
        self.declare_parameter('img_depth', 1.)
        self.declare_parameter('K', [1.]*9)
        self.declare_parameter('output', "output")
        self.frequency = self.get_parameter('frequency').value
        self.offset = self.get_parameter('offset').value
        self.gains = self.get_parameter('gains').value
        self.img_depth = self.get_parameter('img_depth').value
        self.name = self.get_parameter('name').value
        self.ref_image = self.get_parameter('ref_image').value
        self.time = self.get_parameter('time').value
        self.K = self.get_parameter('K').value
        self.output = self.get_parameter('output').value

        #   Camera calibration data
        self.f = [self.K[0], self.K[4]]
        self.pPrinc = [self.K[2],self.K[5]]
        self.K = np.array(self.K).reshape((3,3))
        # self.gains += [self.gains[3], self.gains[3]]
        self.gains = np.array(self.gains)

        if not self.name:
            self.get_logger().info('Empty "name": Setting "bebop" as default.')
            self.name = 'bebop'
        self.get_logger().info(f"Robot Name: {self.name}")



        #   Reference image
        image_ref = cv2.imread(self.ref_image)
        if  image_ref is None :
            self.get_logger().error(f"Image {self.ref_image} could not be read ")
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
        self.points = None

        #   Camera and robot transformations
        self.R_cam = np.array([[0.,  0., 1.],
                               [-1., 0., 0.],
                               [0., -1., 0.]])
        self.t_cam = np.array([0.09, 0., 0.])

        #   internal variables
        self.wait = int(self.time * self.frequency)
        self.counter = 0
        self.idle = True
        self.found_arucos_w = False
        self.u = np.zeros(6)
        self._u = np.zeros(6)

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

        #   output files
        self.arucos_d = os.path.join(self.output, "arUcos.dat")
        with open(self.arucos_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.error_d = os.path.join(self.output, "error.dat")
        with open(self.error_d, 'w') as file:
            pass  # 'w' mode clears the file's contents
        self.vel_d = os.path.join(self.output, "velocities.dat")
        with open(self.vel_d , 'w') as file:
            pass  # 'w' mode clears the file's contents

        #   Loop
        self.not_init = True
        self.timer = self.create_timer(1.0 / self.frequency, self.control_loop)

    def start(self,msg):
        self.takeoff_pub.publish(Empty())
        self.counter = 0
        self.idle = False

    def stop(self, msg):
        self.land_pub.publish(Empty())
        self.land_pub.publish(Empty())
        self.counter = 0
        self.idle = True

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
            data = (t,)
            data += tuple(self.u[[0,1,2,5]].reshape(-1))
            # data += tuple(self._u[[0,1,2,3]].reshape(-1))
            binary = struct.pack('ddddd', *data)
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

    def control_loop(self):

        if self.not_init:
            self.camera_tilt.publish(Vector3())
            self.camera_tilt.publish(Vector3())
            self.not_init = False

        if self.idle:
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
            return

        self.error = self.points - self.points_ref

        #   TODO: depth?
        # self.L = interaction_matrix_xyz(self.points_ref, self.img_depth)
        self.L = interaction_matrix_y(self.points_ref, self.img_depth)
        L_inv = Inv_Moore_Penrose(self.L)
        # _, self.svd, _ = np.linalg.svd(self.L.T @ self.L)

        if L_inv is None:
            self.get_logger().error("Invalid Ls matrix")
            # self.u =  np.zeros(6)
            self.cmd_pub.publish(msg)

        # self.u = - self.gain * L_inv @ self.error.T.reshape((-1,1))
        self._u = - L_inv @ self.error.T.reshape((-1,1)).reshape(-1)


        #   Transformation camera -> robot

        #   6DOF
        # _w = (self.R_cam @ self._u[3:]).reshape(-1)
        # _v = (self.R_cam @ self._u[:3]).reshape(-1)
        # print(_v.shape, self.t_cam.shape, _w.shape)
        # _v += np.cross( self.t_cam , _w )
        # self.u[:3] = _v.copy()
        # self.u[3:] = _w.copy()

        #   4DOF
        self.u[:3] = self.gains[:3] * (self.R_cam @ self._u[:3]).reshape(-1)
        self.u[5] = -self.gains[3] * self._u[3]
        self.u[:3] += np.cross(self.t_cam, self.u[3:])
        # self.u[:3] += - self.t_cam[0]* self.u[5] # TODO simplify cross product

        msg.linear.x = float(self.u[0])
        msg.linear.y = float(self.u[1])
        msg.linear.z = float(self.u[2])
        msg.angular.z = float(self.u[5])
        # self.get_logger().info( f"Control_cmd_vel: {self._u}")
        self.get_logger().info( f"Control_cmd_vel: {self.u}")
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
