#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import  Int32

import cv2
import os


class SimControl(Node):

    def __init__(self):
        super().__init__('sim_control')
        
        # Parameters
        self.declare_parameter('n_agents', 1)
        self.n_agents = self.get_parameter('n_agents').value

        #   Pubs / subs
        qos = QoSProfile(depth=10)
        self.state_sub  = self.create_subscription(Int32,
                            f"/state",
                            self.relay,
                            qos)
        self.state_pub  = []
        qos2 = QoSProfile(depth=2)
        for i in range(self.n_agents):
            _pub = self.create_publisher(Int32,
                            f"/state_{i}",
                            qos2)
            self.state_pub.append(_pub)

    def relay(self, msg):

        for i in range(self.n_agents):
            self.state_pub[i].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    sim_control = SimControl()
    rclpy.spin(sim_control)
    sim_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
