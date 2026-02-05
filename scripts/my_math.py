import numpy as np
from numpy import pi, arctan2

#   Logger
import logging
logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.NOTSET)

def rotation_matrix(ang,ax):
    
    ca = np.cos(ang)
    sa = np.sin(ang)
    if ax == 'x':
        return np.array([[1.0, 0.0, 0.0],
                        [0.0,  ca, -sa],
                        [0.0,  sa,  ca]])
    elif ax == 'y':
        return np.array([[ ca, 0.0,  sa],
                        [0.0, 1.0, 0.0],
                        [-sa, 0.0,  ca]])
    elif ax == 'z':
        return np.array([[ ca, -sa, 0.0],
                        [ sa,  ca, 0.0],
                        [0.0, 0.0, 1.0]])
    
    return None

def rotation_matrix_euler(angs):
    _angs = angs.reshape(-1)
    _R = rotation_matrix(_angs[2], 'z')
    _R = _R @ rotation_matrix(_angs[1], 'y')
    _R = _R @ rotation_matrix(_angs[0], 'x')
    return _R

def get_angles(R, prev_angs= None):
    #print(R)
    if (R[2,0] < 1.0):
        if R[2,0] > -1.0:
            pitch = np.arcsin(-R[2,0])
            if not( prev_angs is None):
                pitch_alt = np.sign(pitch) *(pi - abs(pitch))
                delta_pitch = abs(pitch-prev_angs[1])
                if delta_pitch > pi:
                    delta_pitch = 2*pi-delta_pitch
                delta_pitch2 = abs(pitch_alt-prev_angs[1])
                if delta_pitch2 > pi:
                    delta_pitch2 = 2*pi-delta_pitch2
                if delta_pitch2 < delta_pitch:
                    pitch = pitch_alt
            cp = np.cos(pitch)
            yaw = arctan2(R[1,0]/cp,R[0,0]/cp)
            roll = arctan2(R[2,1]/cp,R[2,2]/cp)
        else:
            pitch = np.pi/2.
            if prev_angs is None:
                yaw = -arctan2(-R[1,2],R[1,1])
                roll = 0.
            else:
                tmp = arctan2(-R[1,2],R[1,1])
                roll = prev_angs[0]
                yaw = roll - tmp
                if yaw > pi:
                    yaw -= 2*pi
                if yaw < -pi:
                    yaw += 2*pi
                
    else:
        pitch = -np.pi/2.
        if prev_angs is None:
            yaw = arctan2(-R[1,2],R[1,1])
            roll = 0.
        else:
            tmp = arctan2(-R[1,2],R[1,1])
            roll = prev_angs[0] 
            yaw = tmp - roll
            if yaw > pi:
                yaw -= 2*pi
            if yaw < -pi:
                yaw += 2*pi
    return np.array( [roll, pitch, yaw])

def moore_penrose_inverse(L):
    A = L.T@L
    if np.linalg.det(A) == 0:
        return None
    return np.linalg.inv(A) @ L.T
