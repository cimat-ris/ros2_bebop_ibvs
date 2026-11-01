import numpy as np

#   Utils
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
    _R = rotation_matrix(angs[2], 'z')
    _R = _R @ rotation_matrix(angs[1], 'y')
    _R = _R @ rotation_matrix(angs[0], 'x')
    return _R

# R = rotation_matrix_euler([0.,0.01,np.pi/4.])
R = rotation_matrix(-0.5,'y')

#   Add your custom funtions if needed
#   Output should be a 6 elment nupy array

def sin(t):
    u = np.array([0.05 * np.sin(0.3 * t),
                  0.05 ,
                  0.,0.,0.,0.])
    return u

def circle(t):
    r = 0.05
    w = 0.5
    u = np.array([r * np.sin(w * t),
                  r * np.cos(w * t) ,
                  0.,0.,0.,0.])
    return u

def constant(t):
    u = np.array([.02,0.,0.,0.,0.,0.])
    return u

def constant2(t):
    u_t = 0.01 * R @ np.array([1.,0.,0.])
    u_r = np.array([0.,0.,0.])
    return np.concatenate((u_t,u_r))







#   Function select for control inputs in leader
#   List the function to follow in each elment of the list
#   Each index corresponds to the label of the agent

# lead = [constant,
lead = [constant2,
        constant,
        constant,
        constant,
        constant,
        ]
