import numpy as np


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







#   Function select for control inputs in leader
#   List the function to follow in each elment of the list
#   Each index corresponds to the label of the agent

lead = [constant,
        constant,
        constant,
        constant,
        constant,
        ]
