
#   libs
import numpy as np
from numpy import pi, arctan2
from numpy.linalg import norm, svd
from scipy.optimize import minimize_scalar
import cv2

#   Plot
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#   LATEX FIX
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#   sys
import os
import argparse
import yaml

#   Custom
from camera import Camera
from my_math import rotation_matrix_euler, get_angles


markers_list = ["4X4_50" ,
        "4X4_100" ,
        "4X4_250" ,
        "4X4_1000" ,
        "5X5_50" ,
        "5X5_100" ,
        "5X5_250" ,
        "5X5_1000" ,
        "6X6_50" ,
        "6X6_100" ,
        "6X6_250" ,
        "6X6_1000" ,
        "7X7_50" ,
        "7X7_100" ,
        "7X7_250" ,
        "7X7_1000" ,
        "ARUCO_ORIGINAL" ,
        "APRILTAG_16h5" ,
        "APRILTAG_25h9" ,
        "APRILTAG_36h10" ,
        "APRILTAG_36h11" ,
        "ARUCO_MIP_36h12"]


class Ploter():

    def __init__(self, arg):

        name = arg.config
        self.directory = arg.directory
        with open(name, 'r') as file:
            _dict = yaml.safe_load(file)

        self.n_agents = _dict['n_agents']
        pd = np.array(_dict['pd'])

        if 'vel_limits' in _dict:
            self.vel_limits = _dict['vel_limits']
        else:
            self.vel_limits = [-.5, .5]

        if 'error_limits' in _dict:
            self.error_limits = _dict['error_limits']
        else:
            self.error_limits = [-1., 1.]

        if 'camera_angle' in _dict:
            self.camera_angle = eval(_dict['camera_angle'])
        else:
            self.camera_angle = 0.



        pd = pd.reshape((-1,4))
        n = pd.shape[0]

        pd = (pd[:,:3],  np.zeros((n,2)), pd[:,3].reshape((-1,1)))
        self.pd = np.concatenate(pd, axis = 1)
        # pd[:,3] = pi/2.
        self.pd[:,4] = self.camera_angle
        self.pd = self.pd.T


        self.error = [None] * self.n_agents
        self.position = [None] * self.n_agents
        self.log = [None] * self.n_agents

        self.arucos = [None] * self.n_agents

        self.joined_error =  None
        self.formation_error = None

    def read_data(self, label):

        b_float = 4
        b_double = 8
        b_int = 8

        name = os.path.join(self.directory ,f"position_{label}.dat")
        if os.path.exists(name):
            length = os.path.getsize(name)
            if length > 0:
                with open(name, 'rb') as fileH:
                    rows = (length) / (5* b_double)
                    rows = int(np.floor(rows))
                    position = np.fromfile(fileH,
                                            dtype = np.float64,
                                            count = 5*rows)
                    position = position.reshape((rows,5))
                    position = position.T
                    _concat = (position[:4,:], np.zeros((2,rows)), position[4,:].reshape((1,-1)))
                    position = np.concatenate(_concat)
                    position[0,:] -= position[0,0]
                    # position[4,:] = -pi/2.
                    position[5,:] = pi
                    # position[6,:] -= pi/2.
                self.position[label] = position

        name = os.path.join(self.directory ,f"velocities_{label}.dat")
        self.velocities = None
        if os.path.exists(name):
            length = os.path.getsize(name)
            if length > 0:
                with open(name, 'rb') as fileH:
                    rows = (length) / (5* b_double)
                    rows = int(np.floor(rows))
                    velocities = np.fromfile(fileH,
                                            dtype = np.float64,
                                            count = 5*rows)
                    velocities = velocities.reshape((rows,5))
                    velocities = velocities.T
                    velocities[0,:] -= velocities[0,0]
                self.velocities = velocities

        self.velocities_log = [None, None]
        name = os.path.join(self.directory ,f"log_vel_prop_{label}.dat")
        if os.path.exists(name):
            length = os.path.getsize(name)
            if length > 0:
                with open(name, 'rb') as fileH:
                    rows = (length) / (5* d)
                    rows = int(np.floor(rows))
                    log = np.fromfile(fileH,
                                            dtype = np.float64,
                                            count = 5*rows)
                    log = log.reshape((rows,5))
                    log = log.T
                    log[0,:] -= log[0,0]
                self.velocities_log[0] = log
        name = os.path.join(self.directory ,f"log_vel_int_{label}.dat")
        if os.path.exists(name):
            length = os.path.getsize(name)
            if length > 0:
                with open(name, 'rb') as fileH:
                    rows = (length) / (5* b_double)
                    rows = int(np.floor(rows))
                    log = np.fromfile(fileH,
                                            dtype = np.float64,
                                            count = 5*rows)
                    log = log.reshape((rows,5))
                    log = log.T
                    log[0,:] -= log[0,0]
                self.velocities_log[1] = log


        name = os.path.join(self.directory ,f"norm_error_{label}.dat")
        self.n_e = None
        if os.path.exists(name):
            length = os.path.getsize(name)
            if length > 0:
                with open(name, 'rb') as fileH:
                    rows = (length) / (2* b_double)
                    rows = int(np.floor(rows))
                    n_e = np.fromfile(fileH,
                                            dtype = np.float64,
                                            count = 2*rows)
                    n_e = n_e.reshape((rows,2))
                    n_e = n_e.T
                    n_e[0,:] -= n_e[0,0]

                    n_e = {'t': n_e[0,:], 'v': n_e[1:,:].T}
                self.n_e = n_e

        name = os.path.join(self.directory, f"features_{label}.dat")
        self.features = None
        if os.path.exists(name):
            length = os.path.getsize(name)
            if length > 0:
                with open(name, 'rb') as fileH:
                    size = 3*b_double + b_int
                    rows = length / size
                    rows = int(np.floor(rows))

                    features = {}

                    for i in range (rows-1):
                        time = np.fromfile(fileH,
                                            dtype = np.float64,
                                            count = 1)
                        time = time[0]
                        idx = np.fromfile(fileH,
                                            dtype = np.int64,
                                            count = 1)
                        idx = idx[0]
                        _feat = np.fromfile(fileH,
                                            dtype = np.float64,
                                            count = 2)
                        # print(time, idx, _feat)

                        # print(time)
                        # print(idx)
                        # print(_feat)

                        if idx in features:
                            features[idx]["t"].append(time)
                            features[idx]["v"] = np.concatenate([features[idx]["v"],_feat])
                        else:
                            _d = {"t":[time], "v":_feat}
                            features[idx] = _d

                    t0 = [ features[key]["t"][0] for  key in features]
                    t0 = min(t0)
                    for i in features:
                        features[i]["v"] = features[i]["v"].reshape((-1,2))
                        features[i]["v"] = features[i]["v"].T
                        features[i]["t"] = [t - t0 for t in features[i]["t"]]
                self.features = features



        self.error[label] = [None]*self.n_agents
        all_idx = set()
        for k in range(self.n_agents):
            if k != label:
                name = os.path.join(self.directory ,f"error_{label}_{k}.dat")

                if os.path.exists(name):
                    length = os.path.getsize(name)
                    if length > 0:
                        with open(name, 'rb') as fileH:
                            size = 3*b_double + b_int
                            rows = (length) / size
                            rows = int(np.floor(rows))

                            error = {}

                            for i in range (rows):
                                time = np.fromfile(fileH,
                                                    dtype = np.float64,
                                                    count = 1)
                                time = time[0]
                                idx = np.fromfile(fileH,
                                                    dtype = np.int64,
                                                    count = 1)
                                idx = idx[0]
                                _error = np.fromfile(fileH,
                                                    dtype = np.float64,
                                                    count = 2)
                                all_idx.add(idx)

                                if (any(_error > 10)):
                                    print(_error)
                                if idx in error:
                                    error[idx]["t"].append(time)
                                    error[idx]["v"] = np.concatenate([error[idx]["v"],_error])
                                else:
                                    _d = {"t":[time], "v":_error}
                                    error[idx] = _d

                            t0 = [ error[key]["t"][0] for  key in error]
                            t0 = min(t0)
                            for i in error:
                                error[i]["v"] = error[i]["v"].reshape((-1,2))
                                # error[i]["v"] = error[i]["v"].T
                                error[i]["t"] = [t - t0 for t in error[i]["t"]]
                        self.error[label][k] = error

            t0 = None # Used for integral time phasing
            name = os.path.join(self.directory ,f"error_{label}.dat")
            if os.path.exists(name):
                length = os.path.getsize(name)
                if length > 0:
                    with open(name, 'rb') as fileH:
                        size = 3*b_double + b_int
                        rows = (length) / size
                        rows = int(np.floor(rows))

                        error = {}

                        for i in range (rows):
                            time = np.fromfile(fileH,
                                                dtype = np.float64,
                                                count = 1)
                            time = time[0]
                            idx = np.fromfile(fileH,
                                                dtype = np.int64,
                                                count = 1)
                            idx = idx[0]
                            _error = np.fromfile(fileH,
                                                dtype = np.float64,
                                                count = 2)
                            all_idx.add(idx)

                            if (any(_error > 10)):
                                print(_error)
                            if idx in error:
                                error[idx]["t"].append(time)
                                error[idx]["v"] = np.concatenate([error[idx]["v"],_error])
                            else:
                                _d = {"t":[time], "v":_error}
                                error[idx] = _d

                        t0 = [ error[key]["t"][0] for  key in error]
                        t0 = min(t0)
                        for i in error:
                            error[i]["v"] = error[i]["v"].reshape((-1,2))
                            # error[i]["v"] = error[i]["v"].T
                            error[i]["t"] = [t - t0 for t in error[i]["t"]]
                    self.error[label][label] = error


        if self.error[label][label] is None and any([not _dict is None for _dict in self.error[label] ]) :
            #   Sum error
            all_idx = list(all_idx)
            all_idx.sort()

            #   Join time
            t = set()
            for _dict in self.error[label]: #   For each agent
                # print(_dict)
                if _dict is None:
                    continue
                for idx in _dict:   # for each idx
                    # print(idx)
                    for _t in _dict[idx]['t']: # For each time step
                        t.add(_t)
            t = list(t)
            t.sort()    #   Just in case

            # Sum error
            new_error = np.zeros((len(t),2*len(all_idx)))
            for i in range(len(t)):
                _v = np.zeros(2*len(all_idx)) # _v the error at a time step
                for _dict in self.error[label]: #   For each agent
                    if not _dict is None:
                        for idx in _dict:   # for each aruco
                            if t[i] in _dict[idx]['t']:  #  get slice of error and add to _v
                                t_id = _dict[idx]['t'].index(t[i])
                                v_id = all_idx.index(idx)
                                _v[v_id*2 : v_id*2+2] += _dict[idx]['v'][t_id]
                new_error[i,:] = _v # Tal vez copy
            t0 = t[0]
            self.error[label][label] = {'t': [_t-t0 for _t in t], 'v': new_error}


        self.error_int = [None]*self.n_agents
        # all_idx = set()
        # for k in range(n):
        #     if k != label:
        #         name = os.path.join(self.directory ,f"error_int_{label}_{k}.dat")
        #
        #         if os.path.exists(name):
        #             length = os.path.getsize(name)
        #             if length > 0:
        #                 with open(name, 'rb') as fileH:
        #                     size = 9*d + 8
        #                     rows = (length) / size
        #                     rows = int(np.floor(rows))
        #
        #                     error_int[k] = {}
        #
        #                     for i in range (rows):
        #                         time = np.fromfile(fileH,
        #                                             dtype = np.float64,
        #                                             count = 1)
        #                         time = time[0]
        #                         idx = np.fromfile(fileH,
        #                                             dtype = np.int64,
        #                                             count = 1)
        #                         idx = idx[0]
        #                         all_idx.add(idx)
        #                         _error = np.fromfile(fileH,
        #                                             dtype = np.float64,
        #                                             count = 8)
        #                         if (any(_error > 10)):
        #                             print(_error)
        #                         if idx in error_int[k]:
        #                             error_int[k][idx]["t"].append(time)
        #                             error_int[k][idx]["v"] = np.concatenate([error_int[k][idx]["v"],_error])
        #                         else:
        #                             _d = {"t":[time], "v":_error}
        #                             error_int[k][idx] = _d
        #
        #                     # t0 = [ error_int[k][key]["t"][0] for  key in error_int[k]]
        #                     # t0 = min(t0)
        #                     for i in error_int[k]:
        #                         error_int[k][i]["v"] = error_int[k][i]["v"].reshape((-1,8))
        #                         # error_int[k][i]["v"] = error_int[k][i]["v"].T
        #                         # error_int[k][i]["t"] = [t - t0 for t in error_int[k][i]["t"]]
        #
        # if any([not _dict is None for _dict in error_int ]):
        #     #   Sum error
        #     all_idx = list(all_idx)
        #     all_idx.sort()
        #     error_int[label] = {}
        #
        #     #   Join time
        #     t = set()
        #     for _dict in error_int: #   For each agent
        #         # print(_dict)
        #         for idx in _dict:   # for each aruco
        #             # print(idx)
        #             for _t in _dict[idx]['t']: # For each time step
        #                 t.add(_t)
        #     t = list(t)
        #     t.sort()    #   Just in case
        #
        #     # Sum error
        #     new_error = np.zeros((len(t),8*len(all_idx)))
        #     for i in range(len(t)):
        #         _v = np.zeros(8*len(all_idx)) # _v the error at a time step
        #         for _dict in error_int: #   For each agent
        #             for idx in _dict:   # for each aruco
        #                 if t[i] in _dict[idx]['t']:  #  get slice of error and add to _v
        #                     t_id = _dict[idx]['t'].index(t[i])
        #                     v_id = all_idx.index(idx)
        #                     _v[v_id*8 : v_id*8+8] += _dict[idx]['v'][t_id]
        #         new_error[i,:] = _v # Tal vez copy
        #     if t0 is None:
        #         t0 = t[0]
        #     error_int = {'t': [_t-t0 for _t in t], 'v': new_error}
        # else:
        #     error_int = None

        name = os.path.join(self.directory ,f"log_{label}.dat")
        self.log[label] = [None]*self.n_agents
        for k in range(self.n_agents):
            if k != label:
                if os.path.exists(name):
                    length = os.path.getsize(name)
                    if length > 0:
                        with open(name, 'rb') as fileH:
                            #   header
                            size = 1+6    # 6 dof only singular values
                            rows = (length) / (size* b_double)
                            rows = int(np.floor(rows))
                            log = np.fromfile(fileH,
                                                    dtype = np.float64,
                                                    count = size*rows)
                            log = log.reshape((rows,size))
                            log = log.T
                        log[0,:] -= log[0,0]
                        self.log[label][k] = log

    def plot_single(self, i = None ):

        sufx = "" if i is None else "_"+str(i)

        if not i is None:
            print(f"\tPloting Agent {i}")

        if not self.n_e is None:
            print("Ploting Error Norm")
            # plotNErr(self.directory, n_e, f"Error{sufx}.pdf")
            plotError(self.directory, self.n_e, f"Error{sufx}.pdf", th = 0.1)
        if not self.velocities is None:
            print("Ploting Velocities ")
            plotVel(self.directory, self.velocities,
                    f"Velocities{sufx}.pdf",
                    lims = self.vel_limits)
        if not self.velocities_log[0] is None:
            print("Ploting Velocities (Log Proportional) ")
            plotVel(self.directory, self.velocities_log[0],
                    f"Velocities_prop{sufx}.pdf",
                    lims = self.vel_limits)
        # if not velocities_log[1] is None:
        #     print("Ploting VELOCITIES Log Integral ")
        #     plotVel(self.directory, velocities_log[1], f"Velocities_int{sufx}.pdf")
        if not self.features is None:
            print("Ploting Features")
            plotFeat(self.directory,  self.features, f"Features{sufx}.pdf")

        if not self.error[i][i] is None:
            print("Ploting Error")
            plotError(self.directory,
                      self.error[i][i],
                      f"Error_feature{sufx}.pdf",
                      lims = self.error_limits)
            for j in range(self.n_agents):
                if j != i and not self.error[i][j] is None:
                    # print(self.error[i][j])
                    plotError(self.directory,
                            self.error[i][j],
                            f"Error_feature{sufx}_{j}.pdf",
                            lims = self.error_limits)
            # print(error[i])
        # if not error_int is None:
        #     print("Ploting Integral Error")
        #     plotError(self.directory, error_int, f"Error_int{sufx}.pdf", lims = [-1,1])
        if not self.position[i] is None:
            print("Ploting pose graphics")
            plotPosition(self.directory, self.position[i][[0,1,2,3,6],:], f"State{sufx}.pdf")

        for j in range(self.n_agents):
            if not self.log[i][j] is None:
                print("Ploting Log")
                plotLog(self.directory, self.log[j], f"LOG_SVD_D{sufx}_{j}.pdf")

    #   Only for arucos
    #   TODO: Adapt
    def join_error(self):

        if any([(i is None) for i in self.error]):
            return

        #   Join time
        t = self.error[0]['t']
        new_error = self.error[0]['v'].copy()
        idx = [0 for i in range(len(self.error))]
        for i in range(len(t)):
            for j in range(1,len(self.error)):
                while idx[j] < len(self.error[j]['t']) and self.error[j]['t'][idx[j]] < t[i] :
                    idx[j] += 1

                if idx[j] == 0:
                    new_error[i,:] +=  self.error[j]['v'][0]
                elif idx[j] >= len(self.error[j]['t']):
                    new_error[i,:] +=  self.error[j]['v'][-1]
                else:
                    delta = t[i] - self.error[j]['t'][idx[j]-1]
                    delta /= self.error[j]['t'][idx[j]] - self.error[j]['t'][idx[j]-1]
                    new_error[i,:] +=  self.error[j]['v'][idx[j]-1]
                    new_error[i,:] +=  delta * (self.error[j]['v'][idx[j]] - self.error[j]['v'][idx[j]-1] )


        self.joined_error =  {'t': t, 'v': new_error}

    def get_formation_error(self):

        if any([(i is None) for i in self.position]):
            return None

        #   Join time
        n = len(self.position)
        t = self.position[0][0,:]
        error = np.zeros((t.shape[0],2))
        idx = [0 for i in range(n)]
        agents = [Camera() for i in range(n)]
        for i in range(len(t)):
            agents[0].pose(self.position[0][1:,i])
            for j in range(1,n):
                while idx[j] < self.position[j].shape[1] and self.position[j][0,idx[j]] < t[i] :
                    idx[j] += 1

                if idx[j] == 0:
                    _position =  self.position[j][1:,0]
                elif idx[j] >= self.position[j].shape[1]:
                    _position =  self.position[j][1:,-1]
                else:
                    delta = t[i] - self.position[j][0,idx[j]-1]
                    delta /= self.position[j][0,idx[j]] - self.position[j][0,idx[j]-1]
                    _position =  self.position[j][1:,idx[j]-1]
                    _position +=  delta * (self.position[j][1:,idx[j]] - self.position[j][1:,idx[j]-1] )
                # print(_position)
                agents[j].pose(_position)

            error[i,:] =  error_state_6(self.pd,  agents)

        #   For plot
        self.agents = agents
        # error_state_6(self.pd,  agents, name = name)
        self.formation_error = {'t': t, 'v': error}


    def plot_error_state(self, name):
        error_state_6(self.pd,  self.agents, name = name)

    def fit_position(self):

        position = [None]*self.n_agents
        for i in range(self.n_agents):
            steps = self.position[i].shape[1]
            _p = (self.position[i][:4,:],  np.zeros((2,steps)), self.position[i][4,:].reshape((1,-1)))
            _p = np.concatenate(_p)
            _p[4,:] = -self.camera_angle
            _p[6,:] -= pi
            position[i] = _p
        self.position_fited = position

    def proc_joined(self):

        if any([not x is None for x in self.arucos]):
            self.join_error()

        self.get_formation_error()
        self.fit_position()

    def plot_joined(self):

        if not self.joined_error is None:
            print("Ploting Joined Error")
            plotError(self.directory, self.joined_error, f"Error_joined.pdf")

        name = os.path.join(self.directory,'Error_final.pdf')
        self.plot_error_state( name = name)
        # formation_error = self.get_formation_error(name)
        if not self.formation_error is None:
            print("Ploting Joined Error")
            plotError(self.directory, self.formation_error, f"Formation_error.pdf", th = 0.0)

        if not any([( i is None) for i in self.position_fited]):
            print("Ploting 3D")
            plot3D(self.directory, self.position_fited, self.pd, f"3DPlot.pdf")

#   AUXILIARY FUNCTIONS

def get_reference(image_name,
                  markers = cv2.aruco.DICT_APRILTAG_36h11,
                  out_directory = '',
                  label = 0):
    image = cv2.imread(f"{image_name}_{label}.png")
    arucos = None

    arucoDict = cv2.aruco.getPredefinedDictionary(markers)
    detectorParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, detectorParams)
    (corners, ids, rejected) = detector.detectMarkers(image)

    print("Detected markers:", ids)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imwrite(os.path.join(out_directory,f"ref_arucos_{label}.png"), image)

        ids = [k[0] for k in ids ]

    return ids, corners

def plot_descriptors_simple(ax,
                            descriptors_array,
                            camera_iMsize,
                            enableLims = True):

    n = descriptors_array.shape[0]/2
    n = int(n)

    # source_path = Path(__file__).resolve()
    source_dir = os.path.dirname(__file__)

    npzfile = np.load(source_dir +"/general.npz")
    colors = npzfile["colors"]
    nColors = colors.shape[0]

    if enableLims:
        plt.xlim([0,camera_iMsize[0]])
        plt.ylim([0,camera_iMsize[1]])

    ax.plot([camera_iMsize[0]/2,camera_iMsize[0]/2],
            [0,camera_iMsize[1]],
            color=[0.25,0.25,0.25])
    ax.plot([0,camera_iMsize[0]],
            [camera_iMsize[1]/2,camera_iMsize[1]/2],
            color=[0.25,0.25,0.25])

    for i in range(n):
        ax.plot(descriptors_array[2*i,:],descriptors_array[2*i+1,:],
                color=colors[i%nColors], lw = 0.5)
    for i in range(n):
        ax.plot(descriptors_array[2*i,0],descriptors_array[2*i+1,0],
                '*',color=colors[i%nColors], mec = 'k')
        ax.plot(descriptors_array[2*i,-1],descriptors_array[2*i+1,-1],
                'o',color=colors[i%nColors], mec = 'k')

    return

#   TODO: my math py

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


def error_state_6(reference,
                agents,
                name= None,
                fontsize = 10.):

    n = len(agents)

    state_t = np.zeros((3,n))
    state_r = np.zeros((3,n))
    for i in range(len(agents)):
        state_t[:,i] = agents[i].p[:3]
        state_r[:,i] = agents[i].p[3:]

    #   Regularization to centroid
    state_c = state_t - state_t.mean(axis = 1).reshape((-1,1))
    ref_c = reference[:3,:] - reference[:3,:].mean(axis = 1).reshape((-1,1))
    ref_c /= norm(ref_c,axis = 0).mean()

    M = state_c.T.reshape((n,1,3))
    D = ref_c.T.reshape((n,3,1))
    H = D @ M
    H = H.sum(axis = 0)

    U, S, VH = svd(H)
    R = VH.T @ U.T

    #   Caso de Reflexión
    if np.linalg.det(R) < 0.:
        VH[2,:] = -VH[2,:]
        R = VH.T @ U.T

    #   Aligning
    state_c = R.T @ state_c

    #   translation error
    f = lambda r : (norm(ref_c - r*state_c,axis = 0)**2).sum()/n
    r_state = minimize_scalar(f, method='brent')
    t_err = f(r_state.x)
    t_err = np.sqrt(t_err)

    #   Scaling
    state_c = r_state.x * state_c

    #   Rotation error
    rot_err = np.zeros(n)
    for i in range(n):
        # print("R>")
        _R = R.T @ agents[i].R
        state_r[:,i] = get_angles(_R)
        _R = rotation_matrix_euler(reference[3:,i]).T @ _R
        #agents[i].pose(new_state[:,i])

        # print("<R")

        #   Get error
        #_R =  cm.rot(new_reference[3,i],'x') @ agents[i].R.T
        #_R = cm.rot(new_reference[4,i],'y') @ _R
        #_R = cm.rot(new_reference[5,i],'z') @ _R
        #_R = rotation_matrix_euler(reference[:,i]).T
        #_R = rotation_matrix_euler(state_r[:,i]).T @ _R

        _arg = (_R.trace()-1.)/2.
        if abs(_arg) < 1.:
            rot_err[i] = np.arccos(_arg)
        else:
            rot_err[i] = np.arccos(np.sign(_arg))

    #   rms
    rot_err = rot_err**2
    rot_err = rot_err.sum()/n
    rot_err = np.sqrt(rot_err)

    if name is None:
        return np.array([t_err, rot_err])

     ##   Plot
    matplotlib.rcParams["mathtext.fontset"] = 'cm'
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(elev=30, azim=165)
    plot_aligned(ax, np.stack( (state_c, state_r)).reshape((6,-1)) ,
                 np.stack( (ref_c, reference[3:,:])).reshape((6,-1)) ,
                 fontsize = fontsize)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    plt.savefig(name,bbox_inches='tight')
    # plt.show()
    plt.close()

    return np.array([t_err, rot_err])

#   My plots

def plot_aligned(ax, state, ref, fontsize = 10):
    camera = Camera()
    for i in range(state.shape[1]):

        #   New pose
        camera.pose(state[:,i])
        ax.plot([camera.p[0],camera.p[0]],
                [camera.p[1],camera.p[1]],
                [0,camera.p[2]],
                color = 'k', linestyle=(0, (5, 10)),lw = 0.5)
        camera.draw_camera(ax, scale=0.2, color='green', lw=1.1)
        ax.text(camera.p[0],
                camera.p[1],
                camera.p[2],
                str(i), fontsize = fontsize)

        camera.pose(ref[:,i])
        ax.plot([camera.p[0],camera.p[0]],
                [camera.p[1],camera.p[1]],
                [0,camera.p[2]],
                color = 'k', linestyle=(0, (5, 10)),lw = 0.5)
        camera.draw_camera(ax, scale=0.2, color='red',
                                        linestyle = (0, (5, 10)) , lw = .7)
        ax.text(camera.p[0],
                camera.p[1],
                camera.p[2],
                str(i), fontsize = fontsize)

def plot_time(ax, t_array,
              var_array,
              ref = None,
              color_offset = 0,
              module = None,
              lw = .6):

    n = var_array.shape[0]

    # source_path = Path(__file__).resolve()
    source_dir = os.path.dirname(__file__)

    npzfile = np.load(source_dir +"/general.npz")
    colors = npzfile["colors"]
    nColors = colors.shape[0]



    symbols = []
    for i in range(n):
        ax.plot(t_array,var_array[i,:] ,
                color=colors[(color_offset+i)%nColors], lw = lw )
        symbols.append(mpatches.Patch(color=colors[(color_offset+i)%nColors]))

    if not ref is None:
        ax.plot([t_array[0],t_array[-1]],[ref,ref],
                'k--', alpha = 0.5)
        symbols.append(mpatches.Patch(color='k'))
        # labels.append(refLab)
    if not module is None:
        for i in range(len(module)):
            ax.plot([t_array[0],t_array[-1]],[module[i],module[i]],
                'r--', lw = 0.5)
        symbols.append(mpatches.Patch(color='r'))
        # labels.append("Limits")

    return symbols





 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #              PLOT TASKS




def plotPosition(directory, data, name):

    time = data[0,:]
    positions = data[1:,:]

    #  plot positions
    labels = ["X","Y","Z","Yaw"]
    fig_p, ax_p = plt.subplots(nrows = 1, figsize=(5,5))
    fig_p.suptitle("State")
    symbols = plot_time(ax_p, time, positions, color_offset = 1)
    ax_p.legend(symbols,labels, loc=1)
    # ax_p.set_ylim((-0.06,0.06))
    # plt.show()
    name = os.path.join(directory ,name)
    plt.savefig(name,bbox_inches='tight')
    plt.close()

def plotVel(directory, data, name, lims = [-1.1,1.1]):

    time = data[0,:]
    velocities = data[1:,:]



    #  plot Velocities
    labels = ["X","Y","Z","Yaw"]
    fig_v, ax_v = plt.subplots(nrows = 1, figsize=(5,5))
    fig_v.suptitle("Velocities")
    symbols = plot_time(ax_v, time, velocities, color_offset = 1)
    ax_v.legend(symbols,labels, loc=1)
    ax_v.set_ylim(lims)
    # plt.show()
    name = os.path.join(directory ,name)
    plt.savefig(name,bbox_inches='tight')
    plt.close()


# def plotNErr(directory, data, name):
#
#     time = data[0,:]
#     error = data[1,:]
#
#
#         #   Plot error
#     fig_e, ax_e = plt.subplots( figsize=(6,2))
#     fig_e.suptitle("Error")
#     plot_time(ax_e, time,error.reshape((1,-1)) )
#
#     print("Minimun error= "+ str(error.min()))
#     name = os.path.join(directory ,name)
#     plt.savefig(name,bbox_inches='tight')
#     plt.close()

def plotError(directory, error, name, th = 0., lims = None):

    fig, ax = plt.subplots( figsize=(6,2))
    fig.suptitle("Error")

    if "t" in error :
        time = np.array(error["t"])
        _error = np.array(error["v"].T).copy()
        plot_time(ax, time, _error, th)
    else:
        offset = 0
        for k in error:
            # print(_dict)
            time = np.array(error[k]["t"])
            _error = np.array(error[k]["v"].T).copy()
            plot_time(ax,
                        time,_error,
                        th, color_offset = offset )
            offset += _error.shape[0]

    if not lims is None:
        ax.set_ylim(lims)
    name = os.path.join(directory ,name)
    plt.savefig(name ,bbox_inches='tight')
    plt.close()


def plotFeat(directory, features, name): #, reference):

    camera_iMsize = [856,480]
    fig, ax = plt.subplots()
    fig.suptitle("Features")
    for i, v in features.items():
        # if i in reference[0]:
        #     k = reference[0].index(i)
            # s_ref = reference[1][k].reshape(8)
        points = np.array(v["v"]).reshape((2,-1))
        symbols = plot_descriptors_simple(ax,
                            points,
                            # s_ref,
                        camera_iMsize,
                        enableLims = True)
    labels = ["Start","End","Reference","trayectory"]
    symbols = [mlines.Line2D([0],[0],marker='*',color='k'),
               mlines.Line2D([0],[0],marker='o',color='k'),
               mlines.Line2D([0],[0],marker='^',color='k'),
               mlines.Line2D([0],[0],linestyle='-',color='k')]
    fig.legend(symbols,labels, loc=1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    name = os.path.join(directory ,name)
    plt.savefig(name,bbox_inches='tight')
    #plt.show()
    plt.close()


def plot3D(directory, allStates, pd, name):
    n_agents = len(allStates)

    source_dir = os.path.dirname(__file__)
    npzfile = np.load(source_dir +"/general.npz")
    colors = npzfile["colors"]
    nColors = colors.shape[0]


    fig, ax = plt.subplots(ncols = 2,
                           frameon=False,
                           figsize=(8,6),
                           #figsize=(12,9),
                            gridspec_kw={'width_ratios': [3,1]})
    #fig = plt.figure(frameon=False, figsize=(5,3))
    ax[0].axis('off')
    ax[0] = fig.add_subplot(1, 2, 1, projection='3d')
    name = os.path.join(directory ,name)

    x_min = allStates[0][1,0]
    x_max = allStates[0][1,0]
    y_min = allStates[0][2,0]
    y_max = allStates[0][2,0]
    z_min = allStates[0][3,0]
    z_max = allStates[0][3,0]

    camera = Camera()
    for i in range(n_agents):
        pos_array = allStates[i][1:,:]
        init = pos_array[:,0]
        end = pos_array[:,-1]
        x_min = min(x_min, init[0], end[0])
        x_max = max(x_max, init[0], end[0])
        y_min = min(y_min, init[1], end[1])
        y_max = max(y_max, init[1], end[1])
        z_min = min(z_min, init[2], end[2])
        z_max = max(z_max, init[2], end[2])

        camera.plot_3Dcam(ax[0],
                    pos_array,
                    pd[:,i],
                    color = colors[i+1],
                    label = str(i),
                    camera_scale = 0.05)



    width = x_max - x_min
    height = y_max - y_min
    depth = z_max - z_min
    sqrfact = max(width,height,depth)

    x_min -= (sqrfact - width )/2
    x_max += (sqrfact - width )/2
    y_min -= (sqrfact - height )/2
    y_max += (sqrfact - height )/2
    z_min -= (sqrfact - depth )/2
    z_max += (sqrfact - depth )/2
    ax[0].set_xlim(x_min,x_max)
    ax[0].set_ylim(y_min,y_max)
    ax[0].set_zlim(z_min,z_max)

    #ax = fig.add_subplot(1, 2, 2)
    symbols = []
    labels = ["Agent "+ str(i) for i in range(n_agents)]
    for i in range(n_agents):
        symbols.append(mpatches.Patch(color=colors[(i+1)%colors.shape[0]]))
    ax[1].legend(symbols,labels, loc=7)
    ax[1].axis('off')

    #fig.legend( loc=1)
    plt.savefig(name)
    # plt.show()
    plt.close()


def plotLog(directory, log, name):

    # print(state.shape)

    time = log[0,:]
    svd = log[1:,:]
    # cutoff = 4*2*4+1 # 4 dof
    # cutoff = 6*2*4+1 # 6 dof
    # interaction = log[1:cutoff,:]
    # svd = log[cutoff:-1,:]


    # #  plot interaction matrix components
    # # print(time.shape)
    # fig_v, ax_v = plt.subplots(nrows = 1, figsize=(5,5))
    # fig_v.suptitle("Interaction matrix")
    # for i in range (6):
    #     symbols = plot_time(ax_v, time, interaction[i:i*8], color_offset = 1)
    # # ax_v.set_ylim([-.5,.5])
    # # plt.show()
    # labels = [str(i) for i in range(8)]
    # ax_v.legend(symbols,labels, loc=1)
    # plt.savefig(directory +"LOG_Interaction.pdf",bbox_inches='tight')
    # plt.close()

    #  plot singular values
    labels = [str(i) for i in range(6)]
    fig_p, ax_p = plt.subplots(nrows = 1, figsize=(5,5))
    fig_p.suptitle("Singular values L.T L ")
    symbols = plot_time(ax_p, time, svd , color_offset = 1)
    ax_p.legend(symbols,labels, loc=1)
    # ax_p.set_ylim((-0.06,0.06))
    # plt.show()
    name = os.path.join(directory ,name)
    plt.savefig(name ,bbox_inches='tight')
    plt.close()




def main(arg):

    ploter = Ploter(arg)

    for i in range(ploter.n_agents):
        ploter.read_data(i)
        ploter.plot_single(i)
    ploter.proc_joined()
    ploter.plot_joined()





if __name__ ==  "__main__":
    description = "Plotting multiple agent experiment data"
    parser = argparse.ArgumentParser(prog = 'python3 plot_mav_desc.py',
                                     description = description)
    parser.add_argument( 'directory', type=str, default = 'output',
        help = "Directory name (default output/)")
    parser.add_argument( '--config', type=str,
        default = 'config/mav_sim.yaml',
        help = "File containign the configurations (defaultconfig/mav_sim.yaml)")

    arg = parser.parse_args()
    main(arg)
