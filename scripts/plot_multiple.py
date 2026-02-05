
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
                            s_ref,
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
        ax.plot(s_ref[i*2],s_ref[i*2+1],
                marker='^',color=colors[i%nColors], mec = 'k')
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
        _R = R.T @ agents[i].R
        state_r[:,i] = get_angles(_R)
        _R = rotation_matrix_euler(reference[:,i]).T @ _R
        #agents[i].pose(new_state[:,i])

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
    #plt.show()
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

def plotVel(directory, data, name):

    time = data[0,:]
    velocities = data[1:,:]



    #  plot Velocities
    labels = ["X","Y","Z","Yaw"]
    fig_v, ax_v = plt.subplots(nrows = 1, figsize=(5,5))
    fig_v.suptitle("Velocities")
    symbols = plot_time(ax_v, time, velocities, color_offset = 1)
    ax_v.legend(symbols,labels, loc=1)
    ax_v.set_ylim([-2.1,2.1])
    # plt.show()
    name = os.path.join(directory ,name)
    plt.savefig(name,bbox_inches='tight')
    plt.close()


def plotNErr(directory, data, name):

    time = data[0,:]
    error = data[1,:]


        #   Plot error
    fig_e, ax_e = plt.subplots( figsize=(6,2))
    fig_e.suptitle("Error")
    plot_time(ax_e, time,error.reshape((1,-1)) )

    print("Minimun error= "+ str(error.min()))
    name = os.path.join(directory ,name)
    plt.savefig(name,bbox_inches='tight')
    plt.close()

def plotError(directory, error, name):

    fig, ax = plt.subplots( figsize=(6,2))
    fig.suptitle("Error")
    time = np.array(error["t"])
    _error = np.array(error["v"].T).copy()
    plot_time(ax, time,_error, 0.1 )
    # ax.set_ylim([-.5,.5])
    name = os.path.join(directory ,name)
    plt.savefig(name ,bbox_inches='tight')
    ax.set_ylim([-.81,.81])
    plt.close()


def plotArucos(directory, arucos, reference, name):

    #   Plot error
    # fig_e, ax_e = plt.subplots( figsize=(6,2))
    # fig_e.suptitle("Error")
    # for i, v in arucos.items():
    #     if i in reference[0]:
    #         k = reference[0].index(i)
    #         time = np.array(v["t"])
    #         error = np.array(v["v"]).copy()
    #         error = error.reshape((8,-1))
    #         s_ref = reference[1][k].reshape((8,-1))
    #         error = error - s_ref
    #         plot_time(ax_e, time,error )
    # name = os.path.join(directory ,f"Error_arucos_{label}.pdf")
    # plt.savefig(name,bbox_inches='tight')
    # plt.close()

    camera_iMsize = [856,480]
    fig, ax = plt.subplots()
    fig.suptitle("ArUcos")
    for i, v in arucos.items():
        if i in reference[0]:
            k = reference[0].index(i)
            points = np.array(v["v"]).reshape((8,-1))
            s_ref = reference[1][k].reshape(8)
            symbols = plot_descriptors_simple(ax,
                                points,
                                s_ref,
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




 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #          READ DATA




def read_data(directory,label, n):

    f = 4
    d = 8
    _i = 4
    
    name = os.path.join(directory ,f"position_{label}.dat")
    position = None
    if os.path.exists(name):
        length = os.path.getsize(name)
        if length > 0:
            with open(name, 'rb') as fileH:
                rows = (length) / (5* d)
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
                position[5,:] = pi/2.
                # position[6,:] -= pi/2.

    name = os.path.join(directory ,f"velocities_{label}.dat")
    velocities = None
    if os.path.exists(name):
        length = os.path.getsize(name)
        if length > 0:
            with open(name, 'rb') as fileH:
                rows = (length) / (5* d)
                rows = int(np.floor(rows))
                velocities = np.fromfile(fileH,
                                        dtype = np.float64,
                                        count = 5*rows)
                velocities = velocities.reshape((rows,5))
                velocities = velocities.T
                velocities[0,:] -= velocities[0,0]

    name = os.path.join(directory ,f"norm_error_{label}.dat")
    n_e = None
    if os.path.exists(name):
        length = os.path.getsize(name)
        if length > 0:
            with open(name, 'rb') as fileH:
                rows = (length) / (2* d)
                rows = int(np.floor(rows))
                n_e = np.fromfile(fileH,
                                        dtype = np.float64,
                                        count = 2*rows)
                n_e = n_e.reshape((rows,2))
                n_e = n_e.T
                n_e[0,:] -= n_e[0,0]

    name = os.path.join(directory, f"arUcos_{label}.dat")
    arucos = None
    if os.path.exists(name):
        length = os.path.getsize(name)
        if length > 0:
            with open(name, 'rb') as fileH:
                size = d*9 + 8
                rows = (length) / size
                rows = int(np.floor(rows))

                arucos = {}

                for i in range (rows):
                    time = np.fromfile(fileH,
                                        dtype = np.float64,
                                        count = 1)
                    time = time[0]
                    idx = np.fromfile(fileH,
                                        dtype = np.int64,
                                        count = 1)
                    idx = idx[0]
                    # print(idx)
                    aruco = np.fromfile(fileH,
                                        dtype = np.float64,
                                        count = 8)
                    if idx in arucos:
                        arucos[idx]["t"].append(time)
                        arucos[idx]["v"] = np.concatenate([arucos[idx]["v"],aruco])
                    else:
                        _d = {"t":[time], "v":aruco}
                        arucos[idx] = _d
                t0 = [ arucos[key]["t"][0] for  key in arucos]
                t0 = min(t0)
                for i in arucos:
                    arucos[i]["v"] = arucos[i]["v"].reshape((-1,8))
                    arucos[i]["v"] = arucos[i]["v"].T
                    arucos[i]["t"] = [t - t0 for t in arucos[i]["t"]]

    error = [None]*n
    all_idx = set()
    for k in range(n):
        if k != label:
            name = os.path.join(directory ,f"error_{label}_{k}.dat")

            if os.path.exists(name):
                length = os.path.getsize(name)
                if length > 0:
                    with open(name, 'rb') as fileH:
                        size = 9*d + 8
                        rows = (length) / size
                        rows = int(np.floor(rows))

                        error[k] = {}

                        for i in range (rows):
                            time = np.fromfile(fileH,
                                                dtype = np.float64,
                                                count = 1)
                            time = time[0]
                            idx = np.fromfile(fileH,
                                                dtype = np.int64,
                                                count = 1)
                            idx = idx[0]
                            all_idx.add(idx)
                            _error = np.fromfile(fileH,
                                                dtype = np.float64,
                                                count = 8)
                            if (any(_error > 10)):
                                print(_error)
                            if idx in error[k]:
                                error[k][idx]["t"].append(time)
                                error[k][idx]["v"] = np.concatenate([error[k][idx]["v"],_error])
                            else:
                                _d = {"t":[time], "v":_error}
                                error[k][idx] = _d

                        # t0 = [ error[k][key]["t"][0] for  key in error[k]]
                        # t0 = min(t0)
                        for i in error[k]:
                            error[k][i]["v"] = error[k][i]["v"].reshape((-1,8))
                            # error[k][i]["v"] = error[k][i]["v"].T
                            # error[k][i]["t"] = [t - t0 for t in error[k][i]["t"]]
    #   Sum error
    all_idx = list(all_idx)
    all_idx.sort()
    error[label] = {}

    #   Join time
    t = set()
    for _dict in error: #   For each agent
        # print(_dict)
        for idx in _dict:   # for each aruco
            # print(idx)
            for _t in _dict[idx]['t']: # For each time step
                t.add(_t)
    t = list(t)
    t.sort()    #   Just in case

    # Sum error
    new_error = np.zeros((len(t),8*len(all_idx)))
    for i in range(len(t)):
        _v = np.zeros(8*len(all_idx)) # _v the error at a time step
        for _dict in error: #   For each agent
            for idx in _dict:   # for each aruco
                if t[i] in _dict[idx]['t']:  #  get slice of error and add to _v
                    t_id = _dict[idx]['t'].index(t[i])
                    v_id = all_idx.index(idx)
                    _v[v_id*8 : v_id*8+8] += _dict[idx]['v'][t_id]
        new_error[i,:] = _v # Tal vez copy
    t0 = t[0]
    error = {'t': [_t-t0 for _t in t], 'v': new_error}

    name = os.path.join(directory ,f"log_{label}.dat")
    log = [None]*n
    for k in range(n):
        if k != label:
            if os.path.exists(name):
                length = os.path.getsize(name)
                if length > 0:
                    with open(name, 'rb') as fileH:
                        #   header
                        size = 1+6    # 6 dof only singular values
                        rows = (length) / (size* d)
                        rows = int(np.floor(rows))
                        log[k] = np.fromfile(fileH,
                                                dtype = np.float64,
                                                count = size*rows)
                        log[k] = log[k].reshape((rows,size))
                        log[k] = log[k].T
                    log[k][0,:] -= log[k][0,0]

    return position, velocities, n_e, arucos, error, log

 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #          READ MAIN

def get_pd(name):
    with open(name, 'r') as file:
        _dict = yaml.safe_load(file)

    pd = np.array(_dict['pd'])
    pd = pd.reshape((-1,4))
    n = pd.shape[0]

    pd = (pd[:,:3],  np.zeros((n,2)), pd[:,3].reshape((-1,1)))
    pd = np.concatenate(pd, axis = 1)
    # pd[:,3] = pi/2.
    pd[:,4] = pi/2.
    # pd[:,5] -= pi/2.
    return pd.T

def join_error(error):

    if any([(i is None) for i in error]):
        return None

    #   Join time
    t = error[0]['t']
    new_error = error[0]['v'].copy()
    idx = [0 for i in range(len(error))]
    for i in range(len(t)):
        for j in range(1,len(error)):
            # print( error[j])
            # print( error[j]['t'])
            # print( j)
            # print( idx[j])
            # print( error[j]['t'][idx[j]])
            while idx[j] < len(error[j]['t']) and error[j]['t'][idx[j]] < t[i] :
                idx[j] += 1

            if idx[j] == 0:
                new_error[i,:] +=  error[j]['v'][0]
            if idx[j] >= len(error[j]['t']):
                new_error[i,:] +=  error[j]['v'][-1]
            else:
                delta = t[i] - error[j]['t'][idx[j]-1]
                delta /= error[j]['t'][idx[j]] - error[j]['t'][idx[j]-1]
                new_error[i,:] +=  error[j]['v'][idx[j]-1]
                new_error[i,:] +=  delta * (error[j]['v'][idx[j]] - error[j]['v'][idx[j]-1] )


    return {'t': t, 'v': new_error}

def get_formation_error(position, pd, name):

    if any([(i is None) for i in position]):
        return None

    #   Join time
    n = len(position)
    t = position[0][0,:]
    error = np.zeros((t.shape[0],2))
    idx = [0 for i in range(n)]
    agents = [Camera() for i in range(n)]
    for i in range(len(t)):
        agents[0].pose(position[0][1:,i])
        for j in range(1,n):
            while idx[j] < position[j].shape[1] and position[j][0,idx[j]] < t[i] :
                idx[j] += 1

            if idx[j] == 0:
                _position =  position[j][1:,0]
            if idx[j] >= position[j].shape[1]:
                _position =  position[j][1:,-1]
            else:
                delta = t[i] - position[j][0,idx[j]-1]
                delta /= position[j][0,idx[j]] - position[j][0,idx[j]-1]
                _position =  position[j][1:,idx[j]-1]
                _position +=  delta * (position[j][1:,idx[j]] - position[j][1:,idx[j]-1] )
            # print(_position)
            agents[j].pose(_position)

        error[i,:] =  error_state_6(pd,  agents)

    #   plot last
    error[i,:] =  error_state_6(pd,  agents, name = name)

    return {'t': t, 'v': error}

def fit_position(position):

    n = len(position)
    for i in range(n):
        steps = position[i].shape[1]
        _p = (position[i][1:4,:],  np.zeros((2,steps)), position[i][4,:].reshape((1,-1)))
        _p = np.concatenate(_p)
        _p[3,:] = -pi/2.
        _p[5,:] -= pi
        position[i] = _p
    return position

def main(arg):

    markers = markers_list.index(arg.markers)
    directory = arg.directory
    pd = get_pd(arg.desired)

    error = [None]*arg.n
    position = [None]*arg.n

    for i in range(arg.n):
        position[i], velocities, n_e, arucos, error[i], log = read_data(directory,i, arg.n)
        s_ref = get_reference(arg.reference,
                            markers = markers,
                            out_directory = directory ,
                            label = i)
        print(s_ref)
        if not n_e is None:
            print("Ploting  ")
            plotNErr(directory, n_e, f"Error_{i}.pdf")
        if not velocities is None:
            print("Ploting VELOCITIES ")
            plotVel(directory, velocities, f"Velocities_{i}.pdf")
        if not arucos is None:
            print("Ploting ArUcos")
            plotArucos(directory,  arucos, s_ref, f"ArUcos_{i}.pdf")
        if not error[i] is None:
            print("Ploting Error")
            plotError(directory, error[i], f"Error_runtime_{i}.pdf")
        if not position[i] is None:
            print("Ploting 3D")
            plotPosition(directory, position[i][[0,1,2,3,6],:], f"State_{i}.pdf")

        for j in range(arg.n):
            if not log[j] is None:
                print("Ploting LOG")
                plotLog(directory, log[j], f"LOG_SVD_D_{i}_{j}.pdf")

    jerror = join_error(error)
    if not jerror is None:
        print("Ploting Joined Error")
        plotError(directory, jerror, f"Error_joined.pdf")

    formation_error = get_formation_error(position, pd, f"Error_final.pdf")
    if not formation_error is None:
        print("Ploting Joined Error")
        plotError(directory, formation_error, f"Formation_error.pdf")

    # position= fit_position(position)
    if not any([( i is None) for i in position]):
        print("Ploting 3D plot")
        plot3D(directory, position, pd, f"3DPlot.pdf")




if __name__ ==  "__main__":
    description = "Plotting multiple agent experiment data"
    parser = argparse.ArgumentParser(prog = 'python3 miguel_iros.py',
                                     description = description)
    parser.add_argument( 'directory', type=str, default = 'output',
        help = "Directory name (default output/)")
    parser.add_argument( '--reference', type=str,
        default = 'config/reference_f',
        help = "IBVS reference image (default config/reference)")
    parser.add_argument( '--desired', type=str,
        default = 'config/ibfc_sim.yaml',
        help = "File containign the desired formation (pd:)")
    parser.add_argument( '--n', type=int, default = 4 ,
        help = "Number of agents (default 4)")
    parser.add_argument( '--markers', type=str, default = '6X6_1000',
        choices = markers_list,
        help = "Fiducial markers family (default 6X6_1000)")

    arg = parser.parse_args()
    main(arg)
