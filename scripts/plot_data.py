
#   libs
import numpy as np
from numpy import pi, sin, cos
from numpy.linalg import norm, svd
from scipy.optimize import minimize_scalar
import cv2

#   Plot
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import gridspec

#   LATEX FIX
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#   sys
import sys
import os
# from pathlib import Path

#   Custom
import camera as cm


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



#   Aux fun

def get_reference(image_name,
                  markers = cv2.aruco.DICT_APRILTAG_36h11,
                  out_directory = ''):
    image = cv2.imread(image_name)
    arucos = None

    # arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    arucoDict = cv2.aruco.getPredefinedDictionary(markers)
    detectorParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, detectorParams)
    (corners, ids, rejected) = detector.detectMarkers(image)

    print("Detected markers:", ids)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imwrite(os.path.join(out_directory,'ref_arucos.png'), image)

        ids = [k[0] for k in ids ]
        print("Detected ArUcos")

    return ids, corners

def plot_descriptors_simple(ax,
                            descriptors_array,
                            s_ref,
                            camera_iMsize,
                            enableLims = True):

    n = descriptors_array.shape[0]/2
    #print(n)
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

def plot_3Dcam(ax, 
               positionArray,
               init_configuration,
               end_configuration,
               desired_configuration,
               color, lw = 1,
               camera_scale    = 0.02,
               paper_1 = False):

    #   Trayectories
    ax.plot(positionArray[0,:],
    #ax.scatter(positionArray[0,:],
            positionArray[1,:],
            positionArray[2,:],
            c = color.reshape((1,3)),
            linewidth = 0.5)
            #s = 0.1 ) # Plot camera trajectory
    
    ##   Z axis refs
    #ax.plot([init_configuration[0],init_configuration[0]],
            #[init_configuration[1],init_configuration[1]],
            #[0,init_configuration[2]],
            #color = 'k', linestyle=(0, (5, 10)),lw = 0.5)
    #ax.plot([end_configuration[0],end_configuration[0]],
            #[end_configuration[1],end_configuration[1]],
            #[0,end_configuration[2]],
            #color = 'k', linestyle=(0, (5, 10)),lw = 0.5)
    #ax.plot([desired_configuration[0],desired_configuration[0]],
            #[desired_configuration[1],desired_configuration[1]],
            #[0,desired_configuration[2]],
            #color = 'k', linestyle=(0, (5, 10)), lw = 0.5)
    
    #   Cameras
    camera = cm.camera()
    camera.pose(end_configuration)
    camera.draw_camera(ax, scale=camera_scale, color='r', lw = lw)
    ax.text(camera.p[0],camera.p[1],camera.p[2],"f")
    camera.pose(desired_configuration)
    camera.draw_camera(ax, scale=camera_scale, color='g', 
                     lw = 0.5*lw)
    ax.text(camera.p[0],camera.p[1],camera.p[2],"*")
    camera.pose(init_configuration)
    camera.draw_camera(ax, scale=camera_scale, color='b', lw = lw)
    ax.text(camera.p[0],camera.p[1],camera.p[2],"0")
    
    
    
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.set_zlabel("$Z$")
    ax.grid(True)

#   My plots

def plot_time(ax, t_array,
              var_array,
              ref = None,
              color_offset = 0,
              module = None):

    n = var_array.shape[0]

    # source_path = Path(__file__).resolve()
    source_dir = os.path.dirname(__file__)

    npzfile = np.load(source_dir +"/general.npz")
    colors = npzfile["colors"]
    nColors = colors.shape[0]



    symbols = []
    for i in range(n):
        ax.plot(t_array,var_array[i,:] , color=colors[(color_offset+i)%nColors], lw = 0.6 )
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



##  Plot task

def plotArucos(directory, arucos, reference):

    print(arucos.keys())
    print(reference[0])

    #   Plot error
    fig_e, ax_e = plt.subplots( figsize=(6,2))
    fig_e.suptitle("Error")
    for i, v in arucos.items():
        if i in reference[0]:
            k = reference[0].index(i)
            time = np.array(v["t"])
            error = np.array(v["v"]).copy()
            error = error.reshape((8,-1))
            s_ref = reference[1][k].reshape((8,-1))
            error = error - s_ref
            plot_time(ax_e, time,error )
    plt.savefig(directory +"Error_arucos.pdf",bbox_inches='tight')
    plt.close()

    camera_iMsize = [856,480]
    fig, ax = plt.subplots()
    fig.suptitle("ArUcos")
    for i, v in arucos.items():
        print(f"Reference {i}")
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
    plt.savefig(directory+'ArUcos.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()

def plotError(directory,error):

    fig, ax = plt.subplots( figsize=(6,2))
    fig.suptitle("Error")
    for i, v in error.items():
        time = np.array(v["t"])
        error = np.array(v["v"]).copy()
        error = error.reshape((8,-1))
        plot_time(ax, time,error )
    # ax.set_ylim([-.5,.5])
    plt.savefig(directory +"Error_runtime.pdf",bbox_inches='tight')
    plt.close()



def plot(directory, state):

    # print(state.shape)

    time = state[0,:]
    positions = state[1:5,:]
    velocities = state[5:9,:]
    error = state[9,:]

        #   Plot error
    fig_e, ax_e = plt.subplots( figsize=(6,2))
    fig_e.suptitle("Error")
    plot_time(ax_e, time,error.reshape((1,-1)) )
    # print("Average error (t<20)  = "+str(np.average(error[time>20])))
    # ax_e.set_ylim((0.,0.18))
    print("Minimun error= "+ str(error.min()))
    plt.savefig(directory +"Error.pdf",bbox_inches='tight')
    plt.close()

    #  plot Velocities
    labels = ["X","Y","Z","Yaw"]
    # print(time.shape)
    fig_v, ax_v = plt.subplots(nrows = 1, figsize=(5,5))
    fig_v.suptitle("Velocities")
    symbols = plot_time(ax_v, time, velocities, color_offset = 1)
    ax_v.legend(symbols,labels, loc=1)
    ax_v.set_ylim([-.5,.5])
    # plt.show()
    plt.savefig(directory +"Velocities.pdf",bbox_inches='tight')
    plt.close()

    #  plot positions
    labels = ["X","Y","Z","Yaw"]
    fig_p, ax_p = plt.subplots(nrows = 1, figsize=(5,5))
    fig_p.suptitle("State")
    symbols = plot_time(ax_p, time, positions, color_offset = 1)
    ax_p.legend(symbols,labels, loc=1)
    # ax_p.set_ylim((-0.06,0.06))
    # plt.show()
    plt.savefig(directory +"State.pdf",bbox_inches='tight')
    plt.close()

def plotLog(directory, log):

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
    labels = ["X","Y","Z","Yaw"]
    fig_p, ax_p = plt.subplots(nrows = 1, figsize=(5,5))
    fig_p.suptitle("Singular values L.T L ")
    plot_time(ax_p, time, svd , color_offset = 1)
    # ax_p.set_ylim((-0.06,0.06))
    # plt.show()
    plt.savefig(directory +"LOG_SVD_D.pdf",bbox_inches='tight')
    plt.close()





def read_data(directory):

    f = 4
    d = 8
    _i = 4
    
    name = directory +'/state.dat'
    # name = directory +'/arUcos.dat'
    length = os.path.getsize(name)
    # print("length = ", length)

    with open(name, 'rb') as fileH:
        #   header
        # np.fromfile(fileH, dtype=np.int32, count= 1)
        # rows = (length-4) / row_bytes
        rows = (length) / (10* d)
        # print("rows = ",rows)
        rows = int(np.floor(rows))
        state = np.fromfile(fileH,
                                dtype = np.float64,
                                count = 10*rows)
        # state = state.reshape((rows,14))
        state = state.reshape((rows,10))
        # state = state[1:,:] # Trim
        state = state.T


    state[0,:] -= state[0,0]



    name = directory +'/arUcos.dat'
    length = os.path.getsize(name)
    # print("length = ", length)

    with open(name, 'rb') as fileH:
        #   header
        size = d*9 + 8
        rows = (length) / size
        # print("rows = ",rows)
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

    name = directory +'/error.dat'
    length = os.path.getsize(name)
    # print("length = ", length)

    with open(name, 'rb') as fileH:
        #   header
        size = 9*d + 8
        rows = (length) / size
        # print("rows = ",rows)
        rows = int(np.floor(rows))

        error = {}

        for i in range (rows):
        # for i in range (2):
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
                                count = 8)
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
            error[i]["v"] = error[i]["v"].reshape((-1,8))
            error[i]["v"] = error[i]["v"].T
            error[i]["t"] = [t - t0 for t in error[i]["t"]]

    name = directory +'/log.dat'
    log = None
    if os.path.exists(name):
        # name = directory +'/arUcos.dat'
        length = os.path.getsize(name)
        # print("length = ", length)

        with open(name, 'rb') as fileH:
            #   header
            # size = 1+9*4    # 4 dof
            # size = 1+9*6    # 6 dof
            size = 1+6    # 6 dof only singular values
            rows = (length) / (size* d)
            rows = int(np.floor(rows))
            log = np.fromfile(fileH,
                                    dtype = np.float64,
                                    count = size*rows)
            log = log.reshape((rows,size))
            log = log.T


        log[0,:] -= log[0,0]

    return state, arucos, error, log

def plot3D(directory, state, pd):
    
    source_dir = os.path.dirname(__file__)
    npzfile = np.load(source_dir +"/general.npz")
    colors = npzfile["colors"]
    nColors = colors.shape[0]
    
    
    fig, ax = plt.subplots(ncols = 1,
                           frameon=False,
                           figsize=(8,6))
    #fig = plt.figure(frameon=False, figsize=(5,3))
    ax.axis('off')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    name = directory+"/3Dplot"
    
    x_min = state[1,0]
    x_max = state[1,0]
    y_min = state[2,0]
    y_max = state[2,0]
    z_min = state[3,0]
    z_max = state[3,0]

    pos_array = np.zeros((6,state.shape[1]))
    pos_array[[0,1,2,5],:] = state[1:5,:]
    pos_array[4,:] = pi/2.

    x_min = min(x_min, pd[0])
    x_max = max(x_max, pd[0])
    y_min = min(y_min, pd[1])
    y_max = max(y_max, pd[1])
    z_min = min(z_min, pd[2])
    z_max = max(z_max, pd[2])

    plot_3Dcam(ax,
                pos_array,
                pos_array[:,0],
                pos_array[:,-1],
                pd,
                color = colors[0],
                camera_scale    = 0.05)

    
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
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_zlim(z_min,z_max)
    
    #ax = fig.add_subplot(1, 2, 2)
    
    #fig.legend( loc=1)
    plt.savefig(name+'.pdf')#,bbox_inches='tight')
    plt.close()
    

    
def main(arg):

    pd = np.array([0., 0., 1., 0, pi/2., pi/2])

    if len(arg) < 3:
        print("USE:\n$ python3 [DIRECTORY] [REFERENCE] [MARKERS]")
        return
    directory = arg[1].rstrip('/')+'/'
    reference = arg[2]
    if len(arg) > 3:
        markers = markers_list.index(arg[3])
    state, arucos, error, log = read_data(directory)

    s_ref = get_reference(reference, markers = markers, out_directory = directory )
    print(s_ref)
    print("Ploting VELOCITIES ")
    plot(directory, state)
    print("Ploting ArUcos")
    plotArucos(directory,  arucos, s_ref)
    print("Ploting Error")
    plotError(directory, error)
    print("Ploting 3D")
    plot3D(directory, state, pd )
    print("Ploting LOG")
    plotLog(directory, log )




if __name__ ==  "__main__":

    main(sys.argv)
