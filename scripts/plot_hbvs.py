
#   libs
import numpy as np
from numpy import pi
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
                  out_directory = ''):
    image = cv2.imread(image_name)
    arucos = None

    arucoDict = cv2.aruco.getPredefinedDictionary(markers)
    detectorParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, detectorParams)
    (corners, ids, rejected) = detector.detectMarkers(image)

    print("Detected markers:", ids)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imwrite(os.path.join(out_directory,'ref_arucos.png'), image)

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
    camera = Camera()
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
        ax.plot(t_array,var_array[i,:] ,
                color=colors[(color_offset+i)%nColors], lw = 0.6 )
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




def plotPosition(directory, data):

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
    name = os.path.join(directory ,"State.pdf")
    plt.savefig(name,bbox_inches='tight')
    plt.close()

def plotVel(directory, data):

    time = data[0,:]
    velocities = data[1:,:]



    #  plot Velocities
    labels = ["X","Y","Z","Yaw"]
    fig_v, ax_v = plt.subplots(nrows = 1, figsize=(5,5))
    fig_v.suptitle("Velocities")
    symbols = plot_time(ax_v, time, velocities, color_offset = 1)
    ax_v.legend(symbols,labels, loc=1)
    ax_v.set_ylim([-.25,.25])
    # plt.show()
    name = os.path.join(directory ,"Velocities.pdf")
    plt.savefig(name,bbox_inches='tight')
    plt.close()


def plotNErr(directory, data):

    time = data[0,:]
    error = data[1,:]


        #   Plot error
    fig_e, ax_e = plt.subplots( figsize=(6,2))
    fig_e.suptitle("Error")
    plot_time(ax_e, time,error.reshape((1,-1)) )

    print("Minimun error= "+ str(error.min()))
    name = os.path.join(directory ,"Error.pdf")
    plt.savefig(name,bbox_inches='tight')
    plt.close()

def plotError(directory,error):

    fig, ax = plt.subplots( figsize=(5,5))
    fig.suptitle("Error")
    plot_time(ax, error["t"].reshape(-1), error["v"], color_offset = 1)
    ax.set_ylim([-.25,.25])
    name = os.path.join(directory ,"Error_runtime.pdf")
    plt.savefig(name ,bbox_inches='tight')
    plt.close()


def plotArucos(directory, arucos, reference):

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
    name = os.path.join(directory ,"Error_arucos.pdf")
    plt.savefig(name,bbox_inches='tight')
    plt.close()

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
    name = os.path.join(directory ,"ArUcos.pdf")
    plt.savefig(name,bbox_inches='tight')
    #plt.show()
    plt.close()


def plot3D(directory, state, pd):

    source_dir = os.path.dirname(__file__)
    npzfile = np.load(source_dir +"/general.npz")
    colors = npzfile["colors"]
    nColors = colors.shape[0]


    fig, ax = plt.subplots(ncols = 1,
                           frameon=False,
                           figsize=(8,6))
    ax.axis('off')
    ax = fig.add_subplot(1, 1, 1, projection='3d')


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

    name = os.path.join(directory ,"3Dplot.pdf")
    plt.savefig(name)#,bbox_inches='tight')
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
    labels = [str(i) for i in range(6)]
    fig_p, ax_p = plt.subplots(nrows = 1, figsize=(5,5))
    fig_p.suptitle("Singular values L.T L ")
    symbols = plot_time(ax_p, time, svd , color_offset = 1)
    ax_p.legend(symbols,labels, loc=1)
    # ax_p.set_ylim((-0.06,0.06))
    # plt.show()
    name = os.path.join(directory ,"LOG_SVD_D.pdf")
    plt.savefig(name ,bbox_inches='tight')
    plt.close()




 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #          READ DATA




def read_data(directory):

    f = 4
    d = 8
    _i = 4
    
    name = os.path.join(directory ,"position.dat")
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
                position[0,:] -= position[0,0]

    name = os.path.join(directory ,"velocities.dat")
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

    name = os.path.join(directory ,"norm_error.dat")
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

    name = os.path.join(directory ,"arUcos.dat")
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

    name = os.path.join(directory ,"error.dat")
    error = None
    if os.path.exists(name):
        length = os.path.getsize(name)
        if length > 0:
            error = {}
            with open(name, 'rb') as fileH:
                size = 7*d
                rows = (length) / size
                rows = int(np.floor(rows))
                _error = np.fromfile(fileH,
                                        dtype = np.float64,
                                        count = 7*rows)
                _error = _error.reshape((rows,7))
                error["v"] = _error[:,1:].T
                error["t"] = _error[:,0] - _error[0,0]

                # error = {}
                #
                # for i in range (rows):
                #     time = np.fromfile(fileH,
                #                         dtype = np.float64,
                #                         count = 1)
                #     time = time[0]
                #     idx = np.fromfile(fileH,
                #                         dtype = np.int64,
                #                         count = 1)
                #     idx = idx[0]
                #     _error = np.fromfile(fileH,
                #                         dtype = np.float64,
                #                         count = 8)
                #     if (any(_error > 10)):
                #         print(_error)
                #     if idx in error:
                #         error[idx]["t"].append(time)
                #         error[idx]["v"] = np.concatenate([error[idx]["v"],_error])
                #     else:
                #         _d = {"t":[time], "v":_error}
                #         error[idx] = _d
                #
                # t0 = [ error[key]["t"][0] for  key in error]
                # t0 = min(t0)
                # for i in error:
                #     error[i]["v"] = error[i]["v"].reshape((-1,8))
                #     error[i]["v"] = error[i]["v"].T
                #     error[i]["t"] = [t - t0 for t in error[i]["t"]]

    name = os.path.join(directory ,"log.dat")
    log = None
    if os.path.exists(name):
        length = os.path.getsize(name)
        if length > 0:
            with open(name, 'rb') as fileH:
                #   header
                size = 1+6    # 6 dof only singular values
                rows = (length) / (size* d)
                rows = int(np.floor(rows))
                log = np.fromfile(fileH,
                                        dtype = np.float64,
                                        count = size*rows)
                log = log.reshape((rows,size))
                log = log.T
            log[0,:] -= log[0,0]

    return position, velocities, n_e, arucos, error, log

 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #          READ MAIN



def main(arg):

    pd = np.array([arg.pose[0], arg.pose[1], arg.pose[2],
                   0, pi/2., pi*arg.pose[3]/180.])
    markers = markers_list.index(arg.markers)
    directory = arg.directory

    #   TODO: revisar tamaño del archivo
    position, velocities, n_e, arucos, error, log = read_data(directory)

    s_ref = get_reference(arg.reference,
                          markers = markers,
                          out_directory = directory )
    print(s_ref)
    if not n_e is None:
        print("Ploting  ")
        plotNErr(directory, n_e)
    if not velocities is None:
        print("Ploting VELOCITIES ")
        plotVel(directory, velocities)
    if not arucos is None:
        print("Ploting ArUcos")
        plotArucos(directory,  arucos, s_ref)
    if not error is None:
        print("Ploting Error")
        plotError(directory, error)
    if not position is None:
        print("Ploting 3D")
        plot3D(directory, position, pd )
        plotPosition(directory, position)

    if not log is None:
        print("Ploting LOG")
        plotLog(directory, log )


if __name__ ==  "__main__":
    description = "Plotting single camera experiment data"
    parser = argparse.ArgumentParser(prog = 'python3 plot_hbvs.py',
                                     description = description)
    parser.add_argument( 'directory', type=str, default = 'output',
        help = "Directory name (default output/)")
    parser.add_argument( '--reference', type=str,
        default = 'config/reference_2_flat.png',
        help = "IBVS reference image (default config/reference_2_flat.png)")
    parser.add_argument( '--pose', type=float, default = [1., 0., 1., 90] ,
        nargs = 4,
        help = "Reference pose [x, y, z, yaw (degs)] default [1., 0., 1., 90]")
    parser.add_argument( '--markers', type=str, default = '6X6_1000',
        choices = markers_list,
        help = "Fiducial markers family (default 6X6_1000)")

    arg = parser.parse_args()
    main(arg)
