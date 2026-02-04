# -*- coding: utf-8 -*-
"""
    2024
    @author: E Chávez Aparicio (Axolomancer)
    @email: edgar.chavez@cimat.mx
    version: 3.0
    This code contains tha basic camera projection model
"""


import numpy as np
from numpy import sin, cos, pi
from numpy.linalg import matrix_rank, inv
import matplotlib.pyplot as plt

#   3D Arrow
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch


#   LATEX FIX
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#   Logger
import logging
logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.NOTSET)

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

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


class Camera:

    def __init__(self, config = None):

        if config is None:
            self.foco=0.002; #Focal de la camara
            self.rho = np.array([1.e-5,1.e-5])
            self.iMsize=[1024, 1024]; #Not working change this
            self.pPrinc=[self.iMsize[0]/2.0, self.iMsize[1]/2.0]; #Not working change this
            self.FOVxlim = self.rho[0]* self.iMsize[0]/(2*self.foco)
            self.FOVylim = self.rho[1]*self.iMsize[1]/(2*self.foco)

            #self.p = np.zeros((6,1))
            self.T = np.eye(4)
            self.f = self.foco/self.rho
            self.K = np.array( [[self.f[0],       0.0, self.pPrinc[0]],
                                [      0.0, self.f[1], self.pPrinc[1]],
                                [      0.0,       0.0,            1.0]])
        #TODO: Añadir una forma de configurar estos parámetros

    def pose(self, p):
        self.p = p.reshape(-1).copy()
        self._pose()

    def _pose(self):
        self.R = rotation_matrix_euler(self.p[3:])
        tmp = np.c_[ self.R, self.p[:3] ]
        #print(tmp)
        self.T = np.r_[ tmp, [[0.0,0.0,0.0,1.0]] ]
        self.Preal = np.c_[ self.R.T, -self.R.T @ self.p[:3] ]
        self.P = self.K @ self.Preal

    def project(self,p):
        n = p.shape[1]
        if p.shape[0] ==3:
            res = np.r_[p,np.ones((1,n))]
        else:
            res = p.copy()
        res = self.P @ res
        self.depth = res[2,:]
        res = res/self.depth
        return res[0:2,:]

    def normalize(self, in_points):
        #print("--- begin normalize")
        #print(in_points)
        points = in_points.copy()
        points[0,:] -= self.pPrinc[0]#cu
        points[1,:] -= self.pPrinc[1]#cv
        points[0,:] /= self.f[0]
        points[1,:] /= self.f[1]
        #print(points)
        #print("--- end normalize")

        return points



    def rectify(self, s_norm,  Z):
        #print("Z_in")
        #print(Z)
        n_points = s_norm.shape[1]

        points_r = s_norm * Z
        points_r = np.r_[points_r, Z.reshape((1,n_points))]

        _R = rotation_matrix(self.p[3],'x')
        _R = rotation_matrix(self.p[4],'y') @ _R
        _R = rotation_matrix(pi,'x').T @ _R
        points_r = _R @ points_r
        points_r = self.K @ points_r
        ret_Z = points_r[2,:].copy()
        points_r = points_r[0:2,:]/points_r[2,:]
        #print("zr")
        #print(ret_Z)
        return [points_r,ret_Z]

    #Revisa cuantos puntos ingresados están en FOV
    def count_points_in_FOV(self,P, enableMargin = True):

        PN = self.Preal @ P
        Z = PN[2,:]

        #   Only front ckeck
        if not enableMargin:
            return np.count_nonzero(Z > 0.)

        PN = PN[[0,1],:]/Z
        a = abs(PN[0,:]) < self.FOVxlim
        b = abs(PN[1,:]) < self.FOVylim

        test = []
        for i in range(PN.shape[1]):
            test.append(a[i] and b[i] and Z[i] > 0.0)
        return test.count(True)

    def draw_axis(self, ax,
                    scale=1.0,
                    lw= .5,
                    alpha=.5,
                    color = None):

        if color is None:
            _color = ['r','g','b']
        else:
            _color = [color]*3

        Oc = np.array([[0.,0,0,1]]).T
        Xc = np.array([[scale,0,0]]).T
        Yc = np.array([[0.,scale,0]]).T
        Zc = np.array([[0.,0,scale]]).T

        Oc1     = self.T @ Oc
        Xc1     = self.R @ Xc
        Yc1     = self.R @ Yc
        Zc1     = self.R @ Zc
        a1 = Arrow3D(Oc1[0,0],Oc1[1,0],Oc1[2,0],
                             Xc1[0,0],Xc1[1,0],Xc1[2,0],
                             mutation_scale=5,
                             lw=lw, arrowstyle="-|>",
                             linestyle = '--',
                             color=_color[0], alpha = alpha)
        a2 = Arrow3D(Oc1[0,0],Oc1[1,0],Oc1[2,0],
                             Yc1[0,0],Yc1[1,0],Yc1[2,0],
                             mutation_scale=5,
                             lw=lw, arrowstyle="-|>",
                             linestyle = '--',
                             color=_color[1], alpha = alpha)
        a3 = Arrow3D(Oc1[0,0],Oc1[1,0],Oc1[2,0],
                             Zc1[0,0],Zc1[1,0],Zc1[2,0],
                             mutation_scale=5,
                             lw=lw, arrowstyle="-|>",
                             linestyle = '--',
                             color=_color[2], alpha = alpha)
        ax.add_artist(a1)
        ax.add_artist(a2)
        ax.add_artist(a3)

    def draw_camera(self, ax,
                    color='cyan',
                    scale=1.0,
                    linestyle='solid',
                    lw= 1,
                    alpha=0.):
        #CAmera points: to be expressed in the camera frame;
        CAMup=np.array([[-1,-1,  1, 1, 1.5,-1.5,-1, 1 ],
                        [ 1, 1,  1, 1, 1.5, 1.5, 1, 1 ],
                        [ 2,-2, -2, 2,   3,   3, 2, 2 ],
                        [ 1, 1,  1, 1, 1  , 1 , 1, 1  ]])
        CAMup[0:3,:] = scale * CAMup[0:3,:]
        CAMupTRASF = self.T @ CAMup
        CAMdwn=np.array([[-1,-1,  1, 1, 1.5,-1.5,-1, 1  ],
                        [ -1,-1, -1,-1,-1.5,-1.5,-1,-1 ],
                        [  2,-2, -2, 2,   3,   3, 2, 2 ],
                        [ 1, 1,  1, 1, 1  , 1 , 1, 1  ]])
        CAMdwn[0:3,:] = scale * CAMdwn[0:3,:]
        CAMdwnTRASF     = self.T @ CAMdwn

        ax.plot(CAMupTRASF[0,:],
                CAMupTRASF[1,:],
                CAMupTRASF[2,:],
                c=color,ls=linestyle,lw=lw)
        ax.plot(CAMdwnTRASF[0,:],
                CAMdwnTRASF[1,:],
                CAMdwnTRASF[2,:],
                c=color,ls=linestyle, lw=lw)
        for i in range(6):
            ax.plot([CAMupTRASF[0,i],CAMdwnTRASF[0,i]],
                    [CAMupTRASF[1,i],CAMdwnTRASF[1,i]],
                    [CAMupTRASF[2,i],CAMdwnTRASF[2,i]],
                    c=color,ls=linestyle, lw=lw)

        self.draw_axis(ax,  scale = scale*10.0, color = 'k')

    def plot_3Dcam(self, ax, position_array,
               desired_configuration,
               color, label = "", lw = 1,
               lw_path = 0.5,
               camera_scale    = 0.02):

        #   Trayectories
        ax.plot(position_array[0,:],
        #ax.scatter(position_array[0,:],
                position_array[1,:],
                position_array[2,:],
                c = color.reshape((1,3)),
                linewidth = lw_path)
                #s = 0.1 ) # Plot camera trajectory

        #   Z axis refs
        ax.plot([position_array[0,0],position_array[0,0]],
                [position_array[1,0],position_array[1,0]],
                [0,position_array[2,0]],
                color = 'k', linestyle=(0, (5, 10)),lw = 0.5)
        ax.plot([position_array[0,-1],position_array[0,-1]],
                [position_array[1,-1],position_array[1,-1]],
                [0,position_array[2,-1]],
                color = 'k', linestyle=(0, (5, 10)),lw = 0.5)
        ax.plot([desired_configuration[0],desired_configuration[0]],
                [desired_configuration[1],desired_configuration[1]],
                [0,desired_configuration[2]],
                color = 'k', linestyle=(0, (5, 10)), lw = 0.5)

        #   Cameras
        print(position_array[:,0])
        self.pose(position_array[:,0])
        self.draw_camera(ax, scale=camera_scale, color='b', lw = lw)
        if label != "":
            ax.text(self.p[0],self.p[1],self.p[2],label)

        self.pose(desired_configuration)
        self.draw_camera(ax, scale=camera_scale, color='k',
                        linestyle=(0, (5, 10)), lw = 0.5*lw)
        if label != "":
            ax.text(self.p[0],self.p[1],self.p[2],label)

        self.pose(position_array[:,-1])
        self.draw_camera(ax, scale=camera_scale, color='r', lw = lw)
        if label != "":
            ax.text(self.p[0],self.p[1],self.p[2],label)



        ax.set_xlabel("$w_x$")
        ax.set_ylabel("$w_y$")
        ax.set_zlabel("$w_z$")
        ax.grid(True)
