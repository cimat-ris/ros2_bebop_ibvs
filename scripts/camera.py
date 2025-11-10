# -*- coding: utf-8 -*-
"""
    2024
    @author: E Chávez Aparicio (Axologger)
    @email: edgar.chavez@cimat.mx
    version: 2.0
    This code contains tha basic camera projection model
"""


import numpy as np
from numpy import sin, cos, pi
from numpy.linalg import matrix_rank, inv
import matplotlib.pyplot as plt

#   LATEX FIX
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


import Arrow3D


def rot(ang,ax):
    
    ca = cos(ang)
    sa = sin(ang)
    
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
    
class camera:
    
    def __init__(self):
        self.foco=0.002; #Focal de la camara
        self.rho = np.array([1.e-5,1.e-5])
        self.iMsize=[1024, 1024]; #Not working change this
        self.pPrinc=[self.iMsize[0]/2.0, self.iMsize[1]/2.0]; #Not working change this
        self.FOVxlim = self.rho[0]* self.iMsize[0]/(2*self.foco)
        self.FOVylim = self.rho[1]*self.iMsize[1]/(2*self.foco)

        self.p = np.zeros((6,1))
        self.T = np.eye(4)
        self.K = np.array( [[self.foco/self.rho[0],       0.0, self.pPrinc[0]],
                            [      0.0, self.foco/self.rho[1], self.pPrinc[1]],
                            [      0.0,       0.0,            1.0]])
    
    def pose(self,p):
       #print("--- begin pose")
       #print(p)
       self.p = p.copy()
       self.R = rot(self.p[5],'z') 
       self.R = self.R @ rot(self.p[4],'y')
       self.R = self.R @ rot(self.p[3],'x')
       tmp = np.c_[ self.R, self.p[:3] ]
       #print(tmp)
       self.T = np.r_[ tmp, [[0.0,0.0,0.0,1.0]] ]
       self.Preal = np.c_[ self.R.T, -self.R.T @ self.p[:3] ]
       self.P = self.K @ self.Preal
       #print(self.P)
       
       #print("--- begin pose")
       
    def project(self,p):
        #print("--- begin project")
        #print(p)
        n = p.shape[1]
        res = np.zeros((4,n))
        if p.shape[0] ==3:
            res = np.r_[p,np.ones((1,n))]
        else:
            res = p.copy()
        #print(res)
        res = self.P @ res
        #print(res)
        res = res/res[2,:]
        res = res[0:2,:]
        #print(res)
        #print("--- begin project")
        return res
    
    def normalize(self, in_points):
        #print("--- begin normalize")
        f = self.foco/self.rho
        #print(in_points)
        points = in_points.copy()
        points[0,:] -= self.pPrinc[0]#cu
        points[1,:] -= self.pPrinc[1]#cv
        points[0,:] /= f[0]
        points[1,:] /= f[1]
        #print(points)
        #print("--- end normalize")
        
        return points
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
        a1 = Arrow3D.Arrow3D(Oc1[0,0],Oc1[1,0],Oc1[2,0],
                             Xc1[0,0],Xc1[1,0],Xc1[2,0],
                             mutation_scale=5,
                             lw=lw, arrowstyle="-|>",
                             linestyle = '--',
                             color=_color[0], alpha = alpha)
        a2 = Arrow3D.Arrow3D(Oc1[0,0],Oc1[1,0],Oc1[2,0],
                             Yc1[0,0],Yc1[1,0],Yc1[2,0],
                             mutation_scale=5,
                             lw=lw, arrowstyle="-|>",
                             linestyle = '--',
                             color=_color[1], alpha = alpha)
        a3 = Arrow3D.Arrow3D(Oc1[0,0],Oc1[1,0],Oc1[2,0],
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
            
        
        self.draw_axis(ax,  scale = scale*5.0, color = 'k')
    
    
        
