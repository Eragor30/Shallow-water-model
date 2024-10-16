# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:59:30 2024

@author: viteb
"""
import os
from IPython.display import clear_output
import sys
import time
import scipy.io as sio
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


#%%Schallow_V2
class shallow_water(object):
    def __init__(self,time_window,g,f,H,h_I,V0,dx,dy,n,p):
        self.time_window=time_window
        self.h_I=h_I
        self.V0=V0
        self.g=g
        self.hb=H
        self.dx=dx
        self.dy=dy
        self.n=n
        self.p=p
        self.indx=np.arange(self.n)
        self.indy=np.arange(self.p)
        self.f=f
        self.t=None
        self.dt=self.time_window[1]-self.time_window[0]
        self.f1=[]
        self.f3=[]
        self.alpha=0.5
        self.boundary=self.Atmo
        self.perturbation=None
    
"""P1 for x-axis momentum equation"""
    
    #C = Courant number
    def Cu(self,V):
        u=V[0]
        v=V[1]
        Cwp = 0.25*((u[:,(self.indx-1)%self.n]+np.abs(u[:,(self.indx-1)%self.n]))+(u+np.abs(u)))*self.dt/self.dx 
        Cwm = 0.25*((u[:,(self.indx-1)%self.n]-np.abs(u[:,(self.indx-1)%self.n]))+(u-np.abs(u)))*self.dt/self.dx
        Cep = 0.25*((u+np.abs(u)+(u[:,(self.indx+1)%self.n]+np.abs(u[:,(self.indx+1)%self.n]))))*self.dt/self.dx
        Cem = 0.25*((u-np.abs(u))+(u[:,(self.indx+1)%self.n]-np.abs(u[:,(self.indx+1)%self.n])))*self.dt/self.dx
        Csp = 0.25*((v[(self.indy-1)%self.p,:]+np.abs(v[(self.indy-1)%self.p,:]))+((v[:,(self.indx+1)%self.n][(self.indy-1)%self.p,:]+np.abs(v[:,(self.indx+1)%self.n][(self.indy-1)%self.p,:]))))*self.dt/self.dy            
        Csm = 0.25*((v[(self.indy-1)%self.p,:]-np.abs(v[(self.indy-1)%self.p,:]))+((v[:,(self.indx+1)%self.n][(self.indy-1)%self.p,:]-np.abs(v[:,(self.indx+1)%self.n][(self.indy-1)%self.p,:]))))*self.dt/self.dy 
        Cnp = 0.25*((v+np.abs(v))+(v[:,(self.indx+1)%self.n]+np.abs(v[:,(self.indx+1)%self.n])))*self.dt/self.dy             
        Cnm = 0.25*((v-np.abs(v))+(v[:,(self.indx+1)%self.n]-np.abs(v[:,(self.indx+1)%self.n])))*self.dt/self.dy
        return np.array([Cwp,Cwm,Cep,Cem,Csp,Csm,Cnp,Cnm])            
    
    # r = up-wind down-wind gradient
    def ru(self,V):
        u=V[0]
        v=V[1]
        
        if 0 in (u-u[:,(self.indx-1)%self.n]):
            rwp=0
            rwm=0
        else:
            rwp = (u[:,(self.indx-1)%self.n]-u[:,(self.indx-2)%self.n])/(u-u[:,(self.indx-1)%self.n])
            rwm = (u[:,(self.indx+1)%self.n]-u)/(u-u[:,(self.indx-1)%self.n])
            
        if 0 in (u[:,(self.indx+1)%self.n]-u):
            rep=0
            rem=0
        else:
            rep = (u-u[:,(self.indx-1)%self.n])/(u[:,(self.indx+1)%self.n]-u)
            rem = (u[:,(self.indx+2)%self.n]-u[:,(self.indx+1)%self.n])/(u[:,(self.indx+1)%self.n]-u)
        
        if 0 in (u-u[(self.indy-1)%self.p,:]):
            rsp=0
            rsm=0
        else:
            rsp = (u[(self.indy-1)%self.p,:]-u[(self.indy-2)%self.p,:])/(u-u[(self.indy-1)%self.p,:])
            rsm = (u[(self.indy+1)%self.p,:]-u)/(u-u[(self.indy-1)%self.p,:])
        
        if 0 in (u[(self.indy+1)%self.p,:]-u):
            rnp=0
            rnm=0
        else:
            rnp = (u-u[(self.indy-1)%self.p,:])/(u[(self.indy+1)%self.p,:]-u)
            rnm = (u[(self.indy+2)%self.p,:]-u[(self.indy+1)%self.p,:])/(u[(self.indy+1)%self.p,:]-u)
            
        return [rwp,rwm,rep,rem,rsp,rsm,rnp,rnm]
    
    
    # evaluation of u at each face of the volume domain
    def u(self,V):
        u=V[0]
        C=self.Cu(V)
        r=self.ru(V)
        Cwp,Cwm,Cep,Cem,Csp,Csm,Cnp,Cnm = C[0],C[1],C[2],C[3],C[4],C[5],C[6],C[7]
        rwp,rwm,rep,rem,rsp,rsm,rnp,rnm = r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]
        
        uwp = u[:,(self.indx-1)%self.n]+0.5*self.psi(rwp)*(1-Cwp)*(u-u[:,(self.indx-1)%self.n])
        uwm = u-0.5*self.psi(rwm)*(1+Cwm)*(u-u[:,(self.indx-1)%self.n])
        uep = u+0.5*self.psi(rep)*(1-Cep)*(u[:,(self.indx+1)%self.n]-u)
        uem = u[:,(self.indx+1)%self.n]-0.5*self.psi(rem)*(1+Cem)*(u[:,(self.indx+1)%self.n]-u)
        usp = u[(self.indy-1)%self.p,:]+0.5*self.psi(rsp)*(1-Csp)*(u-u[(self.indy-1)%self.p,:])
        usm = u-0.5*self.psi(rsm)*(1+Csm)*(u-u[(self.indy-1)%self.p,:])
        unp = u+0.5*self.psi(rnp)*(1-Cnp)*(u[(self.indy+1)%self.p,:]-u)
        unm = u[(self.indy+1)%self.p,:]-0.5*self.psi(rnm)*(1+Cnm)*(u[(self.indy+1)%self.p,:]-u)
        
        return np.array([uwp,uwm,uep,uem,usp,usm,unp,unm])
    
    #P1u = - Δt* ∇(u*vect(u))    
    def P1u(self,V):
        C=self.Cu(V)
        u=self.u(V)
        return C[0]*u[0]+C[1]*u[1]-C[2]*u[2]-C[3]*u[3]+C[4]*u[4]+C[5]*u[5]-C[6]*u[6]-C[7]*u[7]
   

"""P1 for x-axis momentum equation"""

    #Courant number 
    def Cv(self,V):
        u=V[0]
        v=V[1]
        Cwp = 0.25*(((u[:,(self.indx-1)%self.n])[(self.indy+1)%self.p,:]+np.abs((u[:,(self.indx-1)%self.n])[(self.indy+1)%self.p,:]))+(u[:,(self.indx-1)%self.n]+np.abs(u[:,(self.indx-1)%self.n])))*self.dt/self.dx 
        Cwm = 0.25*(((u[:,(self.indx-1)%self.n])[(self.indy+1)%self.p,:]-np.abs((u[:,(self.indx-1)%self.n])[(self.indy+1)%self.p,:]))+(u[:,(self.indx-1)%self.n]-np.abs(u[:,(self.indx-1)%self.n])))*self.dt/self.dx
        Cep = 0.25*((u[(self.indy+1)%self.p,:]+np.abs(u[(self.indy+1)%self.p,:]))+(u+np.abs(u)))*self.dt/self.dx
        Cem = 0.25*((u[(self.indy+1)%self.p,:]-np.abs(u[(self.indy+1)%self.p,:]))+(u-np.abs(u)))*self.dt/self.dx
        Csp = 0.25*((v[(self.indy-1)%self.p,:]+np.abs(v[(self.indy-1)%self.p,:]))+(v+np.abs(v)))*self.dt/self.dy            
        Csm = 0.25*((v[(self.indy-1)%self.p,:]-np.abs(v[(self.indy-1)%self.p,:]))+(v-np.abs(v)))*self.dt/self.dy   
        Cnp = 0.25*((v+np.abs(v))+(v[(self.indy+1)%self.p,:]+np.abs(v[(self.indy+1)%self.p,:])))*self.dt/self.dy             
        Cnm = 0.25*((v-np.abs(v))+(v[(self.indy+1)%self.p,:]-np.abs(v[(self.indy+1)%self.p,:])))*self.dt/self.dy 
        return np.array([Cwp,Cwm,Cep,Cem,Csp,Csm,Cnp,Cnm])       
    
    # r = up-wind down-wind gradient
    def rv(self,V):
        u=V[0]
        v=V[1]
        
        if 0 in (v-v[:,(self.indx-1)%self.n]):
            rwp=1
            rwm=1
        else:
            rwp = (v[:,(self.indx-1)%self.n]-v[:,(self.indx-2)%self.n])/(v-v[:,(self.indx-1)%self.n])
            rwm = (v[:,(self.indx+1)%self.n]-v)/(v-v[:,(self.indx-1)%self.n])
        
        if 0 in (v[:,(self.indx+1)%self.n]-v):
            rep=1
            rem=1
        else:
            rep = (v-v[:,(self.indx-1)%self.n])/(v[:,(self.indx+1)%self.n]-v)
            rem = (v[:,(self.indx+2)%self.n]-v[:,(self.indx+1)%self.n])/(v[:,(self.indx+1)%self.n]-v)
            
        if 0 in (v-v[(self.indy-1)%self.p,:]):
            rsp=1
            rsm=1
        else:
            rsp = (v[(self.indy-1)%self.p,:]-v[(self.indy-2)%self.p,:])/(v-v[(self.indy-1)%self.p,:])
            rsm = (v[(self.indy+1)%self.p,:]-v)/(v-v[(self.indy-1)%self.p,:])
            
        if 0 in ((v[(self.indy+1)%self.p,:]-v)):
            rnp=1
            rnm=1
        else:
            rnp = (v-v[(self.indy-1)%self.p,:])/(v[(self.indy+1)%self.p,:]-v)
            rnm = (v[(self.indy+2)%self.p,:]-v[(self.indy+1)%self.p,:])/(v[(self.indy+1)%self.p,:]-v)
        
        
        return [rwp,rwm,rep,rem,rsp,rsm,rnp,rnm]
     
    # evaluation of v at each face of the volume domain
    def v(self,V):
        v=V[1]
        C=self.Cv(V)
        r=self.rv(V)
        Cwp,Cwm,Cep,Cem,Csp,Csm,Cnp,Cnm = C[0],C[1],C[2],C[3],C[4],C[5],C[6],C[7]
        rwp,rwm,rep,rem,rsp,rsm,rnp,rnm = r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]
        
        vwp = v[:,(self.indx-1)%self.n]+0.5*self.psi(rwp)*(1-Cwp)*(v-v[:,(self.indx-1)%self.n])
        vwm = v-0.5*self.psi(rwm)*(1+Cwm)*(v-v[:,(self.indx-1)%self.n])
        vep = v+0.5*self.psi(rep)*(1-Cep)*(v[:,(self.indx+1)%self.n]-v)
        vem = v[:,(self.indx+1)%self.n]-0.5*self.psi(rem)*(1+Cem)*(v[:,(self.indx+1)%self.n]-v)
        vsp = v[(self.indy-1)%self.p,:]+0.5*self.psi(rsp)*(1-Csp)*(v-v[(self.indy-1)%self.p,:])
        vsm = v-0.5*self.psi(rsm)*(1+Csm)*(v-v[(self.indy-1)%self.p,:])
        vnp = v+0.5*self.psi(rnp)*(1-Cnp)*(v[(self.indy-1)%self.p,:]-v)
        vnm = v[(self.indy-1)%self.p,:]-0.5*self.psi(rnm)*(1+Cnm)*(v[(self.indy+1)%self.p,:]-v)
        
        return np.array([vwp,vwm,vep,vem,vsp,vsm,vnp,vnm])
    
    #P1v = ∇(v*vect(u))
    def P1v(self,V):
        C=self.Cv(V)
        v=self.v(V)
        return C[0]*v[0]+C[1]*v[1]-C[2]*v[2]-C[3]*v[3]+C[4]*v[4]+C[5]*v[5]-C[6]*v[6]-C[7]*v[7]
    
    
"""Ph for the continuity equation""" 

    #Courant number 
    def Ch(self,V):
        u=V[0]
        v=V[1]
        Cwp = 0.5*(u[:,(self.indx-1)%self.n]+np.abs(u[:,(self.indx-1)%self.n]))*self.dt/self.dx 
        Cwm = 0.5*(u[:,(self.indx-1)%self.n]-np.abs(u[:,(self.indx-1)%self.n]))*self.dt/self.dx
        Cep = 0.5*(u[:,(self.indx+1)%self.n]+np.abs(u[:,(self.indx+1)%self.n]))*self.dt/self.dx
        Cem = 0.5*(u[:,(self.indx+1)%self.n]-np.abs(u[:,(self.indx+1)%self.n]))*self.dt/self.dx
        Csp = 0.5*(v[(self.indy-1)%self.p,:]+np.abs(v[(self.indy-1)%self.p,:]))*self.dt/self.dy            
        Csm = 0.5*(v[(self.indy-1)%self.p,:]-np.abs(v[(self.indy-1)%self.p,:]))*self.dt/self.dy 
        Cnp = 0.5*(v[(self.indy+1)%self.p,:]+np.abs(v[(self.indy+1)%self.p,:]))*self.dt/self.dy             
        Cnm = 0.5*(v[(self.indy+1)%self.p,:]-np.abs(v[(self.indy+1)%self.p,:]))*self.dt/self.dy
        return np.array([Cwp,Cwm,Cep,Cem,Csp,Csm,Cnp,Cnm])
    
    
    # r = up-wind down-wind gradient for h
    def rh(self,h):
        if  0 in (h-h[:,(self.indx-1)%self.n])  :
            rwp=1
            rwm=1
        else:
            rwp = (h[:,(self.indx-1)%self.n]-h[:,(self.indx-2)%self.n])/(h-h[:,(self.indx-1)%self.n])
            rwm = (h[:,(self.indx+1)%self.n]-h)/(h-h[:,(self.indx-1)%self.n])
        
        if  0 in (h[:,(self.indx+1)%self.n]-h) : 
            rep=1
            rem=1
        else:
            rep = (h-h[:,(self.indx-1)%self.n])/(h[:,(self.indx+1)%self.n]-h)
            rem = (h[:,(self.indx+2)%self.n]-h[:,(self.indx+1)%self.n])/(h[:,(self.indx+1)%self.n]-h)
        
        if  0 in (h-h[(self.indy-1)%self.p,:]) : 
            rsp=1
            rsm=1
            
        else:
            rsp = (h[(self.indy-1)%self.p,:]-h[(self.indy-2)%self.p,:])/(h-h[(self.indy-1)%self.p,:])
            rsm = (h[(self.indy+1)%self.p,:]-h)/(h-h[(self.indy-1)%self.p,:])
        
        if  0 in (h[(self.indy+1)%self.p,:]-h) : 
            rnp=1
            rnm=1
            
        else:
            rnp = (h-h[(self.indy-1)%self.p,:])/(h[(self.indy+1)%self.p,:]-h)
            rnm = (h[(self.indy+2)%self.p,:]-h[(self.indy+1)%self.p,:])/(h[(self.indy+1)%self.p,:]-h)
       
        return [rwp,rwm,rep,rem,rsp,rsm,rnp,rnm]
    
    # evaluation of h at each face of the volume domain
    def h(self,h,V):
        C=self.Cu(V)
        r=self.rh(h)
        Cwp,Cwm,Cep,Cem,Csp,Csm,Cnp,Cnm = C[0],C[1],C[2],C[3],C[4],C[5],C[6],C[7]
        rwp,rwm,rep,rem,rsp,rsm,rnp,rnm = r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]
        
        hwp = h[:,(self.indx-1)%self.n]+0.5*self.psi(rwp)*(1-Cwp)*(h-h[:,(self.indx-1)%self.n])
        hwm = h-0.5*self.psi(rwm)*(1+Cwm)*(h-h[:,(self.indx-1)%self.n])
        hep = h+0.5*self.psi(rep)*(1-Cep)*(h[:,(self.indx+1)%self.n]-h)
        hem = h[:,(self.indx+1)%self.n]-0.5*self.psi(rem)*(1+Cem)*(h[:,(self.indx+1)%self.n]-h)
        hsp = h[(self.indy-1)%self.p,:]+0.5*self.psi(rsp)*(1-Csp)*(h-h[(self.indy-1)%self.p,:])
        hsm = h-0.5*self.psi(rsm)*(1+Csm)*(h-h[(self.indy-1)%self.p,:])
        hnp = h+0.5*self.psi(rnp)*(1-Cnp)*(h[(self.indy+1)%self.p,:]-h)
        hnm = h[(self.indy+1)%self.p,:]-0.5*self.psi(rnm)*(1+Cnm)*(h[(self.indy+1)%self.p,:]-h)
        
        return np.array([hwp,hwm,hep,hem,hsp,hsm,hnp,hnm])
   
    
   
    # Ph = ∇(hu)
    def Ph(self,V,h):
        C=self.Ch(V)
        h=self.h(h,V)
        return (C[0]*h[0]+C[1]*h[1]-C[2]*h[2]-C[3]*h[3]+C[4]*h[4]+C[5]*h[5]-C[6]*h[6]-C[7]*h[7])/self.dt
        #div=((h*V)[0,(self.indx+1)%self.n,:]-(h*V)[0,(self.indx-1)%self.n,:])/(2*self.dx)+((h*V)[1,:,(self.indy+1)%self.p].T-(h*V)[1,:,(self.indy-1)%self.p].T)/(2*self.dy)
       
        
        
        
        #return -div
        
        
        
"""P2u = u*div(V)"""   
    def P2u(self,V):
        u,v=V[0],V[1]
        terme1 = (u[:,(self.indx+1)%self.n]-u[:,(self.indx-1)%self.n])/(2*self.dx)
        terme2 = (v+v[:,(self.indx+1)%self.n]-v[(self.indy-1)%self.p,:]-(v[:,(self.indx+1)%self.n])[(self.indy-1)%self.p,:])/(2*self.dy)
        
        return u*(terme1+terme2)
 
""""P2u = v*div(V)"""   
    def P2v(self,V):
        u,v=V[0],V[1]
        terme1 = (u+u[(self.indy+1)%self.p,:]-(u[:,(self.indx-1)%self.n])[(self.indy+1)%self.p,:]-u[:,(self.indx-1)%self.n])/(2*self.dx)
        terme2 = (v[(self.indy+1)%self.p,:]-v[(self.indy-1)%self.p,:])/(2*self.dy)
        
        return v*(terme1+terme2)
    
 """dynamic pressure terms for u"""  
    def Ppu(self,h):
        h=h+self.hb
        return -self.g*(h[:,(self.indx+1)%self.n]-h[:,(self.indx-1)%self.n])/(2*self.dx)
 
"""dynamic pressure terms for v"""
    def Ppv(self,h):
        h=h+self.hb
        Ppv=-self.g*((h[(self.indy+1)%self.p,:])-(h[(self.indy-1)%self.p,:]))/(2*self.dy)
        return Ppv
    
"""Flux limiter function"""    
    def psi(self,r):
        #return max(0,max(np.max(2*r),1),min(np.min(r),2))
        return (r+np.abs(r))/(1+np.abs(r))


"""u equation"""          
    def du(self,V,h):
        
        P1u=self.P1u(V)
        P2u= self.P2u(V)
        Ppu=self.Ppu(h)
        return  P1u/self.dt + P2u + Ppu
"""v equation"""      
    def dv(self,V,h):
        P1v=self.P1v(V)
        P2v= self.P2v(V)
        Ppv=self.Ppv(h)
        return (P1v/self.dt + P2v + Ppv)
    
"""vector V equation"""    
    def dV(self,V,h):
        
        return np.array([self.du(V,h),self.dv(V,h)])
  
    
  
    def euler(self,t,h,V,para,trend):
        """ Euler time-scheme """
        
        if para=='h':
            x0=h
            k1 = trend(V,h)
            k2 = self.f3[0]
            k3=self.f3[1]
            self.f3=[k1,k2,k3]
        elif para=='V' :
            x0=V
            k1 = trend(V,h)
            k2 = self.f1[0]
            k3=self.f1[1]
            self.f1=[k1,k2,k3]
        return x0 + self.dt*(23/12*k1-4/3*k2+5/12*k3)
 
        
"""interpolation of u and v for coriolis integration""" 
    def Vavg(self,V):
        u=V[0]
        v=V[1]
        
        uavg=0.25*((u[:,(self.indx+1)%self.n])[(self.indy-1)%self.p,:]+u[:,(self.indx+1)%self.n]+u+u[(self.indy-1)%self.p,:])
        vavg=0.25*((v[:,(self.indx-1)%self.n])[(self.indy+1)%self.p,:]+v[:,(self.indx-1)%self.n]+v+v[(self.indy+1)%self.p,:])
        return np.array([vavg,-uavg])
 
  
"""boundary conditions"""
  
 # Double Periodic Conditions
    def DPC(self,V,h,para='v'): 
        u=V[0]
        v=V[1]
        
        #Buttom
        u[:,0]=u[:,-4]
        u[:,1]=u[:,-3]
        v[:,0]=v[:,-4]
        v[:,1]=v[:,-3]
        
        #Top
        u[:,-2]=u[:,2]
        u[:,-1]=u[:,3]
        v[:,-2]=v[:,2]
        v[:,-1]=v[:,3]
        
        #Left
        u[1,:]=u[-3,:]
        u[0,:]=u[-4,:]
        v[1,:]=v[-3,:]
        v[0,:]=v[-4,:]
        
        #Right
        u[-2,:]=u[2,:]
        u[-1,:]=u[3,:]
        v[-2,:]=v[2,:]
        v[-1,:]=v[3,:]
   
    
   
    # Full Slip Conditions    
    def FSC(self,V,h,para='v'): 
        u=V[0]
        v=V[1]
            
        #Buttom
        u[:,0]=u[:,3]
        u[:,1]=u[:,2]
        v[:,0]=0
        v[:,1]=0
            
        #Top
        u[:,-2]=u[:,-3]
        u[:,-1]=u[:,-4]
        v[:,-2]=0
        v[:,-1]=0
        # v[-3,:]=0
        #Left
        u[1,:]=0
        u[0,:]=0
        v[1,:]=v[2,:]
        v[0,:]=v[3,:]
        
        #Right
        u[-2,:]=0
        u[-1,:]=0
        v[-2,:]=v[-3,:]
        v[-1,:]=v[-4,:]
        # u[:,-3]
        if para=='h':
            h[:,0]=5
            h[:,1]=5
            h[-2,:]=5
            h[-1,:]=5
            h[:,-1]=5
            h[:,-2]=5
            h[0,:]=5
            h[1,:]=5
            
            
            
            
    def Atmo(self,V,V0,h,h0,para='v'):
        u=V[0]
        v=V[1]
        u0=V0[0]
        v0=V0[1]
        
        if para == 'v' :   
            #Left
            u[0,:] = (u[2,:] + u[3,:])/2
            u[1,:] = (u[2,:] + u[3,:])/2
            # u[1,:]=self.V0[0,1,:]
            # u[0,:]=self.V0[0,0,:]
            
            v[1,:]=v[2,:]/2
            v[0,:]=0
            #Right
            u[-1,:] = (u[-3,:] + u[-4,:])/2
            u[-2,:] = (u[-3,:] + u[-4,:])/2
            # u[-2,:]=self.V0[0,-2,:]
            # u[-1,:]=self.V0[0,-1,:]
            v[-1,:]=0
            v[-2,:]=v[-3,:]/2   
        
        
        if para=='h':
            # h[0,:]=h0[0,:]-self.dt*self.hb[0,:]*((u0[0,(self.indx+1)%self.n]-u0[0,(self.indx-1)%self.n]/(2*self.dx))+(v0[0,:]-v0[1,:])/self.dx)
            # h[1,:]=h0[1,:]-self.dt*self.hb[1,:]*((u0[1,(self.indx+1)%self.n]-u0[1,(self.indx-1)%self.n]/(2*self.dx))+(v0[1,:]-v0[2,:])/self.dy)
            
            # h[-1,:]=h0[-1,:]-self.dt*self.hb[-1,:]*((u0[-1,(self.indx+1)%self.n]-u0[-1,(self.indx-1)%self.n]/(2*self.dx))+(v0[-1,:]-v0[-2,:])/self.dx)
            # h[-2,:]=h0[-2,:]-self.dt*self.hb[-2,:]*((u0[-2,(self.indx+1)%self.n]-u0[-2,(self.indx-1)%self.n]/(2*self.dx))+(v0[-2,:]-v0[-3,:])/self.dy)


            


            h[1,:]=(self.h_I[1,:])
            h[0,:]=(self.h_I[0,:])
            
            
            h[-2,:]=(self.h_I[-2,:])
            h[-1,:]=(self.h_I[-1,:])

            
    def Vortex(self,i):
        amplitude=0.1
        sigma=1400000
        sigma_y=1400000
        mu_x=100*self.dx
        mu_y=100*self.dy
        #if i<0.25*len(self.time_window):
        return amplitude * np.exp(-((X - mu_x)**2 / (2 * sigma_x**2) + (Y - mu_y)**2 / (2 * sigma_y**2)))+5
        #else:
         #   return 0
         
         
"""potential vorticity"""         
    def pvu(self,V,h):
        ksi=np.zeros((self.n,self.p))
        u=V[0]
        v=V[1]
        h=h+self.hb
        duy=(u[(self.indy+1)%self.p,:]-u[(self.indy-1)%self.p,:])/(2*self.dy)
        dvx=(v[:,(self.indx+1)%self.n]-v[:,(self.indx-1)%self.n])/(2*self.dx)
        ksi=dvx-duy
        return (ksi-self.f)/h         
    
       
    def forecast(self,V0,h0):
       
        V0=np.array([V0[0],V0[1]])
        h0=h0
        V=[V0]
        h=[h0]
        pvu=[self.pvu(V0,h0)]
        for i in range (2):
            dV0=self.dV(V0,h0)
            self.f1.append(dV0)
            
             
            V1 = V0 + self.dt*dV0 
            V1=  (V1+(self.alpha * self.dt * self.f)**2*V1+ 2*self.alpha * self.f * self.dt * self.Vavg(V0))/(1+(self.alpha * self.f * self.dt)**2)
            
            dh0=self.Ph(V1,h0)
            self.boundary(V1,V0,h0,h0,para='v')
            h1 = h0 + self.dt*dh0
            self.f3.append(dh0)
            if self.perturbation != None:
                h1=h1+ self.perturbation(i)
            
            self.boundary(V1,V0,h1,h0,'h')
            h.append(h1)
            V.append(V1)
            V0=V1
            #pvu.append(self.pvu(V1,h1))
            h0 = h1
        i=0
            
        for t in self.time_window[3:]:
            i=i+1
            
            
           
            V1 = self.euler(t,h0,V0,'V',self.dV)
            
            V1= (V1+(self.alpha * self.dt * self.f)**2*V1+ 2*self.alpha * self.f * self.dt * self.Vavg(V0))/(1+(self.alpha * self.f * self.dt)**2)
            self.boundary(V1,V0,h0,h0,'v')
            
            h1 = self.euler(t,h0,V1,'h',self.Ph)
            if self.perturbation != None:
                h1=h1+ self.perturbation(i)
            
            self.boundary(V1,V0,h1,h0,'h')
            V0 = V1
            h0 = h1
            
            if i%500==0:
                h.append(h1)
                V.append(V1)
                pvu.append(self.pvu(V1,h1))

        
        h.append(h1)
        V.append(V1)
        pvu.append(self.pvu(V1,h1))
        D={'h':h,'V':V,'pvu':pvu}
        return D    
    