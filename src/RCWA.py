# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:02:20 2022

@author: Hoang
"""


import numpy as np
from src.Tools import diag4_inv,diag44_AB,diag2m,vec2m
from numpy.linalg import solve
from src import Computation
from src.NanoStructure import Geo

    
def Polarization_vector(AOI_r,Azimuth_r,pol):
    # from the angle of incidence, azimuth, polarization 
    Px=np.cos(pol)*np.cos(AOI_r)*np.cos(Azimuth_r) - np.sin(pol)*np.sin(Azimuth_r)
    Py=np.cos(pol)*np.cos(AOI_r)*np.sin(Azimuth_r) + np.sin(pol)*np.cos(Azimuth_r)    
    Pz=-np.cos(pol)*np.sin(AOI_r)    
    return np.array([Px,Py,Pz])

def Homogeneous(Kx,Ky,e_r,m_r=1):          
        arg = (np.conj(m_r)*np.conj(e_r)-Kx**2-Ky**2);    
        arg = arg.astype('complex');
        Kz = np.conj(np.sqrt(arg));                   
        eigen_v=1j*Kz         
        V=np.array([Kx*Ky/eigen_v,(e_r-Kx**2)/eigen_v,(Ky**2-e_r)/eigen_v,-Kx*Ky/eigen_v])
        return V,Kz# array of only diagonal

class Scattering:     
    AOI_r=0     ;Azimuth_r=0    
    kz_inc=1
    
     
    def __init__(self):
        pass
    
    @classmethod
    def Meshgrid(cls):        
        
        Geo.Initilize()        
        
        cls.Nharm=(2*Geo.mx+1)*(2*Geo.my+1)
        
        cls.mu_conv=np.identity(cls.Nharm)
        cls.unit=np.array([np.ones(cls.Nharm),np.zeros(cls.Nharm),np.zeros(cls.Nharm),np.ones(cls.Nharm)])  
        cls.unit_mat = np.identity(2*cls.Nharm)          
        
                        
        cls.Lx=Geo.Lx;        cls.Ly=Geo.Ly
        cls.mx=Geo.mx;        cls.my=Geo.my
        
    @classmethod
    def Angles(cls):  
        if cls.AOI_r==0: cls.AOI_r=np.radians(0.001)
         
        #if cls.Azimuth_r==0: cls.Azimuth_r=np.radians(0.001)
        
                
############Scattering matrix####################################################    
    @classmethod
    def KxKy_Component(cls,wavelength,e_ref,e_trn,u_ref=1):
        
        # Kx,Ky,kz_inc
        cls.k0=2*np.pi/wavelength
        n_i =  np.sqrt(e_ref*u_ref)        
        kx_inc = n_i * np.sin(cls.AOI_r) * np.cos(cls.Azimuth_r);
        ky_inc = n_i * np.sin(cls.AOI_r) * np.sin(cls.Azimuth_r);      
        cls.kz_inc = np.sqrt(n_i**2 - kx_inc ** 2 - ky_inc ** 2);
    
        k_x = kx_inc - 2*np.pi*np.arange(-cls.mx, cls.mx+1)/(cls.k0*cls.Lx);
        k_y = ky_inc - 2*np.pi*np.arange(-cls.my, cls.my+1)/(cls.k0*cls.Ly);

        Kx, Ky = np.meshgrid(k_x, k_y,indexing='ij');   
            
        cls.Kx= Kx.flatten(order = 'C')
        cls.Ky=Ky.flatten(order = 'C') 
        
        cls.Vr,Kzr = Homogeneous(cls.Kx,cls.Ky,e_ref) #reflection Medium         
        cls.Kzr=-Kzr        
       
        
        cls.Vt,Kzt=Homogeneous(cls.Kx,cls.Ky,e_trn) #transmission Medium          
        cls.Kzt=Kzt
        cls.Vg,_ = Homogeneous(cls.Kx,cls.Ky,1);        #gap Medium             
        
   
    @classmethod   
    def S_relate_eigen(cls,e_conv):   
        Vg_grt=vec2m(cls.Vg)
        e_inv=np.linalg.inv(e_conv)
        term_y=e_inv@diag2m(cls.Ky);                term_x=e_inv@diag2m(cls.Kx)
        
        #P matrix        
        P11=diag2m(cls.Kx)@term_y
        P12= cls.mu_conv -diag2m(cls.Kx)@term_x
        P21=diag2m(cls.Ky)@term_y - cls.mu_conv
        P22=-diag2m(cls.Ky)@term_x               
    
        P=np.block([[P11,P12],[P21,P22]])
        #Q matrix
        Q11=diag2m(cls.Kx*cls.Ky)
        Q12=e_conv - diag2m(cls.Kx*cls.Kx)
        Q21=diag2m(cls.Ky*cls.Ky) - e_conv
                  
        Q=np.block([[Q11,Q12],[Q21,-Q11]])    
                
        Gamma_squared = P@Q;   
        Lambda,cls.W_i = np.linalg.eig(Gamma_squared);            
        cls.lambda_matrix = np.lib.scimath.sqrt(Lambda.astype('complex'));                
            
        cls.V_i=Q @ cls.W_i @ np.linalg.inv(diag2m(cls.lambda_matrix))  
            
        term1=np.linalg.inv(cls.W_i)
        term2=np.linalg.inv(cls.V_i)@ Vg_grt
            
        cls.A = term1 + term2
        cls.B = term1 - term2   
        
        cls.term_AB=cls.A-cls.B@solve(cls.A,cls.B)
        cls.X=-cls.lambda_matrix*cls.k0 
    
    @classmethod   
    def S_relate_h(cls,Li):  
                   
        X=np.exp(cls.X*Li)           
        X=diag2m(X) 
        
        term=X@cls.B@solve(cls.A,X)            
        term_s=np.linalg.inv(cls.A-term@cls.B)    
            
        S11=term_s@ (term@cls.A -cls.B)
        S12=term_s@ X@cls.term_AB     
        
        return S11,S12           

    @classmethod   
    def S_Layer(cls,Li,e_conv,need_eigen=True,based_homo='matrix'):     
        if np.isscalar(e_conv):            
            S11,S12= Scattering.S_Homogeneous(Li,e_conv,based=based_homo)
        else:
            if need_eigen:
                Scattering.S_relate_eigen(e_conv)            
                S11,S12  =Scattering.S_relate_h(Li)  
            else:
                S11,S12  =Scattering.S_relate_h(Li)
        return [S11,S12]       
        
    @classmethod   
    def S_Homogeneous(cls,Li,e_conv,based='matrix'):      
        e_conv_h=e_conv*np.ones(cls.Nharm)
        
        V_i,_=Homogeneous(cls.Kx, cls.Ky, e_conv_h)
        Gamma_squared=cls.Kx*cls.Kx + cls.Ky*cls.Ky -e_conv_h
        lambda_matrix=np.lib.scimath.sqrt(Gamma_squared.astype('complex'))
        x= np.exp(-lambda_matrix*cls.k0*Li)     
    
    
        A = cls.unit + diag44_AB(diag4_inv(V_i), cls.Vg)
        B = 2*cls.unit - A  
            
        term=x*diag44_AB(B,diag4_inv(A))
            
        S11=diag44_AB(diag4_inv(A -diag44_AB(term,x*B)),diag44_AB(term,x*A)-B)
        S12=diag44_AB(diag4_inv(A -diag44_AB(term,x*B)),x*A-diag44_AB(term,B))   
        if based=='vector':
            return [S11,S12]
        if based=='matrix':
            return [vec2m(S11), vec2m(S12)]         
    
    @classmethod 
    def Ref_medium(cls,based='matrix'):                  
     
        term_Vr=diag44_AB(diag4_inv(cls.Vg),cls.Vr)    #solve(Vg,Vr)
        
        Ar = cls.unit +term_Vr     ;Br = cls.unit - term_Vr
        Ar_inv=diag4_inv(Ar)
        
        S_ref_11 = -diag44_AB(Ar_inv,Br)              #S_ref_11 = - np.linalg.inv(Ar)@Br
        S_ref_12=2*Ar_inv                             # S_ref_21 = 2*np.linalg.inv(Ar)
        S_ref_21=2*cls.unit-S_ref_12
        S_ref_22=-S_ref_11     
        S_ref=[S_ref_11,S_ref_12,S_ref_21,S_ref_22]
        if based=='vector':
            return S_ref   
        if based=='matrix':
            return [vec2m(S_ref[0]),vec2m(S_ref[1]),vec2m(S_ref[2]),vec2m(S_ref[3])]
    
    @classmethod
    def Trn_medium(cls,based='matrix'):  
        
        term_Vt=diag44_AB(diag4_inv(cls.Vg),cls.Vt)    #solve(Vg,Vt)
        
        At = cls.unit +term_Vt     ;Bt = cls.unit - term_Vt
        
        At_inv=diag4_inv(At)
        
        S_trn_11 = diag44_AB(Bt,At_inv)              #S_trn_11 = Bt@ np.linalg.inv(At)
        S_trn_21=2*At_inv                            #S_trn_21 = 2*np.linalg.inv(At)
        S_trn_12=2*cls.unit-S_trn_21
        S_trn_22=-S_trn_11   
        S_trn=[S_trn_11,S_trn_12,S_trn_21,S_trn_22] 
        if based=='vector':
            return S_trn  
        if based=='matrix':
            return [vec2m(S_trn[0]),vec2m(S_trn[1]),vec2m(S_trn[2]),vec2m(S_trn[3])]
    
    @classmethod
    def S_System(cls,S_layer,need_S_trn=True): 
        
        S_global=Scattering.Ref_medium()  
        for lth in range(len(S_layer)):
            S_global=Computation.redheffer_global(cls.unit_mat,S_global, S_layer[lth])    
        if need_S_trn:
            S_trn=Scattering.Trn_medium() 
            S_global=Computation.redheffer_global(cls.unit_mat,S_global, S_trn) 
        return S_global     
                
############### Output########################################################    
    @classmethod    
    def Optical_Response(cls,S_global_sub,polar='pte'):   
        
        if polar=='pte':
            cls.pte=Polarization_vector(cls.AOI_r,cls.Azimuth_r,np.radians(90))
            Polarization=cls.pte
        else: 
            cls.ptm=Polarization_vector(cls.AOI_r,cls.Azimuth_r,0)
            Polarization=cls.ptm
        
        #optical responses: reflectance or transmittance, 
        #depending on S_global_sub: S_global_11 for Reflectance and  S_global_21 for Transmittance   
        delta_vec = np.zeros(cls.Nharm)
        delta_vec[int(np.floor(cls.Nharm/2))] = 1       
           
        #Compute Source Field
        E_inc = np.zeros(2*cls.Nharm).astype('complex')    
        E_inc[0:cls.Nharm] = Polarization[0]*delta_vec
        E_inc[cls.Nharm:2*cls.Nharm] = Polarization[1]*delta_vec  
    
        #Step 12: Compute Reflected  Fields 
        #Compute mode coefficients of the source    
        cls.c_inc=E_inc.reshape(2*cls.Nharm,1)    
       
        #Compute Compute reflected and transmitted fields
        E_ref = S_global_sub@ cls.c_inc # x, y directions: E_ref_xy # Wref or Wtrn=I
        if len(E_ref.shape) <3:
            rx=E_ref[0:cls.Nharm,:]
            ry=E_ref[cls.Nharm:2*cls.Nharm,:] 
        else:
            rx=E_ref[:,0:cls.Nharm,:]
            ry=E_ref[:,cls.Nharm:2*cls.Nharm,:] 
    
        return rx,ry    
       
    @classmethod
    def Rotate_SP(cls,Coeff_XY):  
        
        # Euler rotation   
        rxP=Coeff_XY[0];    ryP=Coeff_XY[1]
        rxS=Coeff_XY[2];    ryS=Coeff_XY[3]
   
        #TM
        rpx_new=-np.array(rxP)*np.cos(cls.Azimuth_r) - np.array(ryP)*np.sin(cls.Azimuth_r)
        rpp=rpx_new/np.cos(cls.AOI_r)
        rps=-np.array(rxP)*np.sin(cls.Azimuth_r) + np.array(ryP)*np.cos(cls.Azimuth_r)

        #TE
        rsx_new=-np.array(rxS)*np.cos(cls.Azimuth_r) -np.array(ryS)*np.sin(cls.Azimuth_r)
        rsp=rsx_new/np.cos(cls.AOI_r)
        rss=-np.array(rxS)*np.sin(cls.Azimuth_r) + np.array(ryS)*np.cos(cls.Azimuth_r)
    

        return rpp, rps,rsp,rss

    @classmethod
    def Reflectance(cls,rx,ry):
        rz=-(cls.Kx*rx[:,0]+cls.Ky*ry[:,0])/cls.Kzr
        r2=np.square(np.abs(rx[:,0])) + np.square(np.abs(rz))+np.square(np.abs(ry[:,0]))
        R=np.real(-cls.Kzr)* r2/np.real(cls.kz_inc)     
        return R

    @classmethod
    def Transmittance(cls,tx,ty):
        tz=-(cls.Kx*tx[:,0]+cls.Ky*ty[:,0])/cls.Kzt
        t2=np.square(np.abs(tx[:,0])) + np.square(np.abs(ty[:,0]))+ np.square(np.abs(tz))
        T=np.real(cls.Kzt)* t2/np.real(cls.kz_inc)        
        return T
        
        
    
        