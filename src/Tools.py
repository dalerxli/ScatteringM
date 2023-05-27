# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 23:04:32 2022

@author: Hoang
"""
import numpy as np

def Round2Reso(CD,reso=1):
    '''
    Input
        - CD: critical dimension in x y plane
        - reso: resolution in nm x nm   
    return   round CD respecting to reso 
    '''        
    interger=CD//reso
    lam=np.array([interger*reso-reso,interger*reso,interger*reso+reso])
    idx=np.argmin(np.abs(CD-lam))
    return lam[idx]

############ vector_based computation for homogeneous layers######################################

def diag4_inv(A):
    '''
    Input
        A: 2D or 3D matrix, which is composed of 4 diagonal matrices
    Return: fast inverse of A
    '''        
    if A.ndim==2:  #4 x Nharm  
       
        res=np.array([A[3], -A[1], -A[2],A[0]] )/(A[0]*A[3] -A[1]*A[2])
        
    elif A.ndim==3: #wl x 4 x Nharm
        res=np.array([A[:,3], -A[:,1], -A[:,2],A[:,0]] )/(A[:,0]*A[:,3] -A[:,1]*A[:,2])
        res=res.swapaxes(0,1)
    return res


def diag44_AB(A,B):
    '''
    Input
        A,B: 2D or 3D matrix, which is composed of 4 diagonal matrices
    Return: fast matmul(A,B)
    '''      
    if A.ndim==2: #4 x Nharm
        res=np.array([A[0]*B[0] +A[1]*B[2], A[0]*B[1] + A[1]*B[3],A[2]*B[0] +A[3]*B[2], A[2]*B[1] +A[3]*B[3] ]) 
        
    elif A.ndim==3: # wl x 4 x Nharm
        res=np.array([A[:,0]*B[:,0] +A[:,1]*B[:,2], A[:,0]*B[:,1] + A[:,1]*B[:,3],A[:,2]*B[:,0] +A[:,3]*B[:,2], A[:,2]*B[:,1] +A[:,3]*B[:,3] ])
        res=res.swapaxes(0,1) # wl x 4 x Nharm    
    return res

def diag2m(a):
    if  a.ndim==2:##(m,n)->(m,n,n) return 3D array with diagonal matrix
        a_m=np.einsum('ij,jk->ijk', a, np.eye(a.shape[1], dtype=a.dtype))
    else: a_m=np.diag(a)    #n ->(n,n)
    return a_m

def vec2m(A):
    # (4,n)->(2n,2n) or (m,4,n)->(m,2n,2n)      
    
    if A.ndim==2:# 4 x Nharm
        A=np.block([[np.diag(A[0]),np.diag(A[1])],[np.diag(A[2]),np.diag(A[3])]])
    elif A.ndim==3:
        A=np.block([[diag2m(A[:,0]),diag2m(A[:,1])],[diag2m(A[:,2]),diag2m(A[:,3])]])     
    return A

########## Optical Model#################################################################
def Tauc_Lorentz(E,parameter):# Eg,e_inf, A, E0, C
    Eg=parameter[0]
    e_inf=parameter[1]    
   
    parameter=parameter[2:]
    A=np.array([parameter[i] for i in range(len(parameter)) if i%3==0]).reshape(-1,1)
    E0=np.array([parameter[i] for i in range(len(parameter))if i%3==1]).reshape(-1,1)
    C=np.array([parameter[i] for i in range(len(parameter))if i%3==2]).reshape(-1,1)

    Eg2=Eg**2;   E2=E**2; C2=C**2; E02=E0**2

    e2= A*E0*C*(E-Eg)**2/(E*(E2 - E02)**2 + E*C2*E2)
    e2[:,E<Eg]=0# if E<=Eg:e2=0
    e2=np.sum(e2,axis=0)

    alpha=np.sqrt(4*E02 - C2); alpha2=alpha**2
    gamma=np.sqrt(E02 -C2/2)  ; gamma2=gamma**2
   
    Psi4=(E2 - gamma2)**2 +alpha2*C2/4    
   
    aln=(Eg2 -E02)*E2 +Eg2*C2 -E02*(E02+3*Eg2)
    atan=(E2-E02)*(E02+Eg2) +Eg2*C2

    t1=A*C*aln/(2*np.pi*Psi4*alpha*E0)*np.log((E02+Eg2+alpha*Eg)/(E02+Eg2 -alpha*Eg))
   
    t2= - A*atan/(np.pi*Psi4*E0)*(np.pi - np.arctan(2*Eg/C+alpha/C)+ np.arctan(-2*Eg/C+alpha/C)   )
   
    t3=4*A*E0*Eg*(E2-gamma2)/(np.pi*Psi4*alpha)*(np.arctan(alpha/C +2*Eg/C)+np.arctan(alpha/C -2*Eg/C))
   
    t4=-A*E0*C*(E2+Eg2)/(np.pi*Psi4*E)*np.log(np.abs(E-Eg)/(E+Eg))
   
    t5=2*A*E0*C*Eg/(np.pi*Psi4)*np.log(np.abs(E-Eg)*(E+Eg)/np.sqrt((E02-Eg2)**2 +Eg2*C2))
   
    e1= t1+t2+t3+t4+t5
    e1=np.sum(e1,axis=0)+e_inf
   
    return e1 -1j*e2

def Cauchy(wl_range,A,B,C,alpha,beta,wl0):
    n_=A + B/(wl_range)**2 + C/(wl_range)**4 

    k_=alpha*np.exp(beta*(1/(wl_range) - 1/wl0)) 

    N_=n_ + 1j*np.array(k_)
    e_=np.conj(N_**2)
    return e_

def Forouhi(E,parameters): #parameters=[A,B,C,Eg,n_inf]
    
    def nk_Forouhi(E,parameters):      
        A=parameters[0]
        B=parameters[1]
        C=parameters[2]
        Eg=parameters[3]
        n_inf=parameters[4]        
    
        #k_E eq. 15
        k_E=A/(E**2 - B*E + C)
        k_E=np.sum(k_E)*(E - Eg)**2  
    
        # n_E eq.19
        Q=0.5*((4*C - B**2)**0.5)
        C0=A/Q*((Eg**2 + C)*B/2 - 2*Eg*C)
        B0=A/Q*(-0.5*B**2 + Eg*B -Eg**2 + C)

        n_E=(B0*E + C0)/(E**2 - B*E + C)
        n_E=np.sum(n_E) + n_inf          

        return n_E, k_E      

    k=[];     n=[]
    for i in range(len(E)):
        n_E,k_E=nk_Forouhi(E[i],parameters)
        
        k.append(k_E);        n.append(n_E)  
    
    N_=np.array(n) + 1j*np.array(k)
    e_=np.conj(N_**2)
        
    return e_ 
   
#LM
