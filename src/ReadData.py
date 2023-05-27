# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 20:36:26 2022

@author: Hoang
"""


import numpy as np
from scipy import interpolate
import re
import os


def read_nk(path):
    f=open(path)
    f1=f.readlines()
    f_str=np.array([i.split() for i in f1 if len(i)!=1])
    f_str=f_str[1:]
    if len(f_str[0])==3:
        wl=f_str[:,0].astype(np.float)
        n=f_str[:,1].astype(np.float)
        k=f_str[:,2].astype(np.float)        
        return wl, [n,k]  
    else:
        wl=f_str[:,0].astype(np.float)
        n=f_str[:,1].astype(np.float)       
        return wl, [n]    

def Interpolate_Optical_Index(wl_target,wl_data, nk_data,n=10):  
    '''
    Input 
        - wl_target,wl_data: wavelength in nm
        - nk_data
    Return: dielectric 
    '''       
    num=len(wl_data)*n # number of point for interpolation 
    
    W_interpolate=np.linspace(wl_data.min(), wl_data.max(), num=num, endpoint=True)
    
    n_data=nk_data[0]
    f_n = interpolate.interp1d(wl_data, n_data)
    n_interpolate=f_n(W_interpolate)
    
    if len(nk_data)>1:
        k_data=nk_data[-1]
        f_k = interpolate.interp1d(wl_data, k_data)
        k_interpolate=f_k(W_interpolate)  
    
    
    index_target=[]
    for wavelength_i in wl_target:
        index_min=np.argmin(np.abs(W_interpolate - wavelength_i))
        index_target.append(index_min )
    
    if len(nk_data)>1:    
        N_index=np.array([n_interpolate[index]+1j*k_interpolate[index] for index in index_target])
    else:
        N_index=np.array([n_interpolate[index] for index in index_target])
    
    return np.conj(N_index**2)




    
