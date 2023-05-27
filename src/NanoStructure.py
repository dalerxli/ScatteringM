#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:31:16 2023

@author: pham
"""

import numpy as np

################### CONV######################################################################


def Fill_material(mask,e_base,e_grt):         
    return e_grt*mask   + e_base*(1-mask)   

def Sigmoid (x):
    return 1/(1 + np.exp(-x))

class Geo:
    reso=1
    Lx_cell=500      ;Ly_cell=500    
    mx=2        ;my=2
    edge_sharpness = 500.   # sharpness of edge
    num_cell_x=1
    num_cell_y=1
        
    def __init__(self):
        pass
    
    @classmethod
    def Initilize(cls):                    
   
        if cls.Lx_cell==float('inf'): 
            cls.Nx=1;   cls.Nx_cell=1;    cls.Lx=cls.Lx_cell 
        else:  
            cls.Lx=cls.Lx_cell*cls.num_cell_x   
               
            cls.Nx_cell = int(cls.Lx_cell/cls.reso);  
            cls.Nx = int(cls.Lx/cls.reso); 
            
        
        if cls.Ly_cell==float('inf'): 
            cls.Ny=1;   cls.Ny_cell=1;     cls.Ly=cls.Lx_cell      
        
        else:    
            cls.Ly=cls.Ly_cell*cls.num_cell_y   
                   
            cls.Ny_cell = int(cls.Ly_cell/cls.reso);  
            cls.Ny = int(cls.Ly/cls.reso); 
                 
        
        x = np.arange(cls.Nx_cell)+0.5 
        y = np.arange(cls.Ny_cell)+0.5       
        
        cls.x_grid, cls.y_grid = np.meshgrid(x,y,indexing='ij')   
        
    @classmethod        
    def Circle(cls,CD,Center):       
        R=CD[0]/cls.reso/2
        Cx,Cy=Center[0]/cls.reso,Center[1]/cls.reso
        
        level = 1. - np.sqrt(((cls.x_grid-Cx)/R)**2 + ((cls.y_grid-Cy)/R)**2)
        return Sigmoid(cls.edge_sharpness*level)
    
        
    @classmethod
    def Rectangle(cls,CD,Center,theta=0.):
        # [Wx,Wy]: x width, y width; [Cx,Cy]: x center, y center
        # theta: rotation angle / center: [Cx, Cy] / axis: z-axis  
        Wx,Wy=CD[0]/cls.reso,CD[1]/cls.reso
        Cx,Cy=Center[0]/cls.reso,Center[1]/cls.reso       
                
        level = 1. - (np.maximum(np.abs(((cls.x_grid-Cx)*np.cos(theta)+(cls.y_grid-Cy)*np.sin(theta))/(Wx/2.)),
                                 np.abs((-(cls.x_grid-Cx)*np.sin(theta)+(cls.y_grid-Cy)*np.cos(theta))/(Wy/2.))))
        return Sigmoid(cls.edge_sharpness*level)
    
      
    @classmethod
    def Rectangle2D(cls,CD,Center):
        # [Wx]: x width; [Cx]: x center        
        Wx=CD[0]/cls.reso
        Cx=Center[0]/cls.reso                        
        level = 1. - np.abs((cls.x_grid-Cx)/(Wx/2.))
                                 
        return Sigmoid(cls.edge_sharpness*level)
    
    @classmethod
    def Rectangle2D_array(cls,CD,Center):
        # [Wx]: x width; [Cx]: x center 
        Cx=Center[0]/cls.reso
        Wx_all=CD[0]/cls.reso
        
        for lth in range(len( Wx_all)):                                     
            level = 1. - np.abs((cls.x_grid-Cx)/(Wx_all[lth]/2.))
            level=Sigmoid(cls.edge_sharpness*level)
            
            if lth==0: geo=np.array(level)
            else: geo=np.vstack((geo,level))
                              
        return geo
    
    @classmethod
    def Rectangle3D_array(cls,CD,Center,theta=0):
        # [Wx]: x width; [Cx]: x center 
        Cx=Center[0]/cls.reso; Cy=Center[1]/cls.reso
        Wx=CD[0]/cls.reso
        Wy=CD[1]/cls.reso
        
        if len(CD)==2: theta=0
        
        for i in range(cls.num_cell_x):    
            for j in range(cls.num_cell_y):                 
                
                if len(CD)>2: theta=CD[2][i][j]
                
                level = 1. - (np.maximum(np.abs(((cls.x_grid-Cx)*np.cos(theta)+(cls.y_grid-Cy)*np.sin(theta))/(Wx[i][j]/2.)), \
                                        np.abs((-(cls.x_grid-Cx)*np.sin(theta)+(cls.y_grid-Cy)*np.cos(theta))/(Wy[i][j]/2.))))
          
            
                level=Sigmoid(cls.edge_sharpness*level)       
                if j==0:        
                    geo_y=np.array(level)
                else:
                    geo_y=np.hstack((geo_y,level))
                    
            if i==0:geo_xy=np.copy(geo_y)
            else: geo_xy=np.vstack((geo_xy,geo_y))
        return geo_xy
    
    @classmethod 
    def Convmat2D(cls,A):         
   
        N = A.shape;

        NH = (2*cls.mx+1) * (2*cls.my+1) # harmonic number
    
        p = list(range(-cls.mx, cls.mx + 1)); 
        q = list(range(-cls.my, cls.my + 1));
  
        Af = (1 / np.prod(N)) * np.fft.fftshift(np.fft.fft2(A));        
    
        p0 = int((N[1] / 2));     q0 = int((N[0] / 2)); 
    
     
        ret = np.zeros((NH, NH),dtype=complex)   
        for qrow in range(2*cls.my+1): 
            for prow in range(2*cls.mx+1):             
                row = (qrow) * (2*cls.mx+1) + prow; 
                for qcol in range(2*cls.my+1):
                    for pcol in range(2*cls.mx+1):
                        col = (qcol) * (2*cls.mx+1) + pcol; 
                        pfft = p[prow] - p[pcol]; 
                        qfft = q[qrow] - q[qcol];
                        ret[row, col] = Af[q0 + pfft, p0 + qfft]; 

        return ret
    
    @classmethod  
    def CONV_layer(cls,wavelength_range,layer):  
        
        if layer['Shape'] =='Homo':           
            
            ERC_CONV=[]
            e_base=layer['e_base'] 
            if np.isscalar (e_base):                     
                for wth in range(len(wavelength_range)): 
                    ERC_CONV.append(e_base)
            else:                    
                for wth in range(len(wavelength_range)):                         
                    ERC_CONV.append(e_base[wth])    
       
        else: #grating layers    
            
            e_base=layer['e_base']
            e_grt=layer['e_grt']
            
            CD=layer['Critical']
            Shape=layer['Shape']
            Center=layer['Center']                     
           
            mask=Shape(CD,Center)   
            
            ERC_CONV=[]    
            for wth in range(len(wavelength_range)):       
                if np.isscalar (e_base): e_base_wth=e_base
                else:e_base_wth=e_base[wth]
        
                if np.isscalar (e_grt): e_grt_wth=e_grt
                else:e_grt_wth=e_grt[wth]    
            
                geo_e=Fill_material(mask,e_base_wth,e_grt_wth)          
                erc_conv = Geo.Convmat2D(geo_e)    
                ERC_CONV.append(erc_conv )   
    
        return np.array(ERC_CONV)  
    
    
    
    def Split_layer(TCD,BCD,N_split=10): # used for lamellar layers
    # TCD,BCD: top and bottom critical dimension
        d=(BCD-TCD)/N_split
        CD_range=np.array([TCD+i*d for i in range(N_split+1)])
        CD_range=(CD_range[1:]+CD_range[:-1])/2    
        return CD_range  
   
    def Split_ellipso_h_even(r0,h0,N_split=10):
        r_range=[]
        for i in range(N_split)[::-1]:
            h_i=i*h0/N_split
            r_i_2=r0**2-(r0*h_i/h0)**2 
            r_i=np.sqrt(r_i_2)        
            r_range.append(r_i)
            Thickness=[h0/N_split]*N_split
        return np.array(r_range),Thickness

    def Split_ellipso(r0,h0,N_split=10):
        delta_r=r0/N_split
        r_range=np.array([delta_r*i for i in range(0,N_split+1)])
        
        y_cord=[]
        for i in range(N_split+1):
            h2=h0**2-(h0/r0*r_range[i])**2
            y_cord.append(np.sqrt(h2))
        
        Thickness=[]
        for i in range(1,len(y_cord)):
            h=y_cord[i-1]-y_cord[i]
            Thickness.append(h)
        r_range=(r_range[1:]+r_range[:-1])/2
        return r_range,Thickness
    