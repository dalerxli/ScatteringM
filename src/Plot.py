# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:54:33 2022

@author: Hoang
"""

import numpy as np
import matplotlib.pyplot as plt   
from src.NanoStructure import Fill_material, Geo

def Geo_viz(layer):  
    
    Nx=Geo.Nx
    Ny=Geo.Ny
        
    if layer['Shape'] =='Homo':        
            
        e_base=layer['e_base']                
        if np.isscalar (e_base): e_base0=e_base
        else: e_base0=e_base[0]           
                               
        geo_e=e_base0*np.ones((Nx, Ny))
                 
    else:            
        
        Center=layer['Center'];    
        
        Shape=layer['Shape']      
        CD=layer['Critical']
        
        mask=Shape(CD,Center)          
       
        e_base=layer['e_base']
        e_grt=layer['e_grt']
        if np.isscalar (e_base): e_base_wth=e_base
        else:e_base_wth=e_base[0]
        
        if np.isscalar (e_grt): e_grt_wth=e_grt
        else:e_grt_wth=e_grt[0]    
            
        geo_e=Fill_material(mask,e_base_wth,e_grt_wth)    
    
    return geo_e

def Viz_xy(geo_e,pos_layer=[0]):
    v_min=np.min(geo_e).real
    v_max=np.max(geo_e).real   
    
    if geo_e.shape[-1]==1: #2D
        print("The current version only support x-y view for 3D structures")
    else:
        
        if len(pos_layer) >1:
            fig_size=(15,4)
            fig, ax = plt.subplots(nrows=1, ncols=len(pos_layer),figsize=fig_size)
            for i in range(len(pos_layer)):
                im=ax[i].imshow(geo_e[i].real,vmin=v_min, vmax=v_max,origin='lower')
                ax[i].title.set_text('Layer '+str(pos_layer[i]))
                if i==0:
                    ax[i].set_xlabel('x (pixel)')
                    ax[i].set_ylabel('y (pixel)')
                else:
                    ax[i].set_xticks([]);       ax[i].set_yticks([])       
        else:
            fig_size=(5,4)
            fig, ax = plt.subplots(1, 1,figsize=fig_size)
        
            lth=pos_layer[0]
            im=ax.imshow(geo_e[lth].real,vmin=v_min, vmax=v_max,origin='lower')
            ax.title.set_text('Layer '+str(lth))
            
            ax.set_xlabel('x (pixel)')
            ax.set_ylabel('y (pixel)')           
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)   
    
        #fig.suptitle('Nanostructure in x-y view ',fontsize=16,y=-0.01)
        fig.suptitle('Nanostructure in x-y view (1 pixel = {} nm)'.format(Geo.reso),fontsize=16,y=-0.01)
        
        plt.show()
    
    
def broad_height(geo_t,Thickness_Sim):
    reso=Geo.reso
    geo_plot=np.broadcast_to(geo_t[0],(int(Thickness_Sim[0]/reso),)+geo_t[0].shape) 
    if len(geo_t)>1:
        for lth in range(len(geo_t))[1:]:    
            geo_plot_lth=np.broadcast_to(geo_t[lth],(int(Thickness_Sim[lth]/reso),)+geo_t[lth].shape) # transform 1D array to 2D array to plot
            geo_plot=np.vstack((geo_plot,geo_plot_lth)) 
    return geo_plot
    

def Viz_z(geo_t,Thickness_Sim):
    
    if geo_t.shape[-1]==1: #2D
        
        geo_xz=geo_t[:,:,0]
        geo_xz_plot=broad_height(geo_xz,Thickness_Sim) 
        
        fig_size=(8,4)
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=fig_size)
        v_min=np.min(geo_t).real
        v_max=np.max(geo_t).real 

        im=ax.imshow(geo_xz_plot.real,vmin=v_min, vmax=v_max)    
        ax.set_xlabel('x (pixel)')
        ax.set_ylabel('z (pixel)')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)   
    
        #fig.suptitle('Nanostructure in x-z',fontsize=16,y=-0.02)  
        fig.suptitle('Nanostructure in x-z view (1 pixel = {} nm)'.format(Geo.reso),fontsize=16,y=-0.02)
     

        plt.show()  
        
    else:
        Nx=geo_t.shape[1]//2
        Ny=geo_t.shape[2]//2

        geo_xz=geo_t[:,:,Ny]
        geo_xz_plot=broad_height(geo_xz,Thickness_Sim)

        geo_yz=geo_t[:,Nx,:]
        geo_yz_plot=broad_height(geo_yz,Thickness_Sim)

        fig_size=(8,4)
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=fig_size)
        v_min=np.min(geo_t).real
        v_max=np.max(geo_t).real 

        im=ax[0].imshow(geo_xz_plot.real,vmin=v_min, vmax=v_max)    
        ax[0].set_xlabel('x (pixel)')
        ax[0].set_ylabel('z (pixel)')

        im=ax[1].imshow(geo_yz_plot.real,vmin=v_min, vmax=v_max)   
        ax[1].set_xlabel('y (pixel)')
        ax[1].set_yticks([])
        #ax[1].set_ylabel('z (nm)')
    

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)   
    
        #fig.suptitle('Nanostructure in x-z and y-z views',fontsize=12,y=-0.01)  
        fig.suptitle('Nanostructure in x-y and y-z view (1 pixel = {} nm)'.format(Geo.reso),fontsize=14,y=-0.02)
     

        plt.show() 