#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:07:24 2023

@author: pham
"""
import numpy as np
from src.Tools import diag4_inv,diag44_AB

def redheffer_global(unit_mat,sA, sB): # general case
     # S=sA ⨂ sB   
     # top down: sG=sG⨂ sL: A=Global, B=Layer        
     # global 4, layer 2 or 4

     sA11 = sA[0]    ;sA12 = sA[1];    sA21 = sA[2]    ;sA22 = sA[3]; 
     if len(sB)==2:
         sB11=sB[0]      ;sB12=sB[1];      sB21=sB12;    sB22=sB11
     else:
         sB11=sB[0]      ;sB12=sB[1];      sB21=sB[2];     sB22=sB[3]      
 
     d_mat = np.linalg.inv(unit_mat - sB11 @ sA22)
     f_mat = np.linalg.inv(unit_mat - sA22 @ sB11)       
 
     s11 = sA11 + sA12 @ d_mat@ sB11@ sA21  
     s12 = sA12 @ d_mat @ sB12
     s21 = sB21 @ f_mat @ sA21              
     s22 = sB22 + sB21 @ f_mat @ sA22 @ sB12       
 
     return s11,s12,s21,s22  
 
def redheffer_global_bottom_up(unit_mat,sA, sB,Sim='quarter'):
    # S=sA ⨂ sB   
    # bottop up: sG=sL⨂ sG: A=Layer, B=Global
    # layer 2 or 4, global can be 1

    # Sim='quarter': S11 for reflectance
    # Sim=half: S12, S21 for reflectance and transmission 
    # Sim=full: S11,S12,S21,S22 # we do not need it in EM       

    if len(sA)==2:
        sA11 = sA[0]    ;sA12 = sA[1];    sA21 = sA12    ;sA22 = sA11;
    else:
        sA11 = sA[0]    ;sA12 = sA[1];    sA21 = sA[2]    ;sA22 = sA[3];   

    sB11=sB[0]      

    d_mat = np.linalg.inv(unit_mat - sB11 @ sA22)
    s11 = sA11 + sA12 @ d_mat@ sB11@ sA21

    if Sim=='quarter':
        return s11     

    elif Sim=='half':       
        sB21=sB[1];
        f_mat = np.linalg.inv(unit_mat - sA22 @ sB11)   
        s21 = sB21 @ f_mat @ sA21  
        return s11,s21
    elif Sim=='full':   
        sB12=sB[1];      sB21=sB[2];     sB22=sB[3] 
        f_mat = np.linalg.inv(unit_mat - sA22 @ sB11)   
    
        s12 = sA12 @ d_mat @ sB12
        s21 = sB21 @ f_mat @ sA21              
        s22 = sB22 + sB21 @ f_mat @ sA22 @ sB12   
    
        return s11,s12,s21,s22
    

def redheffer_global_bottom_up_plane(unit,sA, sB,Sim='quarter'):
     # S=sA ⨂ sB   
     # bottop up: sG=sL⨂ sG: A=Layer, B=Global

     # Sim='quarter': S11 for reflectance
     # Sim=half: S12, S21 for reflectance and transmission 
     # Sim=full: S11,S12,S21,S22 # we do not need it in EM       

     if len(sA)==2:
         sA11 = sA[0]    ;sA12 = sA[1];    sA21 = sA12    ;sA22 = sA11;
     else:
         sA11 = sA[0]    ;sA12 = sA[1];    sA21 = sA[2]    ;sA22 = sA[3];   
 
     sB11=sB[0]      
     
     d_mat=diag4_inv(unit - diag44_AB(sB11 , sA22))    
     s11 = sA11 + diag44_AB(diag44_AB(sA12, d_mat),diag44_AB(sB11, sA21) ) 
 
     if Sim=='quarter':
         return s11     
 
     elif Sim=='half':       
         sB21=sB[1];            
         f_mat=diag4_inv(unit - diag44_AB(sA22 , sB11)) 
         s21 = diag44_AB(diag44_AB(sB21, f_mat),sA21)  
         return s11,s21
     elif Sim=='full':   
         sB12=sB[1];      sB21=sB[2];     sB22=sB[3] 
         
         f_mat=diag4_inv(unit - diag44_AB(sA22 , sB11))           
       
         s12= diag44_AB(diag44_AB(sA12, d_mat),sB12)   
         s21 = diag44_AB(diag44_AB(sB21, f_mat),sA21)              
         s22 = sB22 + sB21 @ f_mat @ sA22 @ sB12   
         
         s22 = sB22 + diag44_AB(diag44_AB(sB21, f_mat),diag44_AB(sA22, sB12) ) 
         return s11,s12,s21,s22  
     
     # top_down plane
     
############"Compute Mueller matrix#########################################################################
def Mueller_m(J11,J12,J21,J22): #rpp,rps,rsp,rss
    
    m11=0.5*(np.abs(J11)**2 + np.abs(J22)**2 + np.abs(J12)**2 + np.abs(J21)**2 )
    m12=0.5*(np.abs(J11)**2 - np.abs(J22)**2 - np.abs(J12)**2 + np.abs(J21)**2 )
    m13= np.real(np.conjugate(J11)*J12 + np.conjugate(J21)*J22)
    m14=-np.imag(np.conjugate(J11)*J12 + np.conjugate(J21)*J22)  
   
    m22=0.5*(np.abs(J11)**2 + np.abs(J22)**2 - np.abs(J12)**2 - np.abs(J21)**2 )
    m23=np.real(np.conjugate(J11)*J12 - np.conjugate(J21)*J22)
    m24=np.imag(-np.conjugate(J11)*J12 + np.conjugate(J21)*J22)
    
    m33=np.real(np.conjugate(J11)*J22 + np.conjugate(J12)*J21) 
    m34=np.imag(-np.conjugate(J11)*J22 + np.conjugate(J12)*J21)
    m44=np.real(np.conjugate(J11)*J22 - np.conjugate(J12)*J21)     
   
    return m11,m12,m13,m14,m22,m23,m24,m33,m34,m44

def Mueller_m_4(J11,J12,J21,J22): #rpp,rps,rsp,rss
    
    m11=0.5*(np.abs(J11)**2 + np.abs(J22)**2 + np.abs(J12)**2 + np.abs(J21)**2 )
    m12=0.5*(np.abs(J11)**2 - np.abs(J22)**2 - np.abs(J12)**2 + np.abs(J21)**2 )
    
    m33=np.real(np.conjugate(J11)*J22 + np.conjugate(J12)*J21) 
    m34=np.imag(-np.conjugate(J11)*J22 + np.conjugate(J12)*J21)     
   
    return m11,m12,m33,m34


def MM9_to_MM16(MM9):  #MM9:  9 x  wavelength
    MM16=np.ones((16,MM9.shape[1])) 

    MM16[1]=MM9[0]
    MM16[2]=MM9[1]
    MM16[3]=MM9[2]

    MM16[4]=MM9[0]
    MM16[5]=MM9[3]
    MM16[6]=MM9[4]
    MM16[7]=MM9[5]

    MM16[8]=-MM9[1]
    MM16[9]=-MM9[4]
    MM16[10]=MM9[6]
    MM16[11]=MM9[7]

    MM16[12]=MM9[2]
    MM16[13]=MM9[5]
    MM16[14]=-MM9[7]
    MM16[15]=MM9[8]  
    return MM16 #MM16:  16 x wavelength

