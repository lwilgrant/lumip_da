#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# This subroutine script generates:
    # dictionaries containing keyed ties to files for models, pichunks and obs
    # based on grid and obs


#%%============================================================================
# import
# =============================================================================

import os
import xarray as xr
from icv_funcs import *

#%%============================================================================

# file organization
def file_subroutine(
    mapDIR,
    modDIR,
    piDIR,
    allpiDIR,
    pi,
    obs_types,
    lulcc,
    y1,
    y2,
    t_ext,
    models,
    exps,
    var
):
    
    map_files = {}
    grid_files = {}
    fp_files = {}
    pi_files = {}
    nx = {}  
    
    #==============================================================================
    
    # grid files
    os.chdir(mapDIR)
    
    for mod in models:
            
        for file in [
            file for file in sorted(os.listdir(mapDIR))\
            if file == '{}_gridarea.nc'.format(mod)
        ]:
                
            grid_files[mod] = file
                    

    #==============================================================================
    
    # model files
    os.chdir(modDIR)
    
    for mod in models:
        
        fp_files[mod] = {}
        nx[mod] = {}
        
        for exp in exps:
            
            e_i = 0
                
            fp_files[mod][exp] = []
            
            for file in [
                file for file in sorted(os.listdir(modDIR))\
                if var in file\
                and mod in file\
                and exp in file\
                and t_ext in file\
                and not obs_types[0] in file\
                and not obs_types[1] in file\
                and 'unmasked' in file\
                and not 'ensmean' in file
            ]:
                    
                fp_files[mod][exp].append(file)  
                e_i += 1
            
            if exp == 'historical' or exp == 'hist-noLu':    
                nx[mod][exp] = e_i
        
        nx[mod] = np.array([[nx[mod]['historical'],nx[mod]['hist-noLu']]])
                    
    #==============================================================================
    
    # pi files
    os.chdir(piDIR)
    
        
    for mod in models:
            
        pi_files[mod] = []
    
        for file in [
            file for file in sorted(os.listdir(piDIR))\
            if var in file\
            and mod in file\
            and t_ext in file\
            and not obs_types[0] in file\
            and not obs_types[1] in file\
            and 'unmasked' in file
        ]:
                
            pi_files[mod].append(file)
    
    return     map_files,grid_files,fp_files,pi_files,nx