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
    # based on grid_type and obs


#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
from da_funcs import *


#%%============================================================================

# file organization
def file_subroutine(mapDIR,
                    modDIR,
                    piDIR,
                    obsDIR,
                    grid_type,
                    obs_types,
                    lulcc,
                    y1,
                    y2,
                    t_ext,
                    models,
                    exps,
                    var):
    
    map_files = {}
    grid_files = {}
    fp_files = {}
    pi_files = {}
    obs_files = {}

    #==============================================================================
            
    # map files
    os.chdir(mapDIR)
    
    if grid_type == 'obs':
        
        for obs in obs_types:
            
            map_files[obs] = {}
            grid_files[obs] = 'tasmax_obs_' + obs + '_gridarea.nc' # won't need this since map files at obs res are already in area
            
            for lu in lulcc:
            
                for file in [file for file in sorted(os.listdir(mapDIR))
                            if obs in file\
                            and lu in file\
                            and str(y1) in file\
                            and str(y2) in file]:
                    
                    map_files[obs][lu] = file   
    
    elif grid_type == 'model':
    
        for mod in models:
        
            map_files[mod] = {}
            grid_files[mod] = mod+'_gridarea.nc'
            
            for lu in lulcc:
                
                for file in [file for file in sorted(os.listdir(mapDIR))
                            if mod in file\
                            and lu in file\
                            and str(y1) in file\
                            and str(y2) in file\
                            and 'absolute_change' in file]:

                        map_files[mod][lu] = file

    #==============================================================================
    
    # model files
    os.chdir(modDIR)
    
    for mod in models:
        
        fp_files[mod] = {}
        
        for exp in exps:
            
            if grid_type == 'obs':
                
                fp_files[mod][exp] = {}
                
                for obs in obs_types:
                
                    fp_files[mod][exp][obs] = []
                
                    for file in [file for file in sorted(os.listdir(modDIR))\
                                if var in file\
                                and mod in file\
                                and exp in file\
                                and t_ext in file\
                                and obs in file\
                                and 'unmasked' in file\
                                and not 'ensmean' in file]:
                        
                        fp_files[mod][exp][obs].append(file)
                    
            elif grid_type == 'model':
                
                fp_files[mod][exp] = []
                
                for file in [file for file in sorted(os.listdir(modDIR))\
                            if var in file\
                            and mod in file\
                            and exp in file\
                            and t_ext in file\
                            and not obs_types[0] in file\
                            and not obs_types[1] in file\
                            and 'unmasked' in file\
                            and not 'ensmean' in file]:
                        
                    fp_files[mod][exp].append(file)                
                    
    #==============================================================================
    
    # pi files
    os.chdir(piDIR)

    for mod in models:
        
        if grid_type == 'obs':
            
            pi_files[mod] = {}
        
            for obs in obs_types:
                
                pi_files[mod][obs] = []
        
                for file in [file for file in sorted(os.listdir(piDIR))\
                            if var in file\
                            and mod in file\
                            and t_ext in file\
                            and obs in file\
                            and 'unmasked' in file]:
                    
                    pi_files[mod][obs].append(file)
                
        if grid_type == 'model':
            
            pi_files[mod] = []
        
            for file in [file for file in sorted(os.listdir(piDIR))\
                        if var in file\
                        and mod in file\
                        and t_ext in file\
                        and not obs_types[0] in file\
                        and not obs_types[1] in file\
                        and 'unmasked' in file]:
                
                pi_files[mod].append(file)
                
    #==============================================================================
    
    # obs files
    os.chdir(obsDIR)            
    
    if grid_type == 'obs':
        
        for obs in obs_types:
        
            for file in [file for file in sorted(os.listdir(obsDIR))\
                        if var in file\
                        and 'obs' in file\
                        and obs in file\
                        and not '-res' in file\
                        and 'unmasked' in file\
                        and t_ext in file]:
                
                obs_files[obs] = file
            
    elif grid_type == 'model':
        
        obs_files = {}

        for obs in obs_types:
        
            obs_files[obs] = {}
        
            for mod in models:
                
                for file in [file for file in sorted(os.listdir(obsDIR))\
                            if var in file\
                            and 'obs' in file\
                            and obs in file\
                            and mod + '-res' in file\
                            and 'unmasked' in file\
                            and t_ext in file]:
                    
                    obs_files[obs][mod] = file

    return     map_files,grid_files,fp_files,pi_files,obs_files