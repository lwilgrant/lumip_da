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
from lu_funcs import *


#%%============================================================================

# file organization
def file_subroutine(mapDIR,
                    modDIR,
                    piDIR,
                    stat,
                    lulcc,
                    y1,
                    y2,
                    t_ext,
                    models,
                    exps,
                    var):
    
    map_files = {}
    grid_files = {}
    mod_files = {}
    pi_files = {}

    #==============================================================================
            
    # map files
    os.chdir(mapDIR)
    
    for mod in models:
    
        map_files[mod] = {}
        grid_files[mod] = mod+'_gridarea.nc'
        
        for lu in lulcc:
            
            for file in [file for file in sorted(os.listdir(mapDIR))
                        if mod in file\
                        and lu in file\
                        and str(y1)+'01' in file\
                        and str(y2)+'12' in file\
                        and stat in file]:

                    map_files[mod][lu] = file

    #==============================================================================
    
    # model files
    os.chdir(modDIR)
    
    for mod in models:
        
        mod_files[mod] = {}
        
        for exp in exps:
                
            mod_files[mod][exp] = []
            
            for file in [file for file in sorted(os.listdir(modDIR))\
                        if var in file\
                        and mod in file\
                        and exp in file\
                        and t_ext in file\
                        and not 'berkley_earth' in file\
                        and not 'cru' in file\
                        and 'unmasked' in file\
                        and not 'ensmean' in file]:
                    
                mod_files[mod][exp].append(file)                
                    
    #==============================================================================
    
    # pi files
    os.chdir(piDIR)

    for mod in models:
            
        pi_files[mod] = []
    
        for file in [file for file in sorted(os.listdir(piDIR))\
                    if var in file\
                    and mod in file\
                    and '196501-201412' in file\
                    and not 'berkley_earth' in file\
                    and not 'cru' in file\
                    and 'unmasked' in file]:
            
            pi_files[mod].append(file)

    return     map_files,grid_files,mod_files,pi_files