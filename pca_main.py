#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# PCA analysis
# update November 13:
    # adding subroutines from da*.py to here; adjusting for this analysis
    # instead of making 2d arrays for ar6 organization, make new data arrays (xr) for ar6 means
    # run pca on global series of ar6 means as well as earlier scales
        # potential for ar6-weighted mean run to then show likewise regions from OF contributing to signal 
        # (e.g. detected regions from OF will have high loadings)
    # check that eofs of individual model means differ between historical and hist-noLu

# repeat latitudinal, continental etc options for luh2
# pseudo-pcs differ for 2nd eof (proj obs onto hist vs histnolu)
    # n-temperate hist has more cooling in NA, for example, but histnolu has morxe cooling in China
    # project LU onto 2nd eof for historical vs hist-nolu?

# removed pi files from fp_sr_file_alloc temporarily, until I get new organization with sr's done
# send modres crop and deforest bash scripts for redoing lu/lulcc eofs ()

#%%==============================================================================
# import
# ===============================================================================


import sys
import os
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import copy as cp
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
import regionmask as rm
from random import shuffle
from matplotlib.lines import Line2D
import xarray as xr
from scipy import stats as sts
from eofs.xarray import Eof


#%%==============================================================================
# path
#================================================================================


# curDIR = '/home/luke/documents/lumip/d_a/'
# curDIR = '/theia/data/brussel/vo/000/bvo00012/vsc10116/lumip/d_a'
# curDIR = '/Users/Luke/Documents/PHD/lumip/da'
# curDIR = '//wsl$/Ubuntu//home//luke//mnt//c//Users//lgrant//documents//repos//lumip_da'
curDIR = 'C:/Users/lgrant/Documents/repos/lumip_da'
os.chdir(curDIR)

# data input directories
obsDIR = os.path.join(curDIR, 'obs')
modDIR = os.path.join(curDIR, 'mod')
piDIR = os.path.join(curDIR, 'pi')
mapDIR = os.path.join(curDIR, 'map')
outDIR = os.path.join(curDIR, 'figures')

# bring in functions
from pca_funcs import *


#%%==============================================================================
# options - analysis
#================================================================================


# adjust these flag settings for analysis choices only change '<< SELECT >>' lines

flag_pickle=1     # 0: do not pickle objects
                  # 1: pickle objects after sections 'read' and 'analyze'

# << SELECT >>
flag_svplt=1      # 0: do not save plot
                  # 1: save plot in picDIR
                  
# # << SELECT >>
# flag_lulcc=0      # 0: treeFrac
#                   # 1: cropFrac

# << SELECT >>
flag_lulcc_measure=0    # 0: absolute change
                        # 1: area change
                        
# << SELECT >>
flag_lulcc_stat=0    # 0: annual mean cover
                     # 1: annual max cover
                        
# << SELECT >>
flag_lu_technique=1     # 0: lu as mean of individual (historical - hist-nolu)
                        # 1: lu as mean(historical) - mean(hist-nolu)

# << SELECT >>
flag_y1=1         # 0: 1915
                  # 1: 1965

# << SELECT >>
flag_len=0        # 0: 50
                  # 1: 100

# << SELECT >>
flag_resample=0    # 0: 5 year block means
                   # 1: 10 year block means

# << SELECT >>
flag_var=0   # 0: tasmax
                   
# << SELECT >>
flag_standardize=0;  # 0: no (standardization before input to PCA and projections)
                     # 1: yes, standardize 
                     
# << SELECT >>
flag_scale=0;         # 0: global
                      # 1: latitudinal
                      # 2: continental
                      # 3: ar6 regions
                      
# << SELECT >>
flag_lulcc=0      # 0: forest loss
                  # 1: crop expansion                      
                   
# << SELECT >>
flag_correlation=0;  # 0: no
                     # 1: yes
                     
# << SELECT >>
flag_resample=0    # 0: 5 year block means
                   # 1: 10 year block means                     
                  
# redefine some flags for luh2 testing
# if flag_analysis == 1:
    
#     flag_y1=0
#     flag_len=1

# << SELECT >>
trunc=0


deforest_options = ['all',
                    'defor',
                    'ar6']
lulcc = ['treeFrac',
         'cropFrac']
lulcc_stats = ['mean',
               'max']
measures = ['absolute_change',
            'area_change',
            'all_pixels']
lu_techniques = ['individual',
                 'mean']
# measures = ['relative_change',
#             'absolute_change',
#             'area_change',
#             'all_pixels']
start_years = [1915,
               1965]
lengths = [50,
           100]
resample=['5Y',
          '10Y']
variables = ['tasmax']
analyses = ['models',
            'luh2']
landcover_types = ['forest',
                   'crops']
standardize_opts = ['no',
                    'yes']
correlation_opts = ['no',
                    'yes']
scale_opts = ['global',
              'latitudinal',
              'continental',
              'ar6']
resample=['5Y',
          '10Y']

y1 = start_years[flag_y1]
length = lengths[flag_len]
var = variables[flag_var]
lulcc_type = lulcc[flag_lulcc]
measure = measures[flag_lulcc_measure]
lu_techn = lu_techniques[flag_lu_technique]
standardize = standardize_opts[flag_standardize]
correlation = correlation_opts[flag_correlation]
scale = scale_opts[flag_scale]
freq = resample[flag_resample]
stat = lulcc_stats[flag_lulcc_stat]


# temporal extent of analysis data
strt_dt = str(y1) + '01'
y2 = y1+length-1
end_dt = str(y2) + '12'
t_ext = strt_dt+'-'+end_dt

models = ['CanESM5',
          'CNRM-ESM2-1',
          'IPSL-CM6A-LR',
          'UKESM1-0-LL']

exps_start = ['historical',
              'hist-noLu']

exps = ['historical',
        'hist-noLu',
        'lu']

continents = {}
continents['North America'] = [1,2,3,4,5,6,7]
continents['South America'] = [9,10,11,12,13,14,15]
continents['Europe'] = [16,17,18,19]
continents['Asia'] = [28,29,30,31,32,33,34,35,37,38]
continents['Africa'] = [21,22,23,24,25,26]
continents['Australia'] = [39,40,41,42]

ns = 0
continent_names = []
for c in continents.keys():
    continent_names.append(c)
    for i in continents[c]:
        ns += 1

labels = {}
labels['North America'] = ['WNA','CNA','ENA','NCA','SCA']
labels['South America'] = ['NWS','NSA','NES','SAM','SES']
labels['Europe'] = ['NEU','WCE','EEU','MED']
labels['Asia'] = ['WSB','ESB','WCA','EAS','SAS']
labels['Africa'] = ['WAF','CAF','NEAF','SEAF','SWAF','ESAF']
labels['Australia'] = ['NAU','CAU','EAU','SAU']

lat_ranges = {}
lat_ranges['boreal'] = slice(51.5,89.5)
lat_ranges['tropics'] = slice(-23.5,23.5)
lat_ranges['temperate_south'] = slice(-50.5,-24.5)
lat_ranges['temperate_north'] = slice(24.5,50.5)


#================================================================================
# mod + obs + luh2 data 
#================================================================================

#%%============================================================================
from pca_sr_file_alloc import *
map_files,grid_files,mod_files,pi_files = file_subroutine(mapDIR,
                                                          modDIR,
                                                          piDIR,
                                                          stat,
                                                          lulcc,
                                                          y1,
                                                          y2,
                                                          t_ext,
                                                          models,
                                                          exps,
                                                          var)

#%%============================================================================

# luh2 maps and ar6 regions
os.chdir(curDIR)
from pca_sr_maps import *
maps,ar6_regs,ar6_land = map_subroutine(map_files,
                                        models,
                                        mapDIR,
                                        lulcc,
                                        y1,
                                        measure,
                                        freq)            

#%%============================================================================

# mod ensembles
os.chdir(curDIR)
from pca_sr_mod_ens import *
mod_ens = ensemble_subroutine(modDIR,
                              models,
                              exps,
                              mod_files,
                              var,
                              lu_techn,
                              y1,
                              freq,
                              ar6_land)

#%%============================================================================

# pi data
os.chdir(curDIR)
from pca_sr_pi import *
pi_data = picontrol_subroutine(piDIR,
                               pi_files,
                               models,
                               var,
                               y1,
                               freq,
                               ar6_land)
           
#%%============================================================================

# pi data
os.chdir(curDIR)
from pca_sr_eof_proj import *
solver_dict,eof_dict,principal_components,pseudo_principal_components = pca_subroutine(lulcc,
                                                                                       models,
                                                                                       maps,
                                                                                       mod_ens,
                                                                                       pi_data,
                                                                                       continents,
                                                                                       lat_ranges,
                                                                                       ar6_regs,
                                                                                       scale)

#%%============================================================================