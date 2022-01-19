#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# This script generates correlation analysis results on model LU and LC transition data

# maps of trends
    # marked by significance
    # also at ar6 scale (and marked at centroids)
    # control map of mean pi trends with significance marking?
        # would help answer, can we have spurious trends of this significance with only noise?

# maps of correlation at pixel and ar6 scale (question over corr of mean series or mean of corr at pixel scale)
        
# use pixel scale trend info from pic and lu to compare their distributions
    # do this where correlation is good?
    # latitudinally? use trendist stuff, ar6?

#%%============================================================================
# import
# =============================================================================

import sys
import os
import numpy as np
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import copy as cp
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
import regionmask as rm
from random import shuffle
from matplotlib.lines import Line2D

#%%============================================================================
# path
#==============================================================================

# curDIR = '/home/luke/documents/lumip/d_a/'
# curDIR = '/theia/data/brussel/vo/000/bvo00012/vsc10116/lumip/d_a'
# curDIR = '/Users/Luke/Documents/PHD/lumip/da'
curDIR = r'C:/Users/lgrant/Documents/repos/lumip_da'
os.chdir(curDIR)

# data input directories
modDIR = os.path.join(curDIR, 'mod')
piDIR = os.path.join(curDIR, 'pi')
mapDIR = os.path.join(curDIR, 'map')
sfDIR = os.path.join(curDIR, 'shapefiles')
outDIR = os.path.join(curDIR, 'figures')

# bring in functions
from lu_funcs import *

#%%============================================================================
# options - analysis
#==============================================================================

# adjust these flag settings for analysis choices only change '<< SELECT >>' lines

# << SELECT >>
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
trunc=0

seasons = ['jja',
           'djf',
           'annual',
           'max']
analyses = ['global',
            'continental',
            'ar6']
deforest_options = ['all',
                    'defor',
                    'ar6']
lulcc = ['treeFrac',
         'cropFrac']
lulcc_stats = ['mean',
               'max']
grids = ['model',
         'obs']
factors = [['hist-noLu','lu'],
           ['historical'],
           ['hist-noLu']]
obs_types = ['cru',
             'berkley_earth']
measures = ['absolute_change',
            'area_change',
            'all_pixels']
lu_techniques = ['individual',
                 'mean']
start_years = [1915,
               1965]
lengths = [50,
           100]
resample=['5Y',
          '10Y']
variables = ['tasmax']

stat = lulcc_stats[flag_lulcc_stat]
measure = measures[flag_lulcc_measure]
lu_techn = lu_techniques[flag_lu_technique]
y1 = start_years[flag_y1]
length = lengths[flag_len]
freq = resample[flag_resample]
var = variables[flag_var]

# temporal extent of analysis data
strt_dt = str(y1) + '01'
y2 = y1+length-1
end_dt = str(y2) + '12'
t_ext = strt_dt+'-'+end_dt

models = ['CanESM5',
          'CNRM-ESM2-1',
          'IPSL-CM6A-LR',
          'UKESM1-0-LL']

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

continent_names = []
for c in continents.keys():
    continent_names.append(c)

labels = {}
labels['North America'] = ['WNA','CNA','ENA','SCA']
labels['South America'] = ['NWS','NSA','NES','SAM','SWS','SES']
labels['Europe'] = ['NEU','WCE','EEU','MED']
labels['Asia'] = ['WSB','ESB','TIB','EAS','SAS','SEA']
labels['Africa'] = ['WAF','CAF','NEAF','SEAF','ESAF']
    
ns = 0
for c in continents.keys():
    for i in continents[c]:
        ns += 1
        
letters = ['a', 'b', 'c',
           'd', 'e', 'f',
           'g', 'h', 'i',
           'j', 'k', 'l',
           'm', 'n', 'o',
           'p', 'q', 'r',
           's', 't', 'u',
           'v', 'w', 'x',
           'y', 'z']

# boundaries of 0 for lu
null_bnds_lu = [-0.0050,
                0.0050]

# boundaries of 0 for lc
null_bnds_lc = [-0.10,
                0.10]

#==============================================================================
# get data 
#==============================================================================

#%%============================================================================
from lu_sr_file_alloc import *
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
from lu_sr_maps import *
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
from lu_sr_mod_ens import *
mod_ens = ensemble_subroutine(modDIR,
                              models,
                              exps,
                              var,
                              lu_techn,
                              y1,
                              freq,
                              mod_files)

#%%============================================================================

# pi data
os.chdir(curDIR)
from lu_sr_pi import *
pi_data = picontrol_subroutine(piDIR,
                               pi_files,
                               models,
                               var,
                               y1,
                               freq,
                               maps)
           
#%%============================================================================

# trends and lu/lc corr
os.chdir(curDIR)
stats_ds = stats_subroutine(models,
                            ar6_land,
                            lulcc,
                            mod_ens,
                            maps)

#%%============================================================================

# global mean computations
glm_ds = glm_subroutine(models,
                        exps,
                        ar6_land,
                        mod_ens,
                        maps,
                        pi_data,
                        lulcc)

#%%============================================================================
   
# plot global mean timeseries
lineplot(glm_ds,
         models,
         exps,
         letters,
         t_ext,
         flag_svplt,
         outDIR)
        
#%%============================================================================

# plot model trends for lu response and land cover transition
trends_plot(stats_ds,
            models,
            lulcc,
            letters,
            null_bnds_lc,
            null_bnds_lu,
            t_ext,
            flag_svplt,
            outDIR)
    
#%%============================================================================

# plot correlation between model lu response and land cover transition
corr_plot(stats_ds,
          models,
          lulcc,
          letters,
          null_bnds_lc,
          t_ext,
          flag_svplt,
          outDIR)

#%%============================================================================

# plot figures combined
combined_plot(
    stats_ds,
    models,
    letters,
    lulcc,
    null_bnds_lc,
    null_bnds_lu,
    t_ext,
    flag_svplt,
    outDIR)
# %%
