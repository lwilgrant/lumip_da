#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# This script generates detection and attribution results on LUMIP data


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
curDIR = 'C:/Users/lgrant/Documents/repos/lumip_da'
os.chdir(curDIR)

# data input directories
obsDIR = os.path.join(curDIR, 'obs')
modDIR = os.path.join(curDIR, 'mod')
piDIR = os.path.join(curDIR, 'pi')
allpiDIR = os.path.join(curDIR, 'allpi')
mapDIR = os.path.join(curDIR, 'map')
sfDIR = os.path.join(curDIR, 'shapefiles')
outDIR = os.path.join(curDIR, 'figures')

# bring in functions
from da_funcs import *

#%%============================================================================
# options - analysis
#==============================================================================

# adjust these flag settings for analysis choices only change '<< SELECT >>' lines

# << SELECT >>
flag_pickle=1     # 0: do not pickle objects
                  # 1: pickle objects after sections 'read' and 'analyze'

# << SELECT >>
flag_svplt=0      # 0: do not save plot
                  # 1: save plot in picDIR

# << SELECT >>
flag_analysis=1   # 0: d&a on global scale (all chosen ar6 regions)
                  # 1: d&a on continental scale (scaling factor per continent; continent represented by AR6 weighted means)
                  # 2: d&a on ar6 scale (scaling factor per ar6 region)
                  
# << SELECT >>
flag_lulcc=0      # 0: forest loss
                  # 1: crop expansion
                  
# << SELECT >>
flag_grid=0       # 0: model grid resolution
                  # 1: uniform obs grid resolution
                  
# << SELECT >>
flag_pi=1         # 0: only use pi from chosen models
                  # 1: use all available pi
                  
# << SELECT >>
flag_factor=0     # 0: 2-factor -> hist-noLu and lu
                  # 1: 1-factor -> historical
                  # 2: 1-factor -> hist-noLu
                  
# << SELECT >>
flag_obs=0       # 0: cru
                 # 1: berkley_earth
                  
# << SELECT >> 
thresh=-20       # flag_lulcc_measure == 0; threshold should be written as grid scale area fraction change of land cover type
                 # flag_lulcc_measure == 1; threshold should be written as area change of land cover type (scatter plots showed +/- 20 km^2 is best)
                 # flag_lulcc_measure == 2; doesn't mean anything if selecting all land pixels

# << SELECT >>
flag_lulcc_measure=2    # 0: absolute change
                        # 1: area change
                        # 2: all_pixels
                        
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
flag_bs=1         # 0: No bootstrapping of covariance matrix build
                  # 1: 50 (e.g. 50 reps of ROF, each with shuffled pichunks for Cov_matrices)
                  # 2: 100
                  # 3: 500
                  # 4: 1000

# << SELECT >>  # confidence intervals on scaling factors
ci_bnds = 0.95    # means  beta - 0.95 cummulative quantile and + 0.95 cummulative quantile, 
  
# << SELECT >> 
flag_reg=0        # 0: OLS
                  # 1: TLS

# << SELECT >>
flag_constest=0   # 0: OLS_AT99 
                  # 1: OLS_Corr
                  # 2: AS03 (TLS only)
                  # 3: MC (TLS only)

# << SELECT >> # confidence internval calculation in case that TLS regression chosen 
flag_ci_tls=0     # 0: AS03
                  # 1: ODP

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
lulcc = ['forest',
         'crops']
grids = ['model',
         'obs']
pi_opts = ['model',
           'allpi']
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
regressions = ['OLS',
               'TLS']
consistency_tests = ['OLS_AT99',
                     'OLS_Corr',
                     'AS03',
                     'MC']
tls_cis = ['AS03',
           'ODP']
shuffle_opts = ['no', 
                'yes']
bootstrap_reps = [0,50,100,500,1000]

analysis = analyses[flag_analysis]
lulcc_type = lulcc[flag_lulcc]
grid = grids[flag_grid]
pi = pi_opts[flag_pi]
exp_list = factors[flag_factor]
obs = obs_types[flag_obs]
measure = measures[flag_lulcc_measure]
lu_techn = lu_techniques[flag_lu_technique]
y1 = start_years[flag_y1]
length = lengths[flag_len]
freq = resample[flag_resample]
var = variables[flag_var]
bs_reps = bootstrap_reps[flag_bs]
reg = regressions[flag_reg]
cons_test = consistency_tests[flag_constest]
formule_ic_tls = tls_cis[flag_ci_tls]

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

#==============================================================================
# get data 
#==============================================================================

#%%============================================================================
from da_sr_file_alloc import *
map_files,grid_files,fp_files,pi_files,obs_files = file_subroutine(mapDIR,
                                                                   modDIR,
                                                                   piDIR,
                                                                   allpiDIR,
                                                                   obsDIR,
                                                                   grid,
                                                                   pi,
                                                                   obs_types,
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
from da_sr_maps import *
maps,ar6_regs,ar6_land = map_subroutine(map_files,
                                        models,
                                        mapDIR,
                                        lulcc,
                                        obs_types,
                                        grid,
                                        y1,
                                        measure,
                                        freq,
                                        thresh)            

#%%============================================================================

# mod ensembles
os.chdir(curDIR)
from da_sr_mod_ens import *
mod_ens,mod_ts_ens,nt = ensemble_subroutine(modDIR,
                                            maps,
                                            models,
                                            exps,
                                            var,
                                            lu_techn,
                                            measure,
                                            lulcc_type,
                                            y1,
                                            grid,
                                            freq,
                                            obs_types,
                                            continents,
                                            ns,
                                            fp_files,
                                            ar6_regs)
ts_pickler(curDIR,
           mod_ts_ens,
           grid,
           t_ext,
           obs_mod='mod')

#%%============================================================================

# mod fingerprint (nx is dummy var not used in OLS OF)
os.chdir(curDIR)
from da_sr_mod_fp import *
fp,fp_continental,fp_ar6,nx = fingerprint_subroutine(obs_types,
                                                     grid,
                                                     ns,
                                                     nt,
                                                     mod_ens,
                                                     exps,
                                                     models,
                                                     ar6_regs,
                                                     continents,
                                                     continent_names,
                                                     exp_list)

#%%============================================================================

# pi data
os.chdir(curDIR)
from da_sr_pi import *
ctl_data,ctl_data_continental,ctl_data_ar6,pi_ts_ens = picontrol_subroutine(piDIR,
                                                                            allpiDIR,
                                                                            pi_files,
                                                                            grid,
                                                                            pi,
                                                                            models,
                                                                            obs_types,
                                                                            continents,
                                                                            continent_names,
                                                                            var,
                                                                            y1,
                                                                            freq,
                                                                            maps,
                                                                            ar6_regs,
                                                                            ar6_land,
                                                                            ns,
                                                                            nt)

ts_pickler(curDIR,
           pi_ts_ens,
           grid,
           t_ext,
           obs_mod='pic')

#%%============================================================================

# obs data
os.chdir(curDIR)
from da_sr_obs import *
obs_data,obs_data_continental,obs_data_ar6,obs_ts = obs_subroutine(obsDIR,
                                                                   grid,
                                                                   obs_files,
                                                                   continents,
                                                                   continent_names,
                                                                   obs_types,
                                                                   models,
                                                                   y1,
                                                                   var,
                                                                   maps,
                                                                   ar6_regs,
                                                                   freq,
                                                                   nt,
                                                                   ns)

ts_pickler(curDIR,
           obs_ts,
           grid,
           t_ext,
           obs_mod='obs')

#%%============================================================================
# detection & attribution 
#==============================================================================

# optimal fingerprinting
os.chdir(curDIR)
from da_sr_of import *
var_sfs,\
var_ctlruns,\
proj,\
U,\
yc,\
Z1c,\
Z2c,\
Xc,\
Cf1,\
Ft,\
beta_hat,\
var_fin,\
models = of_subroutine(grid,
                       models,
                       nx,
                       analysis,
                       exp_list,
                       obs_types,
                       pi,
                       obs_data,
                       obs_data_continental,
                       obs_data_ar6,
                       fp,
                       fp_continental,
                       fp_ar6,
                       ctl_data,
                       ctl_data_continental,
                       ctl_data_ar6,
                       bs_reps,
                       ns,
                       nt,
                       reg,
                       cons_test,
                       formule_ic_tls,
                       trunc,
                       ci_bnds,
                       continents)

# save OF results
pickler(curDIR,
        var_fin,
        analysis,
        grid,
        t_ext,
        exp_list,
        pi)
           
#%%============================================================================
# plotting scaling factors
#==============================================================================    

os.chdir(curDIR)    
if len(exp_list) == 2:
    
    pass

elif len(exp_list) == 1:
    
    start_exp = deepcopy(exp_list[0])
    if start_exp == 'historical':
        second_exp = 'hist-noLu'
    elif start_exp == 'hist-noLu':
        second_exp = 'historical'
    pkl_file = open('var_fin_1-factor_{}_{}-grid_{}_{}.pkl'.format(second_exp,grid,analysis,t_ext),'rb')
    var_fin_2 = pk.load(pkl_file)
    pkl_file.close()
    
    for obs in obs_types:
        for mod in models:
            var_fin[obs][mod][second_exp] = var_fin_2[obs][mod].pop(second_exp)
            
    exp_list = ['historical', 'hist-noLu']

if analysis == 'global':
    
    plot_scaling_global(models,
                        grid,
                        obs_types,
                        pi,
                        exp_list,
                        var_fin,
                        flag_svplt,
                        outDIR)

elif analysis == 'continental':
    
    mod_ts_pkl = open('mod_ts_model-grid_196501-201412.pkl','rb')
    mod_ts = pk.load(mod_ts_pkl)
    mod_ts_pkl.close()
    
    obs_ts_pkl = open('obs_ts_model-grid_196501-201412.pkl','rb')
    obs_ts = pk.load(obs_ts_pkl)
    obs_ts_pkl.close()
    
    var_fin_pkl = open('var_fin_2-factor_model-grid_continental_196501-201412.pkl','rb')
    var_fin = pk.load(var_fin_pkl)
    var_fin_pkl.close()
    
    exps_2f = ['hist-noLu','lu']
    
    plot_scaling_continental(models,
                             exps_2f,
                             var_fin,
                             continents,
                             continent_names,
                             mod_ts,
                             obs_ts,
                             flag_svplt,
                             outDIR,
                             lulcc_type,
                             t_ext,
                             freq,
                             measure,
                             var,
                             obs_types)
    
    plot_scaling_map_continental(sfDIR,
                                 obs_types,
                                 pi,
                                 models,
                                 exp_list,
                                 continents,
                                 var_fin,
                                 grid,
                                 letters,
                                 outDIR)

elif analysis == 'ar6':
    
    plot_scaling_map_ar6(sfDIR,
                         obs_types,
                         pi,
                         models,
                         exp_list,
                         continents,
                         var_fin,
                         grid,
                         letters,
                         outDIR)              
    
    
         
    
    

# %%
