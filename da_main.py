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

# New hydra version:
    # chop into cells (main script and functions) (done)
    # chop into subroutines on other scripts (done)
    # for unmasked, use all land-based ar6 regions per continent (too much uncertainty for luh2 selection) (done)
    # add option for uniform grids (per obs)
        # if statements before list comprehension to gather files
        # BETTER IDEA:
            # subroutine for all file extraction:
                # use flags as input; return things such as pi files, obs files and fp_files
                # use directories as input
                # currently, maps subroutine generates ar6_land maps to select grid cells
                    # but I no longer want to do that. 
                    # change maps subroutine to, based on grid_type, produce ar6_land masks either at obs resolution or as dict for mod resolutions
                    # take grid type as input to file allocation subroutine
                    # tres can be removed: no longer required
                    # based on need, can read in either ensmeans or individual realisations
                # no option exists for looking at area changes in the case of working at obs resolutions (map/*.nc files are all at mod resolution; will have to fix if I want to do this but not necessary now)
    # add option for obs type; needs to be added to subroutine functions
    # add d & a outputs per AR6 region, (latitudinally?)
    # put current (sep 30) fp_main and da_main and funcs scripts on backup branch on github
    # need solution for options in sr_mod_fp and sr_pi and sr_obs to:
        # run fp on one experiment; e.g. separate runs for historical and hist-nolu (for "sr_mod_fp")
        # can a function take fp_data_* objects and 
    # will always have 2 OF results for each obs type, but difference will be whether at model or obs grid
    # first establish working results for obs vs mod res, global vs continental vs ar6 results,
        # then establish historical vs hist-nolu single factor runs

#%%============================================================================
# import
# =============================================================================

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

#%%============================================================================
# path
#==============================================================================

curDIR = '/theia/data/brussel/vo/000/bvo00012/vsc10116/lumip/d_a'
os.chdir(curDIR)

# data input directories
obsDIR = os.path.join(curDIR, 'obs')
modDIR = os.path.join(curDIR, 'mod')
piDIR = os.path.join(curDIR, 'pi')
mapDIR = os.path.join(curDIR, 'map')
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
flag_svplt=1      # 0: do not save plot
                  # 1: save plot in picDIR

# << SELECT >>
flag_analysis=0   # 0: d&a on global scale (all chosen ar6 regions)
                  # 1: d&a on continental scale (scaling factor per continent; continent represented by AR6 weighted means)
                  # 2: d&a on ar6 scale (scaling factor per ar6 region)
                  
# << SELECT >>
flag_lulcc=0      # 0: forest loss
                  # 1: crop expansion
                  
# << SELECT >>
flag_grid=0       # 0: model grid resolution
                  # 1: uniform obs grid resolution
                  
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
flag_ci_bnds=1    # 0: 10-90
                  # 1: 5-95
                  # 2: 2.5-97.5
                  # 3: 0.5-99.5
  
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
confidence_intervals = [0.8,0.9,0.95,0.99]

analysis = analyses[flag_analysis]
lulcc_type = lulcc[flag_lulcc]
grid = grids[flag_grid]
exp_list = factors[flag_factor]
obs = obs_types[flag_obs]
measure = measures[flag_lulcc_measure]
lu_techn = lu_techniques[flag_lu_technique]
y1 = start_years[flag_y1]
length = lengths[flag_len]
freq = resample[flag_resample]
var = variables[flag_var]
bs_reps = bootstrap_reps[flag_bs]
ci_bnds = confidence_intervals[flag_ci_bnds]
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
continents['Asia'] = [29,30,32,33,34,35,37,38]
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

#==============================================================================
# get data 
#==============================================================================

#%%============================================================================
from da_sr_file_alloc import *
map_files,grid_files,fp_files,pi_files,obs_files = file_subroutine(mapDIR,
                                                                   modDIR,
                                                                   piDIR,
                                                                   obsDIR,
                                                                   grid,
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
ctl_data,ctl_data_continental,ctl_data_ar6 = picontrol_subroutine(piDIR,
                                                                  pi_files,
                                                                  grid,
                                                                  models,
                                                                  obs_types,
                                                                  continents,
                                                                  continent_names,
                                                                  var,
                                                                  y1,
                                                                  freq,
                                                                  maps,
                                                                  ar6_regs,
                                                                  ns,
                                                                  nt)

#%%============================================================================

# obs data
from da_sr_obs import *
obs_data,obs_data_continental,obs_ts = obs_subroutine(obsDIR,
                                                      continents,
                                                      continent_names,
                                                      models,
                                                      var,
                                                      tres,
                                                      t_ext,
                                                      freq,
                                                      maps,
                                                      nt,
                                                      ns)


#%%============================================================================
# detection & attribution 
#==============================================================================

# optimal fingerprinting
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
var_fin = of_subroutine(models,
                        nx,
                        analysis,
                        exps,
                        obs_data,
                        obs_data_continental,
                        fp,
                        fp_continental,
                        ctl_data,
                        ctl_data_continental,
                        bs_reps,
                        nt,
                        reg,
                        cons_test,
                        formule_ic_tls,
                        trunc,
                        ci_bnds,
                        continents)
           
#%%============================================================================
# plotting scaling factors
#==============================================================================    
    
if analysis == 'global':
    
    plot_scaling_global(models,
                        exps,
                        var_fin,
                        flag_svplt,
                        outDIR,
                        lulcc_type,
                        t_ext,
                        tres,
                        freq,
                        var,
                        measure)

elif analysis == 'continental':
    
    plot_scaling_continental(models,
                             exps,
                             var_fin,
                             continents,
                             continent_names,
                             mod_ts_ens,
                             obs_ts,
                             flag_svplt,
                             outDIR,
                             lulcc_type,
                             t_ext,
                             tres,
                             freq,
                             measure,
                             var)

                  
    
    
         
    
    
