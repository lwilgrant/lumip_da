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
    # chop into cells (main script and functions)
    # chop into subroutines on other scripts
    # for unmasked, use all land-based ar6 regions per continent (too much uncertainty for luh2 selection)
    # add option for uniform grids (per obs)
    # add d & a outputs per AR6 region, (latitudinally?)
    # put current (sep 30) fp_main and da_main and funcs scripts on backup branch on github


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

# # data input directories
# obsDIR = os.path.join(curDIR, 'data/obs/final')
# modDIR = os.path.join(curDIR, 'data/mod/final')
# piDIR = os.path.join(curDIR, 'data/pi/final')
# mapDIR = os.path.join(curDIR, 'data/map/final')
# outDIR = os.path.join(curDIR, 'figures_v2')

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
flag_svplt=0;     # 0: do not save plot
                  # 1: save plot in picDIR

# << SELECT >>
flag_tres=3;    # 0: jja
                # 1: djf
                # 2: annual
                # 3: max month

# << SELECT >>
flag_analysis=1;  # 0: d&a on global scale (all chosen ar6 regions)
                  # 1: d&a on continental scale (scaling factor per continent; continent represented by AR6 weighted means)
                  
# << SELECT >>
flag_lulcc=0;     # 0: forest loss
                  # 1: crop expansion
                  # 2: urban
                  
# << SELECT >>
thresh=-20

# << SELECT >>
flag_lulcc_measure=3;   # 0: relative change
                        # 1: absolute change
                        # 2: area change
                        # 3: all_pixels

# << SELECT >>
flag_y1=1;        # 0: 1915
                  # 1: 1965

# << SELECT >>
flag_len=0;        # 0: 50
                   # 1: 100

# << SELECT >>
flag_resample=0;   # 0: 5 year block means
                   # 1: 10 year block means

# << SELECT >>
flag_var=0;  # 0: tasmax

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
flag_reg=0;       # 0: OLS
                  # 1: TLS

# << SELECT >>
flag_constest=0;  # 0: OLS_AT99 
                  # 1: OLS_Corr
                  # 2: AS03 (TLS only)
                  # 3: MC (TLS only)

# << SELECT >> # confidence internval calculation in case that TLS regression chosen 
flag_ci_tls=0;    # 0: AS03
                  # 1: ODP

# << SELECT >>
trunc=0

seasons = ['jja',
           'djf',
           'annual',
           'max']
analyses = ['global',
            'continental']
deforest_options = ['all',
                    'defor',
                    'ar6']
lulcc = ['forest',
         'crops',
         'urban']
measures = ['relative_change',
            'absolute_change',
            'area_change',
            'all_pixels']
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

tres = seasons[flag_tres]
analysis = analyses[flag_analysis]
lulcc_type = lulcc[flag_lulcc]
measure = measures[flag_lulcc_measure]
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

exps_start = ['historical',
              'hist-noLu']

exps = ['hist-noLu',
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


#%%============================================================================
# get data 
#==============================================================================


#==============================================================================

# map data
maps_files = {}
maps = {}
gridarea = {}
ar6_regs = {}
ar6_land = {}

for mod in models:
    
    maps_files[mod] = {}
    maps[mod] = {}

    # maps of lulcc data
    os.chdir(mapDIR)
    gridarea[mod] = xr.open_dataset(mod+'_gridarea.nc',decode_times=False)['cell_area']
    
    i = 0
    for lu in lulcc:
        
        for file in [file for file in sorted(os.listdir(mapDIR))
                     if mod in file
                     and lu in file
                     and str(y1) in file
                     and 'absolute_change' in file]:

                maps_files[mod][lu] = file
                
                # get ar6 from lat/lon of sample model res luh2 file
                if i == 0:
                    template = xr.open_dataset(file,decode_times=False).cell_area.isel(time=0).squeeze(drop=True)
                    if 'height' in template.coords:
                        template = template.drop('height')
                    ar6_regs[mod] = ar6_mask(template)
                    ar6_land[mod] = xr.where(ar6_regs[mod]>=0,1,0)
                i += 1
            

        if measure == 'absolute_change':
            
            maps[mod][lu] = nc_read(maps_files[mod][lu],
                                        y1,
                                        var='cell_area',
                                        mod=True,
                                        freq=freq)
            
        elif measure == 'area_change':
            
            maps[mod][lu] = nc_read(maps_files[mod][lu],
                                        y1,
                                        var='cell_area',
                                        mod=True,
                                        freq=freq) * gridarea[mod] / 1e6
            if thresh < 0: # forest
            
                #da = da.where(da <= thresh)
                da = xr.where(maps[mod][lu] <= thresh,1,0).sum(dim='time')
                maps[mod][lu] = xr.where(da >= 1,1,0)
            elif thresh > 0: # crops + urban
            
                da = xr.where(maps[mod][lu] >= thresh,1,0).sum(dim='time')
                maps[mod][lu] = xr.where(da >= 1,1,0)
                
        elif measure == 'all_pixels':
            
            maps[mod][lu] = ar6_land[mod]
            
            
            
        

#%%============================================================================

# mod data
os.chdir(modDIR)

fp_files = {}
map_files = {}
mod_data = {}
mod_ens = {}
mod_ens['mmm'] = {}
mod_ts_ens = {}
mod_ts_ens['mmm'] = {}
for exp in exps:
    mod_ts_ens['mmm'][exp] = []
    

# individual model ensembles and lu extraction
for mod in models:
    
    fp_files[mod] = {}
    mod_data[mod] = {}
    mod_ens[mod] = {}
    
    for exp in exps_start:
        
        fp_files[mod][exp] = []
        mod_data[mod][exp] = []
        
        for file in [file for file in sorted(os.listdir(modDIR))\
                     if var in file\
                         and mod in file\
                         and exp in file\
                         and tres in file\
                         and 'unmasked' in file\
                         and t_ext in file]:
            
            fp_files[mod][exp].append(file)
            
        for file in fp_files[mod][exp]:
        
            # mod data and coords for ar6 mask
            da = nc_read(file,
                         y1,
                         var,
                         mod=True,
                         freq=freq)
            
            mod_data[mod][exp].append(da.where(maps[mod][lulcc_type] == 1))
            
        mod_ens[mod][exp] = da_ensembler(mod_data[mod][exp])
    
    mod_ens[mod]['lu'] = mod_ens[mod]['historical'] - mod_ens[mod]['hist-noLu']
    
    mod_ts_ens[mod] = {}
    
    for exp in exps:
        
        fp_files[mod][exp] = []
        mod_ts_ens[mod][exp] = []
    
        for file in [file for file in sorted(os.listdir(modDIR))\
                     if var in file\
                         and mod in file\
                         and exp in file\
                         and tres in file\
                         and 'unmasked' in file\
                         and t_ext in file]:
            
            fp_files[mod][exp].append(file)
            
        for file in fp_files[mod][exp]:
        
            # mod data and coords for ar6 mask
            da = nc_read(file,
                         y1,
                         var,
                         mod=True,
                         freq=freq)
            
            # weighted mean
            nt = len(da.time.values)
            matrix = np.zeros(shape=(nt,ns))
            mod_ar6 = weighted_mean(continents,
                                    da.where(maps[mod][lulcc_type] == 1),
                                    ar6_regs[mod],
                                    nt,
                                    matrix)
                    
            # remove tsteps with nans (temporal x spatial shaped matrix)
            del_rows = []
            
            for i,row in enumerate(mod_ar6):
                
                nans = np.isnan(row)
                
                if True in nans:
                    
                    del_rows.append(i)
                    
            mod_ar6 = np.delete(mod_ar6,
                                del_rows,
                                axis=0)
                        
            # temporal centering
            mod_ar6_center = temp_center(ns,
                                         mod_ar6)
            mod_ts_ens[mod][exp].append(mod_ar6_center)
            mod_ts_ens['mmm'][exp].append(mod_ar6_center)
            
    
        mod_ts_ens[mod][exp] = np.stack(mod_ts_ens[mod][exp],axis=0)

for exp in exps:        
    mod_ts_ens['mmm'][exp] = np.stack(mod_ts_ens['mmm'][exp],axis=0)
    
# ns for spatial dimension (i.e. s AR6 regions) and nt for temporal
ns = 0
for c in continents.keys():
    for i in continents[c]:
        ns += 1
nt = len(mod_ens[mod]['lu'].time.values)
matrix = np.zeros(shape=(nt,ns))

# fingerprint dictionaries
fp_data = {}
fp = {}

fp_data_continental = {}
fp_continental = {}

nx = {}
mod_ts = {}

# lists for ensemble of mod_ar6 tstep x region arrays and continental slices
mmm = {}
mmm_c = {}

for exp in exps:
    
    mmm[exp] = []
    mmm_c[exp] = {}
    
    for c in continents.keys():
        mmm_c[exp][c] = []

# models + mmm fingerprints
for mod in models:
    
    fp_data[mod] = {}
    fp[mod] = {}
    
    fp_data_continental[mod] = {}
    fp_continental[mod] = {}
    
    nx[mod] = {}
    mod_ts[mod] = {}
    
    for exp in exps:
        
        # weighted mean
        mod_ar6 = weighted_mean(continents,
                                mod_ens[mod][exp],
                                ar6_regs[mod],
                                nt,
                                matrix)
                
        # remove tsteps with nans (temporal x spatial shaped matrix)
        del_rows = []
        for i,row in enumerate(mod_ar6):
            
            nans = np.isnan(row)
            
            if True in nans:
                
                del_rows.append(i)
                
        mod_ar6 = np.delete(mod_ar6,
                            del_rows,
                            axis=0)
        
        # set aside mod_ar6 for mmm
        mmm[exp].append(mod_ar6)
                    
        # temporal centering
        mod_ar6_center = temp_center(ns,
                                     mod_ar6)
        mod_ts[mod][exp] = mod_ar6_center

        # 1-D mod array to go into  DA
        fp_data[mod][exp] = mod_ar6_center.flatten()
        
        # 1-D mod arrays per continent for continental DA
        fp_data_continental[mod][exp] = {}
        for c in continents.keys():
            
            cnt_idx = continent_names.index(c)
            
            if cnt_idx == 0:
                
                strt_idx = 0
                
            else:
                
                strt_idx = 0
                idxs = np.arange(0,cnt_idx)
                
                for i in idxs:
                    
                    strt_idx += len(continents[continent_names[i]])
                    
            n = len(continents[c])
            mmm_c[exp][c].append(mod_ar6[:,strt_idx:strt_idx+n])
            fp_data_continental[mod][exp][c] = mod_ar6_center[:,strt_idx:strt_idx+n].flatten()
        
    fp[mod] = np.stack([fp_data[mod]['hist-noLu'],
                        fp_data[mod]['lu']],
                       axis=0)
    
    for c in continents.keys():
        
        fp_continental[mod][c] = np.stack([fp_data_continental[mod]['hist-noLu'][c],
                                           fp_data_continental[mod]['lu'][c]],
                                          axis=0)
    

# mmm of individual model means for global + continental
mod_ts['mmm'] = {}
fp_data['mmm'] = {}
fp_data_continental['mmm'] = {}

for exp in exps:
    
    ens,_ = ensembler(mmm[exp],
                      ax=0)
    ens_center = temp_center(ns,
                             ens)
    mod_ts['mmm'][exp] = ens
    fp_data['mmm'][exp] = ens_center.flatten()
    fp_data_continental['mmm'][exp] = {}
    
    for c in continents.keys():
        
        n = len(continents[c])
        ens,_ = ensembler(mmm_c[exp][c],
                          ax=0)
        ens_center = temp_center(n,
                                 ens)
        fp_data_continental['mmm'][exp][c] = ens_center.flatten()

# mmm fingerprints
fp['mmm'] = np.stack([fp_data['mmm']['hist-noLu'],
                     fp_data['mmm']['lu']],
                     axis=0)
fp_continental['mmm'] = {}

for c in continents.keys():
    
    fp_continental['mmm'][c] = np.stack([fp_data_continental['mmm']['hist-noLu'][c],
                                        fp_data_continental['mmm']['lu'][c]],
                                        axis=0)
    
# TEST ALL FINGERPRINTS TO MAKE SURE THEY ARE TEMPORALLY CENTERED!!!! (FP AND FP_CONTINENTAL)

#%%============================================================================

# pi data
os.chdir(piDIR)
pi_files = {}
pi_data = {}
pi_data_continental = {}
ctl_data = {}
ctl_data_continental = {}

for mod in models:
    
    pi_files[mod] = []
    pi_data[mod] = []
    pi_data_continental[mod] = {}
    ctl_data_continental[mod] = {}
    
    for c in continents.keys():
        pi_data_continental[mod][c] = []
    
    for file in [file for file in sorted(os.listdir(piDIR))\
                 if var in file\
                     and mod in file\
                         and tres in file\
                             and 'unmasked' in file\
                                 and t_ext in file]:
        
        pi_files[mod].append(file)
    
    shuffle(pi_files[mod])
    
    for file in pi_files[mod]:
        
        # mod data and coords for ar6 mask
        da = nc_read(file,
                     y1,
                     var,
                     mod=True,
                     freq=freq).where(maps[mod][lulcc_type] == 1)
        
        # ns for spatial dimension (i.e. s AR6 regions) and nt for temporal
        ns = 0
        for c in continents.keys():
            
            for i in continents[c]:
                
                ns += 1
        nt = len(da.time.values)
        pi_ar6 = np.zeros(shape=(nt,ns))
        
        # weighted mean
        pi_ar6 = weighted_mean(continents,
                               da,
                               ar6_regs[mod],
                               nt,
                               pi_ar6)
            
        # remove tsteps with nans (temporal x spatial shaped matrix)
        del_rows = []
        for i,row in enumerate(pi_ar6):
            
            nans = np.isnan(row)
            
            if True in nans:
                
                del_rows.append(i)
                
        pi_ar6 = np.delete(pi_ar6,
                           del_rows,
                           axis=0)
                
        # temporal centering
        pi_ar6 = temp_center(ns,
                             pi_ar6)
        
        # 1-D pi array to go into pi-chunks for DA
        pi_data[mod].append(pi_ar6.flatten())
        
        # 1-D pi array per continent
        for c in continents.keys():
            
            cnt_idx = continent_names.index(c)
            
            if cnt_idx == 0:
                
                strt_idx = 0
                
            else:
                
                strt_idx = 0
                idxs = np.arange(0,cnt_idx)
                
                for i in idxs:
                    
                    strt_idx += len(continents[continent_names[i]])
                    
            n = len(continents[c])
            pi_data_continental[mod][c].append(pi_ar6[:,strt_idx:strt_idx+n].flatten())
        
    ctl_data[mod] = np.stack(pi_data[mod],axis=0)
    
    for c in continents.keys():
        
        ctl_data_continental[mod][c] = np.stack(pi_data_continental[mod][c],axis=0)
        
# collect all pi data for mmm approach
ctl_list = []
for mod in models:
    
    ctl_list.append(ctl_data[mod])
    
ctl_data['mmm'] = np.concatenate(ctl_list,
                                 axis=0)
ctl_list_c = {}
ctl_data_continental['mmm'] = {}

for c in continents.keys():
    
    ctl_list_c[c] = []
    
    for mod in models:
        
        ctl_list_c[c].append(ctl_data_continental[mod][c])
        
    ctl_data_continental['mmm'][c] = np.concatenate(ctl_list_c[c],
                                                    axis=0)
        
#%%============================================================================

# obs data
os.chdir(obsDIR)
obs_files = {}
obs_data = {}
obs_data_continental = {}
obs_ts = {}
obs_mmm = []
obs_mmm_c = {}
for c in continents.keys():
    obs_mmm_c[c] = []

for mod in models:
    
    for file in [file for file in sorted(os.listdir(obsDIR))\
                 if var in file\
                     and mod in file\
                         and tres in file\
                             and 'unmasked' in file\
                                 and t_ext in file]:
        
        obs_files[mod] = file
        
        da = nc_read(obs_files[mod],
                     y1,
                     var,
                     mod=True,
                     freq=freq).where(maps[mod][lulcc_type] == 1)
        
        # ns for spatial dimension (i.e. s AR6 regions) and nt for temporal
        ns = 0
        for c in continents.keys():
            
            for i in continents[c]:
                
                ns += 1
                
        nt = len(da.time.values)
        obs_ar6 = np.zeros(shape=(nt,ns))
        
        # weighted mean
        obs_ar6 = weighted_mean(continents,
                                da,
                                ar6_regs[mod],
                                nt,
                                obs_ar6)
            
        # remove tsteps with nans (temporal_rows x spatial_cols shaped matrix)
        del_rows = []
        for i,row in enumerate(obs_ar6):
            
            nans = np.isnan(row)
            
            if True in nans:
                
                del_rows.append(i)
                
        obs_ar6 = np.delete(obs_ar6,
                            del_rows,
                            axis=0)

        obs_mmm.append(obs_ar6)
        
        # temporal centering
        obs_ar6_center = temp_center(ns,
                                     obs_ar6)
        nt=np.shape(obs_ar6)[0]
        
        obs_ts[mod] = obs_ar6_center
        obs_data[mod] = obs_ar6_center.flatten()
        obs_data_continental[mod] = {}
        
        for c in continents.keys():
            
            cnt_idx = continent_names.index(c)
            
            if cnt_idx == 0:
                
                strt_idx = 0
                
            else:
                
                strt_idx = 0
                idxs = np.arange(0,cnt_idx)
                
                for i in idxs:
                    strt_idx += len(continents[continent_names[i]])
                    
            n = len(continents[c])
            obs_mmm_c[c].append(obs_ar6[:,strt_idx:strt_idx+n])
            obs_data_continental[mod][c] = obs_ar6[:,strt_idx:strt_idx+n].flatten()

# mmm of individual obs series for global + continental
obs_ens,_ = ensembler(obs_mmm,
                      ax=0)
obs_ts['mmm'] = temp_center(ns,
                            obs_ens)
obs_data['mmm'] = temp_center(ns,
                              obs_ens).flatten()
obs_data_continental['mmm'] = {}

for c in continents.keys():
    
    obs_ens,_ = ensembler(obs_mmm_c[c],
                          ax=0)
    n = len(continents[c])
    obs_data_continental['mmm'][c] = temp_center(n,
                                                 obs_ens).flatten()
 

#%%============================================================================
# detection & attribution 
#==============================================================================


var_sfs = {}
bhi = {}
b = {}
blow = {}
pval = {}
var_fin = {}
var_xruns = {}
var_ctlruns = {}

# details check
U = {}
yc = {}
Z1c = {}
Z2c = {}
Xc = {}
Cf1 = {}
Ft = {}
beta_hat = {}

# now no diff between individual and mmm options (mmm built into data dictionaries)
models.append('mmm')   
nx['mmm'] = [] 
for mod in models:

    #==============================================================================
    
    # global analysis
    if analysis == "global":
        
        bhi[mod] = {}
        b[mod] = {}
        blow[mod] = {}
        pval[mod] = {}
        var_fin[mod] = {}
        
        for exp in exps:
            
            bhi[mod][exp] = []
            b[mod][exp] = []
            blow[mod][exp] = []
            pval[mod][exp] = []
        
        y = obs_data[mod]
        X = fp[mod]
        ctl = ctl_data[mod]
        nb_runs_x= nx[mod]
        
        if bs_reps == 0: # for no bs, run ROF once
        
            bs_reps += 1
        
        for i in np.arange(0,bs_reps):
            
            # shuffle rows of ctl
            ctl = np.take(ctl,
                          np.random.permutation(ctl.shape[0]),
                          axis=0)
            
            # run detection and attribution
            var_sfs[mod],\
            var_ctlruns[mod],\
            proj,\
            U[mod],\
            yc[mod],\
            Z1c[mod],\
            Z2c[mod],\
            Xc[mod],\
            Cf1[mod],\
            Ft[mod],\
            beta_hat[mod] = da_run(y,
                                   X,
                                   ctl,
                                   nb_runs_x,
                                   ns,
                                   nt,
                                   reg,
                                   cons_test,
                                   formule_ic_tls,
                                   trunc,
                                   ci_bnds)
            
            # [yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat]
            for i,exp in enumerate(exps):
                
                bhi[mod][exp].append(var_sfs[mod][2,i])
                b[mod][exp].append(var_sfs[mod][1,i])
                blow[mod][exp].append(var_sfs[mod][0,i])
                pval[mod][exp].append(var_sfs[mod][3,i])
        
        for exp in exps:
            
            bhi_med = np.median(bhi[mod][exp])
            b_med = np.median(b[mod][exp])
            blow_med = np.median(blow[mod][exp])
            pval_med = np.median(pval[mod][exp])
            var_fin[mod][exp] = [bhi_med,
                                 b_med,
                                 blow_med,
                                 pval_med]
    
    #==============================================================================
        
    # continental analysis
    elif analysis == 'continental':
        
        bhi[mod] = {}
        b[mod] = {}
        blow[mod] = {}
        pval[mod] = {}
        var_fin[mod] = {}
        var_sfs[mod] = {}
        var_ctlruns[mod] = {}
        
        U[mod] = {}
        yc[mod] = {}
        Z1c[mod] = {}
        Z2c[mod] = {}
        Xc[mod] = {}
        Cf1[mod] = {}
        Ft[mod] = {}
        beta_hat[mod] = {}
        
        for exp in exps:
            
            bhi[mod][exp] = {}
            b[mod][exp] = {}
            blow[mod][exp] = {}
            pval[mod][exp] = {}
            var_fin[mod][exp] = {}
            
            for c in continents.keys():
                
                bhi[mod][exp][c] = []
                b[mod][exp][c] = []
                blow[mod][exp][c] = []
                pval[mod][exp][c] = []
            
        for c in continents.keys():
        
            y = obs_data_continental[mod][c]
            X = fp_continental[mod][c]
            ctl = ctl_data_continental[mod][c]
            nb_runs_x= nx[mod]
            ns = len(continents[c])
        
            if bs_reps == 0: # for no bs, run ROF once
            
                bs_reps += 1
            
            for i in np.arange(0,bs_reps):
                
                # shuffle rows of ctl
                ctl = np.take(ctl,
                              np.random.permutation(ctl.shape[0]),
                              axis=0)
                
                # run detection and attribution
                var_sfs[mod][c],\
                var_ctlruns[mod][c],\
                proj,\
                U[mod][c],\
                yc[mod][c],\
                Z1c[mod][c],\
                Z2c[mod][c],\
                Xc[mod][c],\
                Cf1[mod][c],\
                Ft[mod][c],\
                beta_hat[mod][c] = da_run(y,
                                          X,
                                          ctl,
                                          nb_runs_x,
                                          ns,
                                          nt,
                                          reg,
                                          cons_test,
                                          formule_ic_tls,
                                          trunc,
                                          ci_bnds)
                
                # [yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat]
                for i,exp in enumerate(exps):
                    
                    bhi[mod][exp][c].append(var_sfs[mod][c][2,i])
                    b[mod][exp][c].append(var_sfs[mod][c][1,i])
                    blow[mod][exp][c].append(var_sfs[mod][c][0,i])
                    pval[mod][exp][c].append(var_sfs[mod][c][3,i])
            
            for exp in exps:
                for c in continents.keys():
                    
                    bhi_med = np.median(bhi[mod][exp][c])
                    b_med = np.median(b[mod][exp][c])
                    blow_med = np.median(blow[mod][exp][c])
                    pval_med = np.median(pval[mod][exp][c])
                    var_fin[mod][exp][c] = [bhi_med,
                                            b_med,
                                            blow_med,
                                            pval_med]
                
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

                  
    
    
         
    
    
