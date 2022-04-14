#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# This script uses detection and attribution input vectors to assess model
# significance relative to internal variability (uses mod and pic input pkl files)


#%%============================================================================
# import
# =============================================================================

import sys
import os
import numpy as np
import scipy.stats as scp
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

# curDIR = '/theia/data/brussel/vo/000/bvo00012/vsc10116/lumip/d_a'
curDIR = r'C:\Users\adm_lgrant\Documents\repos\lumip_da'
os.chdir(curDIR)

# data input directories
pklDIR = os.path.join(curDIR, 'pickle')
sfDIR = os.path.join(curDIR, 'shapefiles')
outDIR = os.path.join(curDIR, 'figures')

# bring in functions
from icv_funcs import *

#%%============================================================================
# options - analysis
#==============================================================================

# adjust these flag settings for analysis choices only change '<< SELECT >>' lines

# << SELECT >>
flag_svplt=1      # 0: do not save plot
                  # 1: save plot in picDIR
                  
# << SELECT >>
flag_data_agg=1   # 0: global d&a (via flag_analysis) w/ ar6 scale input points (want this for continental scale via flag_analysis)
                  # 1: global d&a w/ continental scale input points             
                             
# << SELECT >>
flag_grid=0       # 0: model grid resolution (decided on always using this; many options don't work otherwise)
                  # 1: uniform obs grid resolution
                  
# << SELECT >>
flag_pi=1         # 0: only use pi from chosen models
                  # 1: use all available pi

# << SELECT >>
flag_factor=0     # 0: 2-factor -> hist-noLu and lu
                  # 1: 1-factor -> historical
                  # 2: 1-factor -> hist-noLu
                  
# << SELECT >>
flag_lulcc_measure=2    # 0: absolute change
                        # 1: area change
                        # 2: all_pixels
                        
# << SELECT >>
flag_weight=1           # 0: no weights on spatial means (not per pixel, which is automatic, but for overall area diff across continents when flag_data_agg == 1)
                        # 1: weights (asia weight of 1, australia weight of ~0.18; same for ar6 within continents)    
                        
# << SELECT >>
flag_y1=1         # 0: 1915
                  # 1: 1965

# << SELECT >>
flag_len=0        # 0: 50
                  # 1: 100

# << SELECT >>
flag_resample=0    # 0: 5 year block means
                   # 1: 10 year block means
                   # 2: 2, 25 yr block means which are subtracted to collapse time dim

# << SELECT >>
flag_var=0   # 0: tasmax

# << SELECT >>  # confidence intervals on scaling factors
ci_bnds = 0.95    # means  beta - 0.95 cummulative quantile and + 0.95 cummulative quantile, 
  
# << SELECT >> 
flag_reg=0        # 0: OLS
                  # 1: TLS


# << SELECT >>
trunc=0

seasons = ['jja',
           'djf',
           'annual',
           'max']
analyses = ['global',
            'continental',
            'ar6',
            'combined']
agg_opts = ['ar6',
            'continental']
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
weight_opts = ['no_weights',
               'weights']
lu_techniques = ['individual',
                 'mean']
start_years = [1915,
               1965]
lengths = [50,
           100]
resample=['5Y',
          '10Y',
          '25Y']
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

agg = agg_opts[flag_data_agg]
grid = grids[flag_grid]
pi = pi_opts[flag_pi]
exp_list = factors[flag_factor]
measure = measures[flag_lulcc_measure]
weight = weight_opts[flag_weight]
y1 = start_years[flag_y1]
length = lengths[flag_len]
freq = resample[flag_resample]
var = variables[flag_var]
reg = regressions[flag_reg]

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

nx = {}
nx['CanESM5'] = np.asarray(([35,7]))
nx['CNRM-ESM2-1'] = np.asarray(([11,3]))
nx['IPSL-CM6A-LR'] = np.asarray(([32,4]))
nx['UKESM1-0-LL'] = np.asarray(([16,4]))

#==============================================================================
# get data 
#==============================================================================
    
#%%============================================================================

# mod fingerprint (nx is dummy var not used in OLS OF)
os.chdir(pklDIR)
if os.path.isfile('mod_inputs_{}-flagfactor_{}-grid_{}-pi_{}-agg_{}-weight_{}_{}_{}.pkl'.format(
    flag_factor,grid,pi,agg,weight,freq,t_ext,reg)):
    
    pkl_file = open('mod_inputs_{}-flagfactor_{}-grid_{}-pi_{}-agg_{}-weight_{}_{}_{}.pkl'.format(
        flag_factor,grid,pi,agg,weight,freq,t_ext,reg),'rb')
    dictionary = pk.load(pkl_file)
    pkl_file.close()
    fp = dictionary['global']
    fp_continental = dictionary['continental']
    fp_ar6 = dictionary['ar6']

#%%============================================================================

# pi data
os.chdir(pklDIR)
if os.path.isfile('pic_inputs_{}-grid_{}-pi_{}-agg_{}-weight_{}_{}_{}.pkl'.format(
    grid,pi,agg,weight,freq,t_ext,reg)):    
    
    pkl_file = open('pic_inputs_{}-grid_{}-pi_{}-agg_{}-weight_{}_{}_{}.pkl'.format(
        grid,pi,agg,weight,freq,t_ext,reg),'rb')
    dictionary = pk.load(pkl_file)
    pkl_file.close()
    ctl_data = dictionary['global']
    ctl_data_continental = dictionary['continental']
    ctl_data_ar6 = dictionary['ar6']

#%%============================================================================

# dataframes
os.chdir(sfDIR)
if agg == 'ar6':
    frame = {
        'trnd':[],
        'sgnl':[],
        'trnd_var':[],
        'sgnl_var':[],
        'trnd_p':[],
        'sgnl_p':[],
        'mod':[],
        'ar6':[]
    }
elif agg == 'continental':
    frame = {
        'trnd':[],
        'sgnl':[],
        'trnd_var':[],
        'sgnl_var':[],
        'trnd_p':[],
        'sgnl_p':[],
        'mod':[],
        'continent':[]        
    }    
df = pd.DataFrame(data=frame)
regions = gp.read_file('IPCC-WGI-reference-regions-v4.shp')
gpd_continents = gp.read_file('IPCC_WGII_continental_regions.shp')
gpd_continents = gpd_continents[
    (gpd_continents.Region != 'Antarctica')&(gpd_continents.Region != 'Small Islands')
] 
regions = gp.clip(regions,gpd_continents)
regions['keep'] = [0]*len(regions.Acronym)

for c in continents.keys():
    for ar6 in continents[c]:
        regions.at[ar6,'Continent'] = c
        regions.at[ar6,'keep'] = 1  
            
regions = regions[regions.keep!=0]  
regions = regions.drop(columns='keep')

add_cols = []
for mod in models:
    for var in ['trnd','sgnl']:
        add_cols.append('{}_{}-p'.format(mod,var))
regions = pd.concat([regions,pd.DataFrame(columns=add_cols)])
if agg == 'continental':
    regions = regions.dissolve(by='Continent')

for mod in models:
    for c in continents.keys():
        if agg == 'ar6':
            for ar6 in continents[c]:
                y = fp_ar6[mod][ar6][1]
                x = np.arange(1,len(y)+1)
                t = np.polyfit(x,y,1)[0]
                s = np.mean(y[int(len(y)/2):]) - np.mean(y[:int(len(y)/2)])
                mn = nx[mod][1]
                pic = np.transpose(ctl_data_ar6[ar6])
                picn = pic.shape[1]
                pic_tv = np.var(np.polyfit(x,pic,1)[0,:])
                pic_sv = np.var((np.mean(pic[int(pic.shape[0]/2):],axis=0) - np.mean(pic[:int(pic.shape[0]/2)],axis=0)))
                pic_tm = np.mean(np.polyfit(x,pic,1)[0,:])
                pic_sm = np.mean((np.mean(pic[int(pic.shape[0]/2):],axis=0) - np.mean(pic[:int(pic.shape[0]/2)],axis=0)))
                tv = np.reciprocal(nx[mod].astype('float'))[0]*pic_tv + np.reciprocal(nx[mod].astype('float'))[1]*pic_tv
                sv = np.reciprocal(nx[mod].astype('float'))[0]*pic_sv + np.reciprocal(nx[mod].astype('float'))[1]*pic_sv
                _,tp = scp.ttest_ind_from_stats(
                    mean1=t,
                    std1=np.sqrt(tv),
                    nobs1=mn,
                    mean2=pic_tm,
                    std2=np.sqrt(pic_tv),
                    nobs2=picn
                )
                _,sp = scp.ttest_ind_from_stats(
                    mean1=s,
                    std1=np.sqrt(sv),
                    nobs1=mn,
                    mean2=pic_sm,
                    std2=np.sqrt(pic_sv),
                    nobs2=picn
                )
                df = df.append(
                    {'trnd':t,
                    'sgnl':s,
                    'pic_trnd':pic_tm,
                    'pic_sgnl':pic_sm,
                    'trnd_var':tv,
                    'sgnl_var':sv,
                    'trnd_p':tp,
                    'sgnl_p':sp,
                    'mod':mod,
                    'ar6':ar6
                    },
                    ignore_index=True
                )
        elif agg == 'continental':
            y = fp[mod][1]
            x = np.arange(1,len(y)+1)
            t = np.polyfit(x,y,1)[0]
            s = np.mean(y[int(len(y)/2):]) - np.mean(y[:int(len(y)/2)])
            mn = nx[mod][1]
            pic = np.transpose(ctl_data)
            picn = pic.shape[1]
            pic_tv = np.var(np.polyfit(x,pic,1)[0,:])
            pic_sv = np.var((np.mean(pic[int(pic.shape[0]/2):],axis=0) - np.mean(pic[:int(pic.shape[0]/2)],axis=0)))
            pic_tm = np.mean(np.polyfit(x,pic,1)[0,:])
            pic_sm = np.mean((np.mean(pic[int(pic.shape[0]/2):],axis=0) - np.mean(pic[:int(pic.shape[0]/2)],axis=0)))
            tv = np.reciprocal(nx[mod].astype('float'))[0]*pic_tv + np.reciprocal(nx[mod].astype('float'))[1]*pic_tv
            sv = np.reciprocal(nx[mod].astype('float'))[0]*pic_sv + np.reciprocal(nx[mod].astype('float'))[1]*pic_sv
            _,tp = scp.ttest_ind_from_stats(
                mean1=t,
                std1=np.sqrt(tv),
                nobs1=mn,
                mean2=pic_tm,
                std2=np.sqrt(pic_tv),
                nobs2=picn
            )
            _,sp = scp.ttest_ind_from_stats(
                mean1=s,
                std1=np.sqrt(sv),
                nobs1=mn,
                mean2=pic_sm,
                std2=np.sqrt(pic_sv),
                nobs2=picn
            )
            df = df.append(
                {'trnd':t,
                'sgnl':s,
                'pic_trnd':pic_tm,
                'pic_sgnl':pic_sm,
                'trnd_var':tv,
                'sgnl_var':sv,
                'trnd_p':tp,
                'sgnl_p':sp,
                'mod':mod,
                'continent':c
                },
                ignore_index=True
            )            

cmap_whole = plt.cm.get_cmap('PRGn')  
color_mapping = {
    1:cmap_whole(0.85),
    2:cmap_whole(0.7),
    3:cmap_whole(0.55),
    4:'lightgrey'
}

for mod in models:
    for c in continents.keys():
        if agg == 'ar6':
            for ar6 in continents[c]:
                for var in ['trnd','sgnl']:
                    p = df.loc[
                        (df['mod']==mod)&(df['ar6']==float(ar6)),
                        '{}_p'.format(var)
                    ].values.item()
                    regions.at[ar6,'{}_{}-p'.format(mod,var)] = classifier(p)
        elif agg == 'continental':
            for var in ['trnd','sgnl']:
                p = df.loc[
                    (df['mod']==mod)&(df['continent']==c),
                    '{}_p'.format(var)
                ].values.item()
                regions.at[c,'{}_{}-p'.format(mod,var)] = classifier(p)            
                
f, axes = plt.subplots(
    nrows=len(models),
    ncols=2,
    figsize=(12,10)
)
i = 0
for ax_row,mod in zip(axes,models):
    for ax,var in zip(ax_row,['trnd','sgnl']):
        regions.plot(
            ax=ax,
            color=regions['{}_{}-p'.format(mod,var)].map(color_mapping),
            edgecolor='black',
            linewidth=0.3
        )
        regions.boundary.plot(
            ax=ax,
            edgecolor='black',
            linewidth=0.3
        )
        ax.set_title(
            letters[i],
            loc='left',
            fontweight='bold',
            fontsize=10
        )        
        if var == 'trnd':
            ax.text(
                -0.07, 
                0.55, 
                mod, 
                va='bottom', 
                ha='center',# # create legend with patche for hsitnolu and lu det/att levels
                fontweight='bold',
                rotation='vertical', 
                rotation_mode='anchor',
                transform=ax.transAxes
            )
        if mod == 'CanESM5':
            ax.set_title(
                var,
                loc='center',
                fontweight='bold',
                fontsize=10
            )     
        ax.set_yticks([])
        ax.set_xticks([])                   
        i += 1
                
# %%
