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
stats_ds = {}

for mod in models:    
    
    stats_ds[mod] = xr.Dataset(
        coords=dict(
            lat = ('lat',ar6_land[mod].lat.values),
            lon = ('lon',ar6_land[mod].lon.values)
        )
    )
    mod_slope,mod_p,_ = vectorize_lreg(mod_ens[mod]['lu'])
    stats_ds[mod]['lu_slope'] = mod_slope
    stats_ds[mod]['lu_p'] = mod_p # maybe change to 0/1 for significance
    
    for lu in lulcc:
        
        # trends
        lc_slope,lc_p,_ = vectorize_lreg(maps[mod][lu])
        stats_ds[mod]['{}_slope'.format(lu)] = lc_slope
        stats_ds[mod]['{}_p'.format(lu)] = lc_p # maybe change to 0/1 for significance
        
        # correlation
        _,_,corr = vectorize_lreg(mod_ens[mod]['lu'],
                                  da_x=maps[mod][lu])
        stats_ds[mod]['lu-{}_corr'.format(lu)] = corr
        

#%%============================================================================

def data_lumper(dataset,
                models,
                maptype):
    
    if maptype == 'lu':
        
        data = np.empty(1)
        for mod in models:
            mod_data = dataset[mod]['lu_slope'].values.flatten()
            data = np.append(data,mod_data)
        
    elif maptype == 'lc':

        data = np.empty(1)
        for mod in models:
            for lc in ['treeFrac','cropFrac']:
                mod_data = dataset[mod]['{}_slope'.format(lc)].values.flatten()
                data = np.append(data,mod_data)
               
    data = data[~np.isnan(data)]
    return data


def colormap_details(sequence_string,
                     data):

    # identify colors for land cover transition trends
    cmap_brbg = plt.cm.get_cmap(sequence_string)
    cmap55 = cmap_brbg(0.01)
    cmap50 = cmap_brbg(0.05)   #blue
    cmap45 = cmap_brbg(0.1)
    cmap40 = cmap_brbg(0.15)
    cmap35 = cmap_brbg(0.2)
    cmap30 = cmap_brbg(0.25)
    cmap25 = cmap_brbg(0.3)
    cmap20 = cmap_brbg(0.325)
    cmap10 = cmap_brbg(0.4)
    cmap5 = cmap_brbg(0.475)
    cmap0 = col_zero
    cmap_5 = cmap_brbg(0.525)
    cmap_10 = cmap_brbg(0.6)
    cmap_20 = cmap_brbg(0.625)
    cmap_25 = cmap_brbg(0.7)
    cmap_30 = cmap_brbg(0.75)
    cmap_35 = cmap_brbg(0.8)
    cmap_40 = cmap_brbg(0.85)
    cmap_45 = cmap_brbg(0.9)
    cmap_50 = cmap_brbg(0.95)  #red
    cmap_55 = cmap_brbg(0.99)

    colors = [cmap_45,
              cmap_35,
              cmap_30,
              cmap_25,
              cmap_10,
              cmap0,
              cmap10,
              cmap25,
              cmap30,
              cmap35,
              cmap45]

    cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))

    cmap_list.set_over(cmap55)
    cmap_list.set_under(cmap_55)

    q_samples = []
    q_samples.append(np.abs(np.quantile(data,0.95)))
    q_samples.append(np.abs(np.quantile(data,0.05)))
        
    start = np.around(np.max(q_samples),decimals=4)
    inc = start/5
    values = [-1*start,
              -1*start+inc,
              -1*start+inc*2,
              -1*start+inc*3,
              -1*start+inc*4,
              -0.001,
              0.001,
              start-inc*4,
              start-inc*3,
              start-inc*2,
              start-inc,
              start]

    tick_locs = [-1*start,
                 -1*start+inc,
                 -1*start+inc*2,
                 -1*start+inc*3,
                 -1*start+inc*4,
                 0,
                 start-inc*4,
                 start-inc*3,
                 start-inc*2,
                 start-inc,
                 start]

    tick_labels = [str(np.around(-1*start,decimals=1)),
                   str(np.around(-1*start+inc,decimals=1)),
                   str(np.around(-1*start+inc*2,decimals=1)),
                   str(np.around(-1*start+inc*3,decimals=1)),
                   str(np.around(-1*start+inc*4,decimals=1)),
                   str(0),
                   str(np.around(start-inc*4,decimals=1)),
                   str(np.around(start-inc*3,decimals=1)),
                   str(np.around(start-inc*2,decimals=1)),
                   str(np.around(start-inc,decimals=1)),
                   str(np.around(start,decimals=1))]

    norm = mpl.colors.BoundaryNorm(values,cmap_list.N)
    
    return cmap_list,tick_locs,tick_labels,norm

col_cbticlbl = '0'   # colorbar color of tick labels
col_cbtic = '0.5'   # colorbar color of ticks
col_cbedg = '0.9'   # colorbar color of edge
cb_ticlen = 3.5   # colorbar length of ticks
cb_ticwid = 0.4   # colorbar thickness of ticks
cb_edgthic = 0   # colorbar thickness of edges between colors
cblabel = 'corr'  # colorbar label
col_zero = 'gray'   # zero change color
sbplt_lw = 0.1   # linewidth on projection panels
cstlin_lw = 0.2   # linewidth for coastlines

title_font = 18
cbtitle_font = 18
tick_font = 18
legend_font=12

# placment lu trends cbar
cb_lu_x0 = 0.025
cb_lu_y0 = 0.0
cb_lu_xlen = 0.45
cb_lu_ylen = 0.015

# placment lc trends cbar
cb_lc_x0 = 0.525
cb_lc_y0 = 0.0
cb_lc_xlen = 0.45
cb_lc_ylen = 0.015

east = 180
west = -180
north = 80
south = -60
extent = [west,east,south,north]

cmap_list_lu,tick_locs_lu,tick_labels_lu,norm_lu = colormap_details('RdBu_r',
                                                                     data_lumper(stats_ds,models,maptype='lu'))
cmap_list_lc,tick_locs_lc,tick_labels_lc,norm_lc = colormap_details('BrBG',
                                                                     data_lumper(stats_ds,models,maptype='lc'))

f, axes = plt.subplots(nrows=len(models),
                       ncols=3,
                       subplot_kw={'projection':ccrs.PlateCarree()})

cbax_lu = f.add_axes([cb_lu_x0, 
                      cb_lu_y0, 
                      cb_lu_xlen, 
                      cb_lu_ylen])

cbax_lc = f.add_axes([cb_lc_x0, 
                      cb_lc_y0, 
                      cb_lc_xlen, 
                      cb_lc_ylen])

for mod,row_axes in zip(models,axes):
    stats_ds[mod]['lu_slope'].plot(ax=row_axes[0],
                                   transform=ccrs.PlateCarree(),
                                   cmap=cmap_list_lu,
                                   cbar_ax=cbax_lu,
                                   center=0,
                                   norm=norm_lu,
                                   add_labels=False)
    stats_ds[mod]['treeFrac_slope'].plot(ax=row_axes[1],
                                         transform=ccrs.PlateCarree(),
                                         cmap=cmap_list_lc,
                                         cbar_ax=cbax_lc,
                                         center=0,
                                         norm=norm_lc,
                                         add_labels=False)
    stats_ds[mod]['cropFrac_slope'].plot(ax=row_axes[2],
                                         transform=ccrs.PlateCarree(),
                                         cmap=cmap_list_lc,
                                         cbar_ax=cbax_lc,
                                         center=0,
                                         norm=norm_lc,
                                         add_labels=False)
    for ax in row_axes:
        ax.set_extent(extent,
                      crs=ccrs.PlateCarree())
    
cb_lu = mpl.colorbar.ColorbarBase(ax=cbax_lu, 
                                   cmap=cmap_list_lu,
                                   norm=norm_lu,
                                   spacing='uniform',
                                   orientation='horizontal',
                                   extend='both',
                                   ticks=tick_locs_lu,
                                   drawedges=False)
cb_lu.set_label('Depth bias (ISIMIP - GLDB)',
                 size=title_font)
cb_lu.ax.xaxis.set_label_position('top')
cb_lu.ax.tick_params(labelcolor=col_cbticlbl,
                      labelsize=tick_font,
                      color=col_cbtic,
                      length=cb_ticlen,
                      width=cb_ticwid,
                      direction='out'); 
cb_lu.ax.set_xticklabels(tick_labels_lu)
                        #   rotation=45)
cb_lu.outline.set_edgecolor(col_cbedg)
cb_lu.outline.set_linewidth(cb_edgthic)

cb_lc = mpl.colorbar.ColorbarBase(ax=cbax_lc, 
                                   cmap=cmap_list_lc,
                                   norm=norm_lc,
                                   spacing='uniform',
                                   orientation='horizontal',
                                   extend='both',
                                   ticks=tick_locs_lc,
                                   drawedges=False)
cb_lc.set_label('Depth bias (ISIMIP - GLDB)',
                 size=title_font)
cb_lc.ax.xaxis.set_label_position('top')
cb_lc.ax.tick_params(labelcolor=col_cbticlbl,
                      labelsize=tick_font,
                      color=col_cbtic,
                      length=cb_ticlen,
                      width=cb_ticwid,
                      direction='out'); 
cb_lc.ax.set_xticklabels(tick_labels_lc)
                        #   rotation=45)
cb_lc.outline.set_edgecolor(col_cbedg)
cb_lc.outline.set_linewidth(cb_edgthic)
    
    

