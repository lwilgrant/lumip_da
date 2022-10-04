#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 2018
@author: alexander.winkler@mpimet.mpg.de
Title: Optimal Fingerprinting after Ribes et al., 2009
"""

#%%=============================================================================
# import
#===============================================================================


import numpy as np
from pooch import test
import scipy.linalg as spla
import scipy.stats as sps
import xarray as xr
import pandas as pd
import os
import regionmask as rm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle as pk
import copy as cp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from random import shuffle
from matplotlib.lines import Line2D
import matplotlib as mpl
import seaborn as sns
from scipy import stats as sts
from eofs.xarray import Eof
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from matplotlib.patches import ConnectionPatch


#==============================================================================
# functions
#==============================================================================

#%%==============================================================================    

def nc_read(file,
            y1,
            var,
            flag_temp_center,
            flag_standardize,
            obs=False,
            freq=False):
    
    """ Read in netcdfs based on variable and set time.
    
    Parameters
    ----------
    file : files in data directory
    
    Returns
    ------- 
    Xarray data array
    """
    
    ds = xr.open_dataset(file,decode_times=False)
    da = ds[var].squeeze()
    units, reference_date = da.time.attrs['units'].split('since')
    reference_date = reference_date.replace(reference_date[1:5],str(y1))[1:]
    new_date = pd.date_range(start=reference_date, periods=da.sizes['time'], freq='YS')
    da['time'] = new_date
    
    if 'height' in da.coords:
        da = da.drop('height')
    if obs == 'berkley_earth':
        da = da.rename({'latitude':'lat','longitude':'lon'})
        
    da['time'] = da['time.year']
    
    
    
    if flag_temp_center == 1:
        
        da = da - da.mean(dim='time')
        
    if flag_standardize == 1:
        
        da = standard_data(da)
    
    return da

#%%==============================================================================   

def standard_data(da):
    
    climatology_mean = da.mean("time")
    climatology_std = da.std("time")
    stand_anomalies = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        da,
        climatology_mean,
        climatology_std,
    )
    return stand_anomalies

#%%==============================================================================   

def pickler(pklDIR,
            var_fin,
            analysis,
            grid,
            t_ext,
            exp_list):
    
    os.chdir(pklDIR)
    if len(exp_list) == 2:
        pkl_file = open('var_fin_2-factor_{}-grid_{}_{}.pkl'.format(grid,analysis,t_ext),'wb')
    elif len(exp_list) == 1:
        exp = exp_list[0]
        pkl_file = open('var_fin_1-factor_{}_{}-grid_{}_{}.pkl'.format(exp,grid,analysis,t_ext),'wb')
    pk.dump(var_fin,pkl_file)
    pkl_file.close()

#%%==============================================================================    

def ar6_mask(da):
    

    lat = da.lat.values
    lon = da.lon.values
    ar6_regs = rm.defined_regions.ar6.land.mask(lon,lat)
    landmask = rm.defined_regions.natural_earth.land_110.mask(lon,lat)
    ar6_regs = ar6_regs.where(landmask == 0)
    
    return ar6_regs

#%%==============================================================================    

def weighted_mean(continents,
                  da,
                  ar6_regs,
                  ns):
    
    nt = len(da.time.values)
    matrix = np.zeros(shape=(nt,ns))
    s = 0
    for c in continents.keys():
        for i in continents[c]:
            da_i = da.where(ar6_regs==i,
                            drop=True)                    
            for t in np.arange(nt):
                da_i_t = da_i.isel(time=t)
                weights = np.cos(np.deg2rad(da_i_t.lat))
                weights.name='weights'
                da_i_t_w = da_i_t.weighted(weights).mean(('lon','lat')).values
                matrix[t,s]= da_i_t_w
            s += 1
            
    return matrix

#%%==============================================================================    

def temp_center(ns,
                mod_ar6):
    
    for s in np.arange(ns):
        arr = mod_ar6[:,s]
        arr_mean = np.mean(arr)
        arr_center = arr - arr_mean
        mod_ar6[:,s] = arr_center
        
    return mod_ar6

#%%==============================================================================    

def ensembler(data_list,
              ax=False):
    if not ax:
        matrix = np.stack(data_list,axis=1)
        nx = np.shape(matrix)[1]
        ens_mean = np.mean(matrix,axis=1)
    
    if ax:
        matrix = np.stack(data_list,axis=ax)
        nx = np.shape(matrix)[ax]
        ens_mean = np.mean(matrix,axis=ax)
    
    return ens_mean,nx

#%%==============================================================================    

def da_ensembler(data):
    
    concat_dim = np.arange(len(data))
    aligned = xr.concat(data,dim=concat_dim)
    mean = aligned.mean(dim='concat_dim')
    
    return mean
    
    
#%%==============================================================================    

def c(x):
   col = plt.cm.BrBG(x)
   fig, ax = plt.subplots(figsize=(1,1))
   fig.set_facecolor(col)
   ax.axis("off")
   plt.show()

#%%==============================================================================           

def lengther(mask):
    mask = mask.values.flatten()
    mask_length = np.count_nonzero(mask)
    return(mask_length)

#%%============================================================================

def data_lumper(dataset,
                models,
                scale,
                lulcc_type):
    
    data = np.empty(1)
    for mod in models:
        if scale == 'global':
            mod_data = dataset[mod][lulcc_type].values.flatten()
        elif scale == 'latitudinal':
            for ltr in ['boreal','temperate_north','tropics']:
                mod_data = dataset[mod][lulcc_type][ltr].values.flatten()
        data = np.append(data,mod_data)
                        
    data = data[~np.isnan(data)]
    return data

#%%============================================================================

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
    cmap0 = 'gray'
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

    colors = [cmap_50,
              cmap_45,
              cmap_40,
              cmap_35,
              cmap_30,
              cmap_25,
              cmap_10,
              cmap0,
              cmap10,
              cmap25,
              cmap30,
              cmap35,
              cmap40,
              cmap45,
              cmap50]

    cmap_list = mpl.colors.ListedColormap(colors,
                                          N=len(colors))
    
    cmap_list.set_over(cmap55)
    cmap_list.set_under(cmap_55)

    q_samples = []
    q_samples.append(np.abs(np.quantile(data,0.99)))
    q_samples.append(np.abs(np.quantile(data,0.01)))
        
    start = np.around(np.max(q_samples),decimals=4)
    inc = start/6
    values = [np.around(-1*start,decimals=2),
              np.around(-1*start+inc,decimals=2),
              np.around(-1*start+inc*2,decimals=2),
              np.around(-1*start+inc*3,decimals=2),
              np.around(-1*start+inc*4,decimals=2),
              np.around(-1*start+inc*5,decimals=2),
              np.around(start-inc*5,decimals=2),
              np.around(start-inc*4,decimals=2),
              np.around(start-inc*3,decimals=2),
              np.around(start-inc*2,decimals=2),
              np.around(start-inc,decimals=2),
              np.around(start,decimals=2)]
    
    # values = [np.around(-1*start,decimals=2),
    #           np.around(-1*start+inc,decimals=2),
    #           np.around(-1*start+inc*2,decimals=2),
    #           np.around(-1*start+inc*3,decimals=2),
    #           np.around(-1*start+inc*4,decimals=2),
    #           null_bnds[0],
    #           null_bnds[1],
    #           np.around(start-inc*4,decimals=2),
    #           np.around(start-inc*3,decimals=2),
    #           np.around(start-inc*2,decimals=2),
    #           np.around(start-inc,decimals=2),
    #           np.around(start,decimals=2)]

    tick_locs = [np.around(-1*start,decimals=2),
                 np.around(-1*start+inc,decimals=2),
                 np.around(-1*start+inc*2,decimals=2),
                 np.around(-1*start+inc*3,decimals=2),
                 np.around(-1*start+inc*4,decimals=2),
                 0,
                 np.around(start-inc*4,decimals=2),
                 np.around(start-inc*3,decimals=2),
                 np.around(start-inc*2,decimals=2),
                 np.around(start-inc,decimals=2),
                 np.around(start,decimals=2)]

    tick_labels = [str(np.around(-1*start,decimals=2)),
                   str(np.around(-1*start+inc,decimals=2)),
                   str(np.around(-1*start+inc*2,decimals=2)),
                   str(np.around(-1*start+inc*3,decimals=2)),
                   str(np.around(-1*start+inc*4,decimals=2)),
                   str(0),
                   str(np.around(start-inc*4,decimals=2)),
                   str(np.around(start-inc*3,decimals=2)),
                   str(np.around(start-inc*2,decimals=2)),
                   str(np.around(start-inc,decimals=2)),
                   str(np.around(start,decimals=2))]

    norm = mpl.colors.BoundaryNorm(values,cmap_list.N)
    
    return cmap_list,tick_locs,tick_labels,norm,values


#%%==============================================================================

def sig_noise_plot(sig_noise,
                   lat_pcts,
                   eof_dict,
                   scale,
                   lulcc,
                   models,
                   continents,
                   lat_ranges,
                   letters,
                   t_ext,
                   flag_svplt,
                   outDIR,
                   lulcc_type):

    x=7
    y=6
    eof_color='BrBG'
    cmap=plt.cm.get_cmap(eof_color)
    colors = {}
    colors['treeFrac'] = cmap(0.85)
    colors['lu_treeFrac_rls'] = cmap(0.6)
    colors['lu_treeFrac'] = cmap(0.95)
    colors['cropFrac'] = cmap(0.15)
    colors['lu_cropFrac'] = cmap(0.05)
    colors['pi'] = 'lightgray'

    # legend location
    le_x0 = 0.7
    le_y0 = 0.4
    le_xlen = 0.2
    le_ylen = 0.5
    
    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 2
    
    col_cbticlbl = '0'   # colorbar color of tick labels
    col_cbtic = '0.5'   # colorbar color of ticks
    col_cbedg = '0.9'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors
    cblabel = 'corr'  # colorbar label
    sbplt_lw = 0.1   # linewidth on projection panels
    cstlin_lw = 0.75   # linewidth for coastlines

    # fonts
    title_font = 11
    cbtitle_font = 11
    tick_font = 10
    legend_font=14

    # placment eof cbar
    cb_x0 = 1.05
    cb_y0 = 0.05
    cb_xlen = 0.025
    cb_ylen = 0.9

    # extent
    east = 180
    west = -180
    north = 80
    south = -80
    extent = [west,east,south,north]

    cmap_list,_,_,_,_ = colormap_details('BrBG_r',
                                        data_lumper(eof_dict,
                                                    models,
                                                    scale,
                                                    lulcc_type))
    levels = np.around(np.arange(-0.04,0.045,0.005),decimals=3)
    levels = np.delete(levels,np.where(levels==0))
    tick_labels = np.around(np.arange(-0.04,0.045,0.01),decimals=3)
    tick_locs = tick_labels
    norm = mpl.colors.BoundaryNorm(levels,cmap_list.N)
    

    if scale == 'global':
        
        f = plt.figure(figsize=(x,y))
        
        # histograms
        gs1 = gridspec.GridSpec(4,1)
        h_left = 0.0
        h_bottom = 0.0
        h_right = 0.45
        h_top = 1.0
        h_rect = [h_left, h_bottom, h_right, h_top]
        
        ax0 = f.add_subplot(gs1[0])    
        ax1 = f.add_subplot(gs1[1])
        ax2 = f.add_subplot(gs1[2])
        ax3 = f.add_subplot(gs1[3])
        
        h_axes = [ax0,ax1,ax2,ax3]    
        
        gs1.tight_layout(figure=f, rect=h_rect, h_pad=2)
        
        # maps of eof
        gs2 = gridspec.GridSpec(4,1)
        m_left = 0.45
        m_bottom = 0.0
        m_right = 1.0
        m_top = 1.0
        m_rect = [m_left, m_bottom, m_right, m_top]
        
        ax4 = f.add_subplot(gs2[0],projection=ccrs.PlateCarree()) # hist-nolu bias
        ax5 = f.add_subplot(gs2[1],projection=ccrs.PlateCarree()) # obs eof (cru)
        ax6 = f.add_subplot(gs2[2],projection=ccrs.PlateCarree()) # delta eof
        ax7 = f.add_subplot(gs2[3],projection=ccrs.PlateCarree()) # eof bias
    
        map_axes = [ax4,ax5,ax6,ax7]       
        
        gs2.tight_layout(figure=f, rect=m_rect, h_pad=2) 

        cbax = f.add_axes([cb_x0, 
                           cb_y0, 
                           cb_xlen, 
                           cb_ylen])
        i = 0
        
        for mod,ax in zip(models,h_axes):

            # collect pic across treeFrac and treeFrac_inv
            pic = []

            for lc in lulcc:
                
                pic.append(sig_noise['pic_S_N_{}'.format(mod)].sel(landcover=lc))
                
            pic = xr.concat(pic,dim='landcover')
            
            sns.distplot(pic,
                         ax=ax,
                         fit=sts.norm,
                         fit_kws={"color":colors['pi']},
                         color = colors['pi'],
                         label='PIC',
                         kde=False)
                
            # plot lu s/n
            ax.vlines(x=np.abs(sig_noise['lu_S_N'].sel(models=mod,landcover='treeFrac')),
                      ymin=0,
                      ymax=0.5,
                      colors=colors['lu_treeFrac'],
                      label='LU mean',
                      zorder=20)     
            # plot lu rls s/n
            ax.vlines(x=np.abs(sig_noise['lu_rls_S_N_{}'.format(mod)].sel(landcover='treeFrac')),
                      ymin=0,
                      ymax=0.5,
                      colors=colors['lu_treeFrac_rls'],
                      label='LU rls',
                      zorder=10)                 
                
            # likelihood s/n for detectability
            ax.vlines(x=0.95,
                      ymin=0,
                      ymax=0.1,
                      colors='indianred',
                      label=None,
                      zorder=30)
                    #   label='likely')
            ax.vlines(x=1.64,
                      ymin=0,
                      ymax=0.1,
                      colors='firebrick',
                      label=None,
                      zorder=30)
                    #   label='very likely')
            ax.vlines(x=2.57,
                      ymin=0,
                      ymax=0.1,
                      colors='maroon',
                      label=None,
                      zorder=30)
                    #   label='virtually certain')
                    
                    # Horizontal lines correspond to the thresholds of:
                    #     likely (S/N > 0.95; detectable at 66% confidence), 
                    #     very likely (S/N > 1.64, 90% confidence) 
                    #     virtually certain (S/N > 2.57, 99% confidence).
                
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_title(None)
            ax.set_xlabel(None)
            ax.xaxis.set_ticklabels([])
            ax.set_xticks(np.arange(-4,5,2))
            ax.set_xlim(-4,4)
            
            if mod == 'CanESM5':
                height = 0.2
            else:
                height= 0.0

            ax.set_ylabel('Frequency',
                            fontsize=title_font)
            ax.text(-0.3,
                    height,
                    mod,
                    fontweight='bold',
                    rotation='vertical',
                    fontsize=title_font,
                    transform=ax.transAxes) 
            
            if mod == models[-1]:
                ax.set_xlabel('S/N')
                ax.xaxis.set_ticklabels(np.arange(-4,5,2))
            
            if i == 0:
                ax.legend(
                    frameon=False,
                    bbox_to_anchor=(le_x0, le_y0, le_xlen, le_ylen),
                    fontsize=legend_font,
                    labelspacing=legend_entrypad
                )
                
            i += 1
            
        i = 0
        
        for mod,ax in zip(models,map_axes):

            eof_dict[mod]['treeFrac'].plot(ax=ax,
                                           cmap=cmap_list,
                                           cbar_ax=cbax,
                                           levels=levels,
                                           extend='both',
                                           add_labels=False)
            ax.set_extent(extent,
                          crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=cstlin_lw)
            
            i += 1

        cb = mpl.colorbar.ColorbarBase(ax=cbax, 
                                          cmap=cmap_list,
                                          norm=norm,
                                          spacing='uniform',
                                          orientation='vertical',
                                          extend='both',
                                          ticks=tick_locs,
                                          drawedges=False)
        cb.set_label('EOF loading [-]',
                        size=cbtitle_font)
        cb.ax.xaxis.set_label_position('top')
        cb.ax.tick_params(labelcolor=col_cbticlbl,
                             labelsize=tick_font,
                             color=col_cbtic,
                             length=cb_ticlen,
                             width=cb_ticwid,
                             direction='out'); 
        cb.ax.set_yticklabels(tick_labels)
        cb.outline.set_edgecolor(col_cbedg)
        cb.outline.set_linewidth(cb_edgthic)
        
            
        for i,ax in enumerate([ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7]):
            
            ax.set_title(letters[i],
                         loc='left',
                         fontweight='bold',
                         fontsize=title_font)
        
        f.savefig(outDIR+'/pca_noise_{}_{}.png'.format(scale,t_ext),bbox_inches='tight',dpi=250)

    elif scale == 'latitudinal':

        x=13
        y=15
        f = plt.figure(figsize=(x,y))
        
        eof_color='BrBG'
        cmap=plt.cm.get_cmap(eof_color)
        colors = {}
        colors['treeFrac'] = cmap(0.85)
        colors['lu_treeFrac_rls'] = cmap(0.6)
        colors['lu_treeFrac'] = cmap(0.95)
        colors['cropFrac'] = cmap(0.15)
        colors['lu_cropFrac'] = cmap(0.05)
        colors['pi'] = 'lightgray'

        # legend location
        le_x0 = 0.75
        le_y0 = -1.2
        le_xlen = 0.2
        le_ylen = 0.5
        
        # space between entries
        legend_entrypad = 0.5

        # length per entry
        legend_entrylen = 2
        
        col_cbticlbl = '0'   # colorbar color of tick labels
        col_cbtic = '0.5'   # colorbar color of ticks
        col_cbedg = '0.9'   # colorbar color of edge
        cb_ticlen = 3.5   # colorbar length of ticks
        cb_ticwid = 0.4   # colorbar thickness of ticks
        cb_edgthic = 0   # colorbar thickness of edges between colors
        cblabel = 'corr'  # colorbar label
        sbplt_lw = 0.1   # linewidth on projection panels
        cstlin_lw = 0.75   # linewidth for coastlines

        # fonts
        title_font = 11
        cbtitle_font = 11
        tick_font = 10
        legend_font=14        
            
        # placment eof cbar
        cb_x0 = 0.5725
        cb_y0 = 0.075
        cb_xlen = 0.305
        cb_ylen = 0.015                
        
        # spec = f.add_gridspec(12,2)
        gs0 = gridspec.GridSpec(4,2)
        
        ax01 = f.add_subplot(gs0[0,1],projection=ccrs.PlateCarree())
        ax02 = f.add_subplot(gs0[1,1],projection=ccrs.PlateCarree())
        ax03 = f.add_subplot(gs0[2,1],projection=ccrs.PlateCarree())
        ax04 = f.add_subplot(gs0[3,1],projection=ccrs.PlateCarree())
        
        map_axes = [ax01,ax02,ax03,ax04]
        h_axes = []
        
        gs00 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0,0])
        ax00 = f.add_subplot(gs00[0])
        ax10 = f.add_subplot(gs00[1])
        ax20 = f.add_subplot(gs00[2])
        h_axes.append([ax00,ax10,ax20])
        
        gs10 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1,0])
        ax30 = f.add_subplot(gs10[0])
        ax40 = f.add_subplot(gs10[1])
        ax50 = f.add_subplot(gs10[2])
        h_axes.append([ax30,ax40,ax50])
        
        gs20 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[2,0])
        ax60 = f.add_subplot(gs20[0])
        ax70 = f.add_subplot(gs20[1])
        ax80 = f.add_subplot(gs20[2])
        h_axes.append([ax60,ax70,ax80])
        
        gs30 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[3,0])
        ax90 = f.add_subplot(gs30[0])
        ax100 = f.add_subplot(gs30[1])
        ax110 = f.add_subplot(gs30[2]) 
        h_axes.append([ax90,ax100,ax110])  
        
        cbax = f.add_axes([cb_x0, 
                           cb_y0, 
                           cb_xlen, 
                           cb_ylen])             

        cmap_list,_,_,_,_ = colormap_details('BrBG_r',
                                            data_lumper(eof_dict,
                                                        models,
                                                        scale,
                                                        lulcc_type))
        levels = np.around(np.arange(-0.04,0.045,0.005),decimals=3)
        levels = np.delete(levels,np.where(levels==0))
        tick_labels = np.around(np.arange(-0.04,0.045,0.01),decimals=3)
        tick_locs = tick_labels
        norm = mpl.colors.BoundaryNorm(levels,cmap_list.N)

        i = 0        

        for mod,ax_set in zip(models,h_axes):
            
            for ax,ltr in zip(ax_set,lat_ranges.keys()):

                # collect pic across treeFrac and treeFrac_inv
                pic = []

                for lc in lulcc:
                    
                    pic.append(sig_noise['pic_S_N_{}_{}'.format(mod,ltr)].sel(landcover=lc))
                    
                pic = xr.concat(pic,dim='landcover')
                
                sns.distplot(pic,
                             ax=ax,
                             fit=sts.norm,
                             fit_kws={"color":colors['pi']},
                             color = colors['pi'],
                            #  label='PIC',
                             label=r'$\mathregular{t_{PIC}}$',
                             kde=False)
                    
                # plot lu s/n
                # ax.vlines(x=np.abs(sig_noise['lu_S_N'].sel(models=mod,landcover='treeFrac',lat_keys=ltr)),
                ax.vlines(x=np.abs(sig_noise['lu_S_N'].sel(models=mod,landcover=lulcc_type,lat_keys=ltr)),
                          ymin=0,
                          ymax=0.5,
                          colors=colors['lu_treeFrac'],
                        #   label='LU',
                          label=r'$\mathregular{t_{LU}}$',
                          zorder=20)     
                # plot lu rls s/n
                # ax.vlines(x=np.abs(sig_noise['lu_rls_S_N_{}_{}'.format(mod,ltr)].sel(landcover='treeFrac')),
                #           ymin=0,
                #           ymax=0.5,
                #           colors=colors['lu_treeFrac_rls'],
                #           label='LU rls',
                #           zorder=10)                 
                    
                # likelihood s/n for detectability
                # lat_pcts[lat][str(pct)]
                # ax.vlines(x=0.95,
                #           ymin=0,
                #           ymax=0.1,
                #           colors='indianred',
                #           label=None,
                #           zorder=30)
                ax.vlines(x=lat_pcts[ltr]['66'],
                          ymin=0,
                          ymax=0.1,
                          colors='indianred',
                          lw=1,
                          label='66%',
                          zorder=30)                
                # ax.vlines(x=1.64,
                #           ymin=0,
                #           ymax=0.1,
                #           colors='firebrick',
                #           label=None,
                #           zorder=30)
                ax.vlines(x=lat_pcts[ltr]['90'],
                          ymin=0,
                          ymax=0.1,
                          colors='firebrick',
                          lw=2,
                          label='90%',
                          zorder=30)                
                # ax.vlines(x=2.57,
                #           ymin=0,
                #           ymax=0.1,
                #           colors='maroon',
                #           label=None,
                #           zorder=30)
                ax.vlines(x=lat_pcts[ltr]['99'],
                          ymin=0,
                          ymax=0.1,
                          colors='maroon',
                          lw=3,
                          label='99%',
                          zorder=30)                
                
                
            for ax,lat_label in zip(ax_set,['Boreal','Temperate-north','Tropical \n to temperate-south']):
                
                if ax == ax_set[0]:
                    
                    ax.set_title(letters[i],
                                 loc='left',
                                 fontweight='bold')  
                if mod == 'UKESM1-0-LL':
                    ax.text(.05,
                            .75,
                            lat_label,
                            horizontalalignment='left',
                            transform=ax.transAxes)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_title(None)
                ax.set_xlabel(None)
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
                ax.set_xticks(np.arange(-4,5,2))
                ax.set_yticks([])
                ax.set_xlim(-4,4)
                    
                if ax == ax_set[-1]:
                    
                    ax.set_xlabel('S/N')
                    ax.xaxis.set_ticklabels(np.arange(-4,5,2)) 
                    
                if mod == models[-1]:
                    
                    ax_set[-1].legend(
                        frameon=False,
                        ncol=3,
                        bbox_to_anchor=(le_x0, le_y0, le_xlen, le_ylen),
                        fontsize=legend_font,
                        labelspacing=legend_entrypad
                    )                
            
            ax = ax_set[1]
            
            if mod == 'CanESM5':
                
                height = 0.35
                
            else:
                
                height= 0.25

            ax.set_ylabel('Frequency',
                            fontsize=12)
            ax.text(-0.15,
                    height,
                    mod,
                    fontweight='bold',
                    rotation='vertical',
                    transform=ax.transAxes)                 

                    
            i += 1
                
        for mod,ax in zip(models,map_axes):
            
            for ltr in lat_ranges:

                # eof_dict[mod]['treeFrac'][ltr].plot(ax=ax,
                eof_dict[mod][lulcc_type][ltr].plot(ax=ax,
                                                    cmap=cmap_list,
                                                    cbar_ax=cbax,
                                                    levels=levels,
                                                    extend='both',
                                                    add_labels=False)
                
            gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                              draw_labels=False,
                              linewidth=2, 
                              color='gray', 
                              alpha=1, 
                              linestyle='-')
            gl.xlabels_top = False
            gl.xlabels_bottom = False
            gl.ylabels_left = False
            gl.ylabels_right = False
            gl.xlines = False
            # gl.ylocator = mticker.FixedLocator([89.5,51.5,23.5,-23.5])
            gl.ylocator = mticker.FixedLocator([90,50,23,-50])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            ax.set_extent(extent,
                          crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=cstlin_lw)
            ax.set_title(letters[i],
                         loc='left',
                         fontweight='bold')
            
            i += 1
            
        # lines between plots
        for ax,ax_set in zip(map_axes,h_axes):
            
            x_h = 4
            y_h1 = 0.5
            # template = eof_dict['CanESM5']['treeFrac']['boreal']
            template = eof_dict['CanESM5'][lulcc_type]['boreal']
            x_m = template.lon[int(len(template.lon)/2)]*-1
            y_m1 = 80
            con = ConnectionPatch(xyA=(x_h,y_h1),
                                    xyB=(x_m,y_m1),
                                    coordsA=ax_set[0].transData,
                                    coordsB=ax.transData,
                                    color=colors['pi'])
            ax.add_artist(con)            
            
            # for ax_in,y_m1 in zip(ax_set,[51.5,23.5,-23.5]):
            for ax_in,y_m1 in zip(ax_set,[50,23,-50]):
                
                x_h = 4
                y_h1 = 0
                # template = eof_dict['CanESM5']['treeFrac']['boreal']
                template = eof_dict['CanESM5'][lulcc_type]['boreal']
                x_m = template.lon[int(len(template.lon)/2)]*-1
                con = ConnectionPatch(xyA=(x_h,y_h1),
                                      xyB=(x_m,y_m1),
                                      coordsA=ax_in.transData,
                                      coordsB=ax.transData,
                                      color=colors['pi'])
                ax.add_artist(con)
        
        cb = mpl.colorbar.ColorbarBase(ax=cbax, 
                                       cmap=cmap_list,
                                       norm=norm,
                                       spacing='uniform',
                                       orientation='horizontal',
                                       extend='both',
                                       ticks=tick_locs,
                                       drawedges=False)
        cb.set_label('EOF loading [-]',
                        size=cbtitle_font)
        cb.ax.xaxis.set_label_position('top')
        cb.ax.tick_params(labelcolor=col_cbticlbl,
                             labelsize=tick_font,
                             color=col_cbtic,
                             length=cb_ticlen,
                             width=cb_ticwid,
                             direction='out'); 
        cb.ax.set_xticklabels(tick_labels,
                              rotation=45)
        cb.outline.set_edgecolor(col_cbedg)
        cb.outline.set_linewidth(cb_edgthic)

        if flag_svplt == 0:
            pass
        elif flag_svplt == 1:
            f.savefig(outDIR+'/pca_noise_{}_{}_{}_newlabel.png'.format(scale,t_ext,lulcc_type))

    elif scale == 'continental':

        for c in continents.keys():

            f,axes = plt.subplots(nrows=len(models),
                                ncols=1,
                                figsize=(x,y))
            i = 0
            
            for mod,ax in zip(models,axes):

                for lc in lulcc:
                    
                    sns.distplot(sig_noise['pic_S_N_{}'.format(mod)].sel(landcover=lc),
                                ax=ax,
                                fit=sts.norm,
                                fit_kws={"color":colors[lc]},
                                color = colors[lc],
                                label='PIC_{}'.format(lc),
                                kde=False)
                    
                    # plot lu s/n
                    ax.vlines(x=np.abs(sig_noise['lu_S_N'].sel(models=mod,landcover='treeFrac')),
                            ymin=0,
                            ymax=0.5,
                            colors=colors['lu_treeFrac'],
                            label='lu_{}'.format(lc))         
                    
                # likelihood s/n for detectability
                ax.vlines(x=0.95,
                        ymin=0,
                        ymax=0.1,
                        colors='indianred',
                        label='likely')
                ax.vlines(x=1.64,
                        ymin=0,
                        ymax=0.1,
                        colors='firebrick',
                        label='very likely')
                ax.vlines(x=2.57,
                        ymin=0,
                        ymax=0.1,
                        colors='maroon',
                        label='virtually certain')
                        
                        # Horizontal lines correspond to the thresholds of:
                        #     likely (S/N > 0.95; detectable at 66% confidence), 
                        #     very likely (S/N > 1.64, 90% confidence) 
                        #     virtually certain (S/N > 2.57, 99% confidence).                   
                    
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_title(None)
                ax.set_title(letters[i],
                            loc='left',
                            fontweight='bold')
                ax.set_xlabel(None)
                ax.xaxis.set_ticklabels([])
                ax.set_xlim(-4,4)
                
                if mod == 'CanESM5':
                    height = 0.4
                else:
                    height= 0.3

                ax.set_ylabel('Frequency',
                                fontsize=12)
                ax.text(-0.15,
                        height,
                        mod,
                        fontweight='bold',
                        rotation='vertical',
                        transform=ax.transAxes) 
                
                if mod == models[-1]:
                    ax.set_xlabel('S/N')
                    ax.xaxis.set_ticklabels(np.arange(-4,5,1))
                
                if i == 0:
                    ax.legend(frameon=False,
                            bbox_to_anchor=(le_x0, le_y0, le_xlen, le_ylen),
                            labelspacing=legend_entrypad)
                    
                i += 1

            f.savefig(outDIR+'/pca_noise_{}_{}_{}.png'.format(scale,c,t_ext))
            
    elif scale == 'ar6':

        for c in continents.keys():
            
            for ar6 in continents[c]:

                f,axes = plt.subplots(nrows=len(models),
                                    ncols=1,
                                    figsize=(x,y))
                i = 0
                
                for mod,ax in zip(models,axes):

                    for lc in lulcc:
                        
                        sns.distplot(sig_noise['pic_S_N_{}'.format(mod)].sel(landcover=lc),
                                     ax=ax,
                                     fit=sts.norm,
                                     fit_kws={"color":colors[lc]},
                                     color = colors[lc],
                                     label='PIC_{}'.format(lc),
                                     kde=False)
                        
                        # plot lu s/n
                        ax.vlines(x=np.abs(sig_noise['lu_S_N'].sel(models=mod,landcover='treeFrac')),
                                ymin=0,
                                ymax=0.5,
                                colors=colors['lu_treeFrac'],
                                label='lu_{}'.format(lc))    
                        
                    # likelihood s/n for detectability
                    ax.vlines(x=0.95,
                              ymin=0,
                              ymax=0.1,
                              colors='indianred',
                              label='likely')
                    ax.vlines(x=1.64,
                              ymin=0,
                              ymax=0.1,
                              colors='firebrick',
                              label='very likely')
                    ax.vlines(x=2.57,
                              ymin=0,
                              ymax=0.1,
                              colors='maroon',
                              label='virtually certain')
                    
                    # Horizontal lines correspond to the thresholds of:
                    #     likely (S/N > 0.95; detectable at 66% confidence), 
                    #     very likely (S/N > 1.64, 90% confidence) 
                    #     virtually certain (S/N > 2.57, 99% confidence).                       
                        
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.set_title(None)
                    ax.set_title(letters[i],
                                 loc='left',
                                 fontweight='bold')
                    ax.set_xlabel(None)
                    ax.xaxis.set_ticklabels([])
                    ax.set_xlim(-4,4)
                    
                    if mod == 'CanESM5':
                        height = 0.4
                    else:
                        height= 0.3

                    ax.set_ylabel('Frequency',
                                    fontsize=12)
                    ax.text(-0.15,
                            height,
                            mod,
                            fontweight='bold',
                            rotation='vertical',
                            transform=ax.transAxes) 
                    
                    if mod == models[-1]:
                        ax.set_xlabel('S/N')
                        ax.xaxis.set_ticklabels(np.arange(-4,5,1))
                    
                    if i == 0:
                        ax.legend(frameon=False,
                                bbox_to_anchor=(le_x0, le_y0, le_xlen, le_ylen),
                                labelspacing=legend_entrypad)
                        
                    i += 1

                f.savefig(outDIR+'/pca_noise_{}_{}_{}_newlabel.png'.format(scale,ar6,t_ext))

        # %%
