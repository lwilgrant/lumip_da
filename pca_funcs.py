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

def pickler(curDIR,
            var_fin,
            analysis,
            grid,
            t_ext,
            exp_list):
    
    os.chdir(curDIR)
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
                models):
    
    data = np.empty(1)
    for mod in models:
        mod_data = dataset[mod]['treeFrac'].values.flatten()
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
                   eof_dict,
                   scale,
                   lulcc,
                   models,
                   continents,
                   lat_ranges,
                   letters,
                   t_ext,
                   outDIR):

    x=7
    y=6
    eof_color='BrBG'
    cmap=plt.cm.get_cmap(eof_color)
    colors = {}
    colors['treeFrac'] = cmap(0.85)
    colors['lu_treeFrac_rls'] = cmap(0.75)
    colors['lu_treeFrac'] = cmap(0.95)
    colors['cropFrac'] = cmap(0.15)
    colors['lu_cropFrac'] = cmap(0.05)
    colors['pi'] = 'lightgray'

    # legend location
    le_x0 = 0.7
    le_y0 = 0.4
    le_xlen = 0.2
    le_ylen = 0.5

    legend_font = 5
    
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
    legend_font=10

    # placment eof cbar
    cb_x0 = 1.05
    cb_y0 = 0.05
    cb_xlen = 0.025
    cb_ylen = 0.9

    # extent
    east = 180
    west = -180
    north = 80
    south = -60
    extent = [west,east,south,north]

    cmap_list,_,_,_,_ = colormap_details('BrBG_r',
                                        data_lumper(eof_dict,
                                                    models))
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
                ax.legend(frameon=False,
                        bbox_to_anchor=(le_x0, le_y0, le_xlen, le_ylen),
                        fontsize=legend_font,
                        labelspacing=legend_entrypad)
                
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
        
#%%==============================================================================        

    elif scale == 'latitudinal':

        for ltr in lat_ranges.keys():

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

            # f.savefig(outDIR+'/pca_noise_{}_{}_{}.png'.format(scale,ltr,t_ext))
            
#%%==============================================================================

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

                f.savefig(outDIR+'/pca_noise_{}_{}_{}.png'.format(scale,ar6,t_ext))

        # %%
