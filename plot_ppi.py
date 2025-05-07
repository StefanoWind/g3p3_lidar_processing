# -*- coding: utf-8 -*-
'''
Plot detailed screenshots of of 360-degree PPI
'''
import os
cd=os.path.dirname(__file__)
import warnings
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib
warnings.filterwarnings('ignore')
plt.close('all')
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source=os.path.join(cd,'data/g3p3/roof.lidar.z01.b0/roof.lidar.z01.b0.20250401.010009.user5.g3p3.360.ppi.nc')
max_inv_cos=3#maximum gemoetric amplification

#location of towers [m]
x_tower1=-56
y_tower1=-358
x_tower2=-10
y_tower2=-358

#graphics
umin=3 
umax=10

#%% Initalization
data=xr.open_dataset(source)

#%%% Functions
def cosine_fit(x,ws,wd):
    return ws*np.cos(np.radians((90-x)-(270-wd)))

#%% Main
rws_qc=data.wind_speed.where(data.qc_wind_speed==0)

#%% Plots
fig=plt.figure(figsize=(18,8))

gs = gridspec.GridSpec(1, len(data.scanID)+1,width_ratios=[1]*len(data.scanID)+[0.05])

for i in data.scanID:
   ax=fig.add_subplot(gs[0,int(i.values)])
   azi_sel=np.tile(data.azimuth.sel(scanID=i).values,(len(data.range),1))
   rws_sel=rws_qc.sel(scanID=i).values
   params, covariance = curve_fit(cosine_fit, azi_sel.ravel()[~np.isnan(rws_sel.ravel())], rws_sel.ravel()[~np.isnan(rws_sel.ravel())],bounds=([0,0],[30,359.9]))
   u_sel=rws_sel/cosine_fit(azi_sel,1,params[1])
   u_sel[1/np.abs(cosine_fit(azi_sel,1,params[1]))>max_inv_cos]=np.nan
   cp=plt.pcolor(data.x,data.y,u_sel,vmin=umin,vmax=umax,cmap='coolwarm')
   ax=plt.gca()
   plt.xlim([-250,250])
   plt.ylim([-500,500])
   ax.arrow(-np.cos(np.radians(270-params[1]))*50,-np.sin(np.radians(270-params[1]))*50,
             np.cos(np.radians(270-params[1]))*50,np.sin(np.radians(270-params[1]))*50,head_width=50, head_length=50, fc='g', ec='k',width=20)
   plt.plot(0,0,'ok',markersize=7)
   plt.plot(x_tower1,y_tower1,'sk',markersize=7)
   plt.plot(x_tower2,y_tower2,'sk',markersize=7)
   plt.xlabel('W-E [m]')
   if int(i.values)==0:
       plt.ylabel('S-N [m]')
   else:
       ax.set_yticklabels([])
   xlim=ax.get_xlim()
   ylim=ax.get_ylim()
   ax.set_box_aspect(np.diff(ylim)/np.diff(xlim))
   plt.grid()
   time1=str(data.time.sel(scanID=i,beamID=0).values).replace('T',' ')[:19]
   time2=str(data.time.sel(scanID=i,beamID=len(data.beamID)-1).values).replace('T',' ')[11:19]
   plt.title(f'{time1} - {time2}')

ax=fig.add_subplot(gs[0,int(i.values)+1])
cb=plt.colorbar(cp,ax,label=r'Wind speed [m s${-1}$]')

   


