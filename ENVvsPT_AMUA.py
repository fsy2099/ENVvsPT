#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:11:37 2022

@author: shiyi
"""

import pandas as pd
from sys import platform
import numpy as np
from scipy import stats
import os
import seaborn as sns  
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window, resample

#give path of predict artifact and original data
if platform == "darwin":  
    os.chdir('/Users/fei/Documents/CI_projects/20220127_Analysis')
    fname = '20220127_ENVvsPT_5_P10_1'   
elif platform == "linux":
    Sig_path = '/home/shiyi/Documents/CI_project/2022_01_27/Results_data/'
Fs = 24414.0625
Fs_downsample = 2000

#%%
def AMUAFilterCoeffs(fs,lowpass=6000):
        nyq = 0.5*fs
        bBand,aBand = butter(2,(300/nyq, 6000/nyq),'bandpass')
        bLow,aLow = butter(2,(lowpass/nyq),'lowpass')
        bNotch, aNotch = iirnotch(50, 30, fs)
        return [[bBand, aBand], [bLow, aLow], [bNotch, aNotch]]
    
def calcAMUA(fs, ori_signal, Fs_downsample, padLen=300):
        coefs=AMUAFilterCoeffs(fs)
        bpCoefs=coefs[0]
        lpCoefs=coefs[1]
        NotchCoefs = coefs[2]
        insig = filtfilt(NotchCoefs[0], NotchCoefs[1], ori_signal, axis=0, padlen=padLen)
        insig = np.flip(insig)
        insig=filtfilt(bpCoefs[0],bpCoefs[1], insig, axis=0, padlen=padLen)
        insig=np.abs(insig)
        insig=filtfilt(lpCoefs[0],lpCoefs[1],insig,axis=0, padlen=padLen)
        insig = np.flip(insig)          
        # Fs_downsample
        # signal = resample_poly(insig, Fs_downsample, int(fs), axis=0)
        downsample_length = int((insig.shape[0]/fs)*Fs_downsample)
        signal=resample(insig,downsample_length)
        
        return signal

    
#def calcAMUA(fs, ori_signal, Fs_downsample, padLen=300):
#        coefs=AMUAFilterCoeffs(fs)
#        bpCoefs=coefs[0]
#        lpCoefs=coefs[1]
#        NotchCoefs = coefs[2]
#        insig = filtfilt(NotchCoefs[0], NotchCoefs[1], ori_signal, axis=0, padlen=padLen)
#        insig = np.flip(insig)
#        insig=filtfilt(bpCoefs[0],bpCoefs[1], insig,axis=0, padlen=padLen)
#        insig=np.abs(insig)
#        insig=filtfilt(lpCoefs[0],lpCoefs[1],insig,axis=0, padlen=padLen)
#        insig = np.flip(insig)          
#        # Fs_downsample
#        signal = resample_poly(insig, Fs_downsample, int(fs), axis=0)
#        return signal
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.random((10,10,)))

fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0.01, wspace=0.01)
axn = gs.subplots(sharex='col', sharey='row')
#fig, axn = plt.subplots(2, 2, sharex='col', sharey='row')
cbar_ax = fig.add_axes([.91, .3, .03, .5])

for i, ax in enumerate(axn.flat):
    print(i)
    sns.heatmap(df, ax=ax,
                cbar=i == 0,
                vmin=0, vmax=1,
                cbar_ax=None if i else cbar_ax)

fig.tight_layout(rect=[0, 0, .9, 1])
#%%
position = ['P4']
file_names = os.listdir(Sig_path)
for pp in position:    
    sig_name = [file_name for file_name in file_names if all([x in file_name for x in [pp, "_AMUA.npy"]])]
    for x in sig_name:
        amua_array = np.load(Sig_path+x, allow_pickle = True)
t = np.arange(0.005, 0.04, 1/Fs_downsample)
IdxStart = round(0.005*Fs_downsample)
IdxEnd = round(0.04*Fs_downsample)
channel = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
ff = 0
dd = 0
stiRate = [900, 4500]
stiDur = [0.01, 0.05, 0.2]
stiITD = [-0.1, 0, 0.1]
amua_plot = np.mean(np.mean(amua_array[channel, ff, dd, :, :, IdxStart:IdxEnd, :], -1), -1)
amua_z = np.zeros((len(channel), 3, 3))
for x in range(len(channel)):
    amua_z[x, :, :] = stats.zscore(amua_plot[x, :, :])
#amua_z = stats.zscore(amua_plot, axis=1)
#%%
fig = plt.figure()
gs = fig.add_gridspec(1, 7, hspace=0.1, wspace=0.01)
axs = gs.subplots(sharex='col', sharey='row')
fig.suptitle('ND_900pps_0.01s AMUA')
for x in range(6):
        g = sns.heatmap(amua_plot[x, :, :],cmap='plasma',cbar=False,ax=axs[x])
        axs[x].set_xticklabels(['-0.1', '0', '0.1'])
        axs[x].set_title('channel' + str(channel[x]))
        axs[x].set_aspect('equal')
g = sns.heatmap(amua_plot[6, :, :],cmap='plasma',ax=axs[6])
axs[6].set_aspect('equal')
axs[6].set_title('channel' + str(channel[6]))
axs[6].set_xticklabels(['-0.1', '0', '0.1'])
axs[0].set_yticklabels(['-0.1', '0', '0.1'])
axs[3].set_xlabel('ENV_ITD')
axs[0].set_ylabel('PT_ITD')
#%%
fig = plt.figure(figsize=(15,7), frameon = False)
gs = fig.add_gridspec(2, 5, hspace=0.2, wspace=0.1)
axs = gs.subplots(sharex='col', sharey='row')
fig.suptitle('ND_900pps_0.01s AMUA')
cbar_ax = fig.add_axes([.91, .3, .03, .5])
for i, ax in enumerate(axs.flat):
    sns.heatmap(amua_z[i, :, :], ax=ax, cbar=i == 0, vmin=-1.5, vmax=1.5, cmap='plasma',
                cbar_ax=None if i else cbar_ax)
    ax.set_title('channel' + str(channel[i]))
    ax.set_xticklabels(['-0.1', '0', '0.1'])
    ax.set_yticklabels(['-0.1', '0', '0.1'])
axs[1, 2].set_xlabel('ENV_ITD')
fig.text(0.1, 0.5, 'PT_ITD', va='center', rotation='vertical')
#    ax.set_aspect('equal')
#for x in range(6):
#        g = sns.heatmap(amua_z[x, :, :],cmap='plasma',cbar=False,ax=axs[x])
#        axs[x].set_xticklabels(['-0.1', '0', '0.1'])
#        axs[x].set_title('channel' + str(channel[x]))
#g = sns.heatmap(amua_z[6, :, :],cmap='plasma',ax=axs[6])
#axs[6].set_title('channel' + str(channel[6]))
#axs[6].set_xticklabels(['-0.1', '0', '0.1'])
#axs[0].set_yticklabels(['-0.1', '0', '0.1'])
#axs[3].set_xlabel('ENV_ITD')
#axs[0].set_ylabel('PT_ITD')
#%%
position = ['P4']
file_names = os.listdir(Sig_path)
for pp in position:    
    sig_name = [file_name for file_name in file_names if all([x in file_name for x in [pp, "_IC.npy"]])]
    for x in sig_name:
        dic = np.load(Sig_path+x, allow_pickle = True).item()
        sig_array = dic['OriginSig_array']

stiRate = [900, 4500]
stiDur = [0.01, 0.05, 0.2]
stiITD = [-0.1, 0, 0.1]

cc = 12
ff = 0
dd = 0
IdxStart = round(0.012*Fs)

fig,axs = plt.subplots(3,3)
fig.suptitle(sig_name[0][:-4]+'_chan'+str(cc)+'_'+str(stiRate[ff])+'pps'+'_Dur'+str(stiDur[dd]))
t = np.arange(0, 0.1, 1/Fs_downsample)
for ii in range(3):
    for jj in range(3):
        sig = sig_array[cc, ff, dd, ii, jj, IdxStart:, :]-np.mean(sig_array[cc, ff, dd, ii, jj, IdxStart:, :], 0)
        amua = np.mean(calcAMUA(Fs, sig, Fs_downsample), 1)
        axs[ii, jj].plot(t, amua[:len(t)])
        axs[ii, jj].set_title('PT_ITD:' + str(stiITD[ii]) + ' ENV_ITD:' + str(stiITD[jj]))
        for ax in fig.get_axes():
            ax.label_outer()
#%%
position = ['P4']
file_names = os.listdir(Sig_path)
for pp in position:    
    sig_name = [file_name for file_name in file_names if all([x in file_name for x in [pp, "_AMUA.npy"]])]
    for x in sig_name:
        amua_array = np.load(Sig_path+x, allow_pickle = True)          
        
stiRate = [900, 4500]
stiDur = [0.01, 0.05, 0.2]
stiITD = [-0.1, 0, 0.1]

cc = 10
ff = 0
dd = 0

t = np.arange(0, 0.1, 1/Fs_downsample)
fig = plt.figure(figsize=(10,15), frameon = False)
gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.15)
axs = gs.subplots(sharex='col', sharey='row')
amua_uv = amua_array*1000000
#fig,axs = plt.subplots(3,3, figsize=(10,15))
for ii in range(3):
    for jj in range(3):
        amua = np.mean(amua_uv[cc, ff, dd, ii, jj, :len(t), :], -1)
        axs[ii, jj].plot(t, amua[:len(t)])
        axs[ii, jj].set_title('PT_ITD:' + str(stiITD[ii]) + ' ENV_ITD:' + str(stiITD[jj]))
        axs[ii, jj].set_ylim(3, 15)
        
        for ax in fig.get_axes():
            ax.label_outer()
fig.suptitle('ND_900pps_0.01s AMUA trace')
axs[2, 1].set_xlabel('Time')
fig.text(0.08, 0.5, 'MicroVolt', va='center', rotation='vertical')
#%%
position = ['P4']
file_names = os.listdir(Sig_path)
for pp in position:    
    sig_name = [file_name for file_name in file_names if all([x in file_name for x in [pp, "_AMUA.npy"]])]
    for x in sig_name:
        amua_array = np.load(Sig_path+x, allow_pickle = True)          
        
stiRate = [900, 4500]
stiDur = [0.01, 0.05, 0.2]
stiITD = [-0.1, 0, 0.1]

cc = 10
ff = 0
dd = 0

Fs_downsample = 2000
Wn = 2*200/Fs_downsample
bLow,aLow = butter(2, Wn, 'lowpass')
t = np.arange(0, 0.1, 1/Fs_downsample)
fig = plt.figure(figsize=(10,15), frameon = False)
gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.15)
axs = gs.subplots(sharex='col', sharey='row')
amua_uv = amua_array*1000000
#fig,axs = plt.subplots(3,3, figsize=(10,15))
for ii in range(3):
    for jj in range(3):
        insig = amua_uv[cc, ff, dd, ii, jj, :len(t), :]
        insig = np.flip(insig)
        insig=filtfilt(bLow,aLow, insig, axis=0, padlen=100)
        insig=np.abs(insig)
        insig = np.flip(insig)
        amua = np.mean(insig, -1)
        axs[ii, jj].plot(t, amua[:len(t)])
        axs[ii, jj].set_title('PT_ITD:' + str(stiITD[ii]) + ' ENV_ITD:' + str(stiITD[jj]))
        axs[ii, jj].set_ylim(3, 15)
        
        for ax in fig.get_axes():
            ax.label_outer()
fig.suptitle('ND_900pps_0.01s AMUA trace')
axs[2, 1].set_xlabel('Time')
fig.text(0.08, 0.5, 'MicroVolt', va='center', rotation='vertical')











