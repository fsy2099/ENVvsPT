#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:04:16 2022

@author: shiyi
"""

from sys import platform
import numpy as np
import numpy.matlib
import os 
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window, resample
import random

if platform == "darwin":  
    os.chdir('/Users/fei/Documents/CI_projects/20220127_Analysis')
    fname = '20220127_ENVvsPT_5_P10_1'   
elif platform == "linux":
    Sig_path = '/media/shiyi/CIdata/2022_08_22/Results_Data/'
    results_path = '/media/shiyi/CIdata/2022_08_22/Results_Data/'
# Stimulus waveform    
if platform == "darwin":
    stimPath = '/Users/fei/Documents/CI_projects/StimData_v1/'
elif platform == "linux":
    stimPath = '/home/colliculus/ephys/4/CIproject/0_ENVvsPTephys/Analysis/'
StimulusData = np.load(stimPath+'Stim_ENVvsFS_template.npy')
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


def plotRejectMark(X):
    x = plt.figure(figsize=(10,15))
    plt.xticks(np.arange(1, 33, 1))
    plt.title('reject mark')
    #plt.subplot(1,2,1)
    y = 0
    for dd in range(3):
        for ii in range(3):
            for jj in range(3):
                y = y+1
                for cc in range(32):
                    if reject_mark[cc, dd, ii, jj] == 0:                        
                        plt.plot(cc+1,y, 'ko')
                    elif reject_mark[cc, dd, ii, jj] == 1:
                        plt.plot(cc+1,y, 'ro')
                    elif reject_mark[cc, dd, ii, jj] == 2:
                        plt.plot(cc+1,y, 'bo')
                    else:
                        plt.plot(cc+1,y, 'go')
    return x
#%% 
uni_num = 13
Fs = 24414.0625
Fs_down = 2000
nsamples_down = int(Fs_down*0.5)-1 
stiDur = [0.01, 0.05, 0.2]
stiRate = [900, 4500]
stiITD = [-0.1, 0, 0.1]
stienvITD = [-0.1, 0, 0.1]
file_names = os.listdir(Sig_path)
corr_names = [file_name for file_name in file_names if all([x in file_name for x in ["_CleanArtifactcCorrPeak.npy"]])]
clean_names = [file_name for file_name in file_names if all([x in file_name for x in ["_CleanSig.npy"]])]
#%%
for x in range(len(corr_names)):
#x = 0
    fname = clean_names[x][:-13]
    print(fname)
    corr_array = np.load(results_path+corr_names[x])
    clean_array = np.load(results_path+clean_names[x])
    reject_mark = np.zeros((32, 3, 3, 3))
    AMUA_array = np.zeros((32, 2, 3, 3, 3, nsamples_down, clean_array.shape[-1]))
    for cc in range(32):
        for dd in range(3):
            for ii in range(3):
                for jj in range(3):
                    # reject mark
                    StimParam = [cc, stiDur[dd], stiITD[ii], stienvITD[jj]]
                    print(StimParam)
                    corr_1d = corr_array[cc, dd, ii, jj, :]
                    corr_uninum = len(np.unique(corr_array[cc, dd, ii, jj, :]))
                    if corr_uninum <= uni_num:
                        reject_mark[cc, dd, ii, jj] = 1
                    # calculate AMUA
                    for ff in range(2):
                        clean = clean_array[cc, ff, dd, ii, jj, :]
                        AMUA_array[cc, ff, dd, ii, jj, :, :] = calcAMUA(Fs, clean, Fs_down)
    np.save(results_path+fname+'_RejectMark.npy',reject_mark)
    np.save(results_path+fname+'_AMUA.npy',AMUA_array)
    print('data saved')
    plotRejectMark(reject_mark)
#%%
reject_mark = np.load(results_path+fname+'_RejectMark.npy')
AMUA_array = np.load(results_path+fname+'_AMUA.npy')
#%%
reject_idx = np.array(np.where(reject_mark == 1))
reject_random = random.sample(range(0, reject_idx.shape[1]), 10)
time_range = 0.01                   
for x in reject_random:
    [cc, dd, ii, jj] = reject_idx[:, x]
    plt.figure(figsize=(15, 10))
    plt.subplot(1,2,1)
    plt.plot(AMUA_array[cc, 0, dd, ii, jj, :int(time_range*Fs_down), :])
    plt.subplot(1,2,2)
    plt.plot(AMUA_array[cc, 1, dd, ii, jj, :int(time_range*Fs_down), :])
#%%
accept_idx = np.array(np.where(reject_mark == 0))
accept_random = random.sample(range(0, accept_idx.shape[1]), 10)
t = np.arange(0, 0.1, 1/Fs_down)
Wn = 2*200/Fs_down 
bLow,aLow = butter(2, Wn, 'lowpass')                  
for x in accept_random:
    plt.figure()
    [cc, dd, ii, jj] = accept_idx[:, x]
    plt.subplot(1,2,1)
    insig = AMUA_array[cc, 0, dd, ii, jj, :len(t), :]
    insig = np.flip(insig)
    insig=filtfilt(bLow,aLow, insig, axis=0, padlen=100)
    insig=np.abs(insig)
    insig = np.flip(insig)
    amua = np.mean(insig, -1)
    plt.plot(t, amua)
    plt.subplot(1,2,2)
    insig = AMUA_array[cc, 1, dd, ii, jj, :len(t), :]
    insig = np.flip(insig)
    insig=filtfilt(bLow,aLow, insig, axis=0, padlen=100)
    insig=np.abs(insig)
    insig = np.flip(insig)
    amua = np.mean(insig, -1)
    plt.plot(t, amua)























    