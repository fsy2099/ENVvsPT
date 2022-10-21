#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:45:14 2022

@author: shiyi
"""
from sys import platform
import numpy as np
import numpy.matlib
import os 
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window
import random

# Recoding signal
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
Fs = 24414.0625
stiDur = [0.01, 0.05, 0.2]
stiRate = [900, 4500]
stiITD = [-0.1, 0, 0.1]
stienvITD = [-0.1, 0, 0.1]
file_names = os.listdir(Sig_path)
origin_names = [file_name for file_name in file_names if all([x in file_name for x in ["_OriginSigArray.npy"]])]
corr_names = [file_name for file_name in file_names if all([x in file_name for x in ["_CleanArtifactcCorrPeak.npy"]])]
H0_names = [file_name for file_name in file_names if all([x in file_name for x in ["_H0.npy"]])]
clean_names = [file_name for file_name in file_names if all([x in file_name for x in ["_CleanSig.npy"]])]
SNR_names = [file_name for file_name in file_names if all([x in file_name for x in ["_SNR.npy"]])]
SF_names = [file_name for file_name in file_names if all([x in file_name for x in ["_ScaleFactor.npy"]])]
FFT_names = [file_name for file_name in file_names if all([x in file_name for x in ["_FFT.npy"]])]
Predict_names = [file_name for file_name in file_names if all([x in file_name for x in ["Predict.npy"]])]
#%% load test data to have H0 threshold
x = 3
sig_name = FFT_names[x][:-23]
sig_array = np.load(results_path+origin_names[x])
corr_array = np.load(results_path+corr_names[x])
H0_array = np.load(results_path+H0_names[x])
clean_array = np.load(results_path+clean_names[x])
SNR_array = np.load(results_path+SNR_names[x])
SF_array = np.load(results_path+SF_names[x])
FFT_array = np.load(results_path+FFT_names[x])
Predict_array = np.load(results_path+Predict_names[x])

# set specific threshold to filt unmatch H0 results and set reject mark as 1
reject_mark = np.zeros((32, 3, 3, 3))
clean_uni = np.zeros((32, 3, 3, 3))
uni_num = 13
for cc in range(32):
    for dd in range(3):
        for ii in range(3):
            for jj in range(3):
                x = len(np.unique(corr_array[cc, dd, ii, jj, :]))
                clean_uni[cc, dd, ii, jj] = x
                if x <= uni_num:
                    reject_mark[cc, dd, ii, jj] = 1
idx_test = np.array(np.where(clean_uni==uni_num))
#%%                    
idx_random = random.sample(range(0, idx_test.shape[1]), 10)
for xx in idx_random:
#for xx in reject:
#    [cc, dd, ii, jj] = reject_idx[:, xx]
    [cc, dd, ii, jj] = idx_test[:, xx]
    StimParam = [stiDur[dd], stiITD[ii], stienvITD[jj]]
    idx_num = len(np.unique(corr_array[cc, dd, ii, jj, :]))
    print(StimParam)
    print(cc, dd, ii, jj)
#    print(idx_num)
    
    #[cc, dd, ii, jAbstract j] = [31, 2, 2, 2]
    Artifact_length = int(np.around(Fs*stiDur[dd]))
    tt = 9
    stim_9 = StimulusData[0, dd, ii, jj, :, 0]
    stim_9[stim_9 < 0] = 0
    predict = np.convolve(H0_array[cc, dd, ii, jj, :], stim_9)*SF_array[cc, 0, dd, ii, jj]
    idx = np.argmax(predict[:Artifact_length])
    plt.figure(figsize=(15, 10))
    plt.suptitle(sig_name[:-4]+'_900pps_ch'+str(cc)+'_Dur'+str(stiDur[dd])+'_ptITD'+str(stiITD[ii])+'_envITD'+str(stienvITD[jj])+'_reject'+' ['+str(cc)+str(dd)+str(ii)+str(jj)+'] '+ str(idx_num))
        
    #plt.subplot(2,2,1)
    #plt.title('clean signal [: artifact length]')
    #clean_sig = clean_array[cc, 0, dd, ii, jj, 2:Artifact_length, :] - clean_array[cc, 0, dd, ii, jj, 2, :]
    #plt.plot(clean_sig)
    
    plt.subplot(2,2,1)
    plt.title('clean signal [maxidx-100:maxidx+100]')
    plt.plot(clean_array[cc, 0, dd, ii, jj, idx-100:idx+100, :]) 
    
    plt.subplot(2,2,2)
    original_sig = sig_array[cc, 0, dd, ii, jj, idx-12:idx+13, :] - sig_array[cc, 0, dd, ii, jj, idx-12, :]
    original_mean = np.mean(sig_array[cc, 0, dd, ii, jj, idx-12:idx+13, :] - sig_array[cc, 0, dd, ii, jj, idx-12, :], 1)
    predict_sig = predict[idx-12:idx+13]
    plt.plot(original_sig, 'r-o', label = 'original')
    plt.plot(predict_sig, 'b-o', label = 'predict')
    plt.plot(original_mean, 'g-o', label = 'mean')
                  
    plt.subplot(2,2,3)
    fft_length = FFT_array.shape[-2]
    fre = np.linspace(0,Fs/2,fft_length)
    plt.plot(FFT_array[cc, 0, dd, ii, jj, :, :])
    plt.xlim([800, 1000])
    plt.ylim([-0.01, 0.02])
    plt.title('FFT 800~1000Hz')
#%%

#%%
accept_idx = np.array(np.where(reject_mark==0))
reject_idx = np.array(np.where(reject_mark==1))
reject = random.sample(range(0, reject_idx.shape[1]), 10)
accept = random.sample(range(0, accept_idx.shape[1]), 10)                    
#%% plot reject mark
plt.figure(figsize=(10,15))
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
#%%


#%%
for x in range(reject_idx.shape[-1]):
    [cc, dd, ii, jj] = reject_idx[:, x]
    SNR_array[cc, :, dd, ii, jj, :] = 0

avg_array = np.zeros((2, 3))
for ff in range(2):
    for dd in range(3):        
        Y = SNR_array[:, ff, dd, :, :, :]
        b = np.reshape(Y, (np.product(Y.shape),1))
        avg_array[ff, dd] = sum(b)/len(np.nonzero(b)[0])                    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
