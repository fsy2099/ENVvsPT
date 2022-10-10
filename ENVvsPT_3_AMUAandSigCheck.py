#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:35:30 2022

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
    
def CalcFFT(S, A, Fs):
    insig = S*np.hanning(len(S))
    insig = np.concatenate((S, np.zeros((int(Fs/2) - Artifact_length))), axis = 0)
    fft = np.abs(np.fft.rfft(insig))
    return fft 
   
def CalcSNR(S, F, Fs):
    fre = np.around(np.linspace(0,Fs/2,len(S)))
    SNR = np.mean(S[np.where((fre<F+10) & (fre>F-10))[0]])/np.mean(S[np.where((fre<F+100) & (fre>F+20))[0]])
    return SNR

#%%
file_names = os.listdir(Sig_path)
sig_names = [file_name for file_name in file_names if all([x in file_name for x in ["_ArtifactRejected.npy"]])][1:2]
dic = np.load(Sig_path+sig_names[0], allow_pickle = True).item()
reject_mark_clean  = dic['reject_mark']
reject_mark = dic['reject_mark']
H0_array = dic['H0_array']
clean_array = dic['clean_array']
SNR_array = dic['SNR_array']
SF_array = dic['SF_array']
fft_array = dic['fft_array']
Fs = dic['Fs']
original_names = [file_name for file_name in file_names if all([x in file_name for x in ["IC.npy"]])][1:2]
dic_origin = np.load(Sig_path+original_names[0], allow_pickle = True).item()
sig_array = dic_origin['OriginSig_array']
stmDur = [0.01, 0.05, 0.2]
stmRate = [900, 4500]
stmITD = [-0.1, 0, 0.1]
stmenvITD = [-0.1, 0, 0.1]
#%% 
fft_new = np.zeros((32, 2, 3, 3, 3, int(np.around(Fs/4)), 30))
for cc in range(32):
    for dd in range(3):
        for ii in range(3):
            for jj in range(3):
                for tt in range(30):
                    Artifact_length = int(np.around(Fs*stmDur[dd]))
                    fft_new[cc, 0, dd, ii, jj, :, tt] = CalcFFT(clean_array[cc, 0, dd, ii, jj, :Artifact_length, tt]-clean_array[cc, 0, dd, ii, jj, 0, tt], Artifact_length, Fs)

#%%
fft_new = np.zeros((32, 2, 3, 3, 3, int(np.around(Fs/4)), 30))
for cc in range(32):
    for dd in range(3):
        for ii in range(3):
            for jj in range(3):
                for tt in range(30):
                    Artifact_length = int(np.around(Fs*stmDur[dd]))
                    fft_new[cc, 0, dd, ii, jj, :, tt] = CalcFFT(clean_array[cc, 0, dd, ii, jj, :Artifact_length, tt], Artifact_length, Fs)
                    fft_new[cc, 1, dd, ii, jj, :, tt] = CalcFFT(clean_array[cc, 1, dd, ii, jj, :Artifact_length, tt]-clean_array[cc, 1, dd, ii, jj, 0, tt], Artifact_length, Fs)
#%% set the SNR ratio using mean+-3sd
reject_mark = reject_mark_clean
SNR_thres = np.zeros((2, 3))
for rr in range(2):
    for dd in range(3):
        SNR_temp = np.reshape(SNR_array[:, rr, dd, :, :, :], (1, 32*3*3*30))
        SNR_temp = SNR_temp[SNR_temp != 0]
        SNR_thres[rr, dd] = np.mean(SNR_temp)+np.std(SNR_temp)    
for cc in range(32):
    for rr in range(2):
        for dd in range(3):
            for ii in range(3):
                for jj in range(3):
                    if np.array(np.where(SNR_array[cc, rr, dd, ii, jj, :]>SNR_thres[rr, dd])).shape[1] > 5:
                        reject_mark[cc, dd, ii, jj] = 3
                        print('artifact residue, reject the data')
                        continue
#%%
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
reject_type = 0
# 0 is clean, 1 is H0 shape shift, 2 is H0 resifual, 3 is SNR residual
reject_idx = np.array(np.where(reject_mark==reject_type))
print(reject_idx.shape)
#%%
[cc, dd, ii, jj] = reject_idx[:, 42]
StimParam = [stmDur[dd], stmITD[ii], stmenvITD[jj]]
print(StimParam)
#%%
#[cc, dd, ii, jj] = [31, 2, 2, 2]
Artifact_length = int(np.around(Fs*stmDur[dd]))
tt = 13
stim_9 = StimulusData[0, dd, ii, jj, :, 0]
stim_9[stim_9 < 0] = 0
predict = np.convolve(H0_array[cc, dd, ii, jj, :], stim_9)*SF_array[cc, 0, dd, ii, jj]
idx = np.argmax(predict[:Artifact_length])
plt.figure(figsize=(13, 14))
plt.suptitle(sig_names[0][:-4]+'_900pps_ch'+str(cc)+'_Dur'+str(stmDur[dd])+'_ptITD'+str(stmITD[ii])+'_envITD'+str(stmenvITD[jj])+'_reject'+str(reject_type))

#plt.subplot(2,2,1)
#plt.title('clean signal [: artifact length]')
#clean_sig = clean_array[cc, 0, dd, ii, jj, 2:Artifact_length, :] - clean_array[cc, 0, dd, ii, jj, 2, :]
#plt.plot(clean_sig)

plt.subplot(3,2,1)
plt.title('clean signal [maxidx-100:maxidx+100]')
plt.plot(clean_array[cc, 0, dd, ii, jj, idx-100:idx+100, :]) 

plt.subplot(3,2,2)
plt.title('clean signal [maxidx-100:maxidx+100]')
plt.plot(clean_array[cc, 1, dd, ii, jj, idx-100:idx+100, :]) 

plt.subplot(3,2,3)
original_sig = sig_array[cc, 0, dd, ii, jj, idx-12:idx+13, :] - sig_array[cc, 0, dd, ii, jj, idx-12, :]
predict_sig = predict[idx-12:idx+13]
plt.plot(original_sig[:, tt], 'r-o', label = 'original')
plt.plot(predict_sig, 'b-o', label = 'predict')
plt.legend()
plt.title('original #' + str(tt)+' & predict')

plt.subplot(3,2,4)  
fft_length = fft_array.shape[-2]
fre = np.linspace(0,Fs/2,fft_length)
for x in range(30):
    plt.plot(fre, fft_array[cc, 0, dd, ii, jj, :, x])
    plt.xlim([800, 1000])
    plt.ylim([-0.01, 0.02])
plt.title('FFT 800~1000Hz (900pps)')

plt.subplot(3,2,5)  
fft_length = fft_array.shape[-2]
fre = np.linspace(0,Fs/2,fft_length)
for x in range(30):
    plt.plot(fre, fft_array[cc, 1, dd, ii, jj, :, x])
    plt.xlim([4400, 4600])
    plt.ylim([-0.01, 0.02])
plt.title('FFT 800~1000Hz (4500pps)')

plt.subplot(3,2,6)
plt.plot(SNR_array[cc, 0, dd, ii, jj, :], label = '900pps')
plt.plot(SNR_array[cc, 1, dd, ii, jj, :], label = '4500pps')
plt.legend()
plt.title('SNR ratio')

#%%
#%%
#[cc, dd, ii, jj] = [31, 2, 2, 2]
Artifact_length = int(np.around(Fs*stmDur[dd]))
tt = 13
stim_9 = StimulusData[0, dd, ii, jj, :, 0]
stim_9[stim_9 < 0] = 0
predict = np.convolve(H0_array[cc, dd, ii, jj, :], stim_9)*SF_array[cc, 0, dd, ii, jj]
idx = np.argmax(predict[:Artifact_length])
plt.figure(figsize=(13, 14))
plt.suptitle(sig_names[0][:-4]+'_900pps_ch'+str(cc)+'_Dur'+str(stmDur[dd])+'_ptITD'+str(stmITD[ii])+'_envITD'+str(stmenvITD[jj])+'_reject'+str(reject_type))

#plt.subplot(2,2,1)
#plt.title('clean signal [: artifact length]')
#clean_sig = clean_array[cc, 0, dd, ii, jj, 2:Artifact_length, :] - clean_array[cc, 0, dd, ii, jj, 2, :]
#plt.plot(clean_sig)

plt.subplot(3,2,1)
plt.title('clean signal [maxidx-100:maxidx+100]')
plt.plot(clean_array[cc, 0, dd, ii, jj, idx-100:idx+100, :]) 

plt.subplot(3,2,2)
plt.title('clean signal [maxidx-100:maxidx+100]')
plt.plot(clean_array[cc, 1, dd, ii, jj, idx-100:idx+100, :]) 

plt.subplot(3,2,3)
original_sig = np.mean(sig_array[cc, 0, dd, ii, jj, idx-12:idx+13, :] - sig_array[cc, 0, dd, ii, jj, idx-12, :], -1)
predict_sig = predict[idx-12:idx+13]
plt.plot(original_sig, 'r-o', label = 'original')
plt.plot(predict_sig, 'b-o', label = 'predict')
plt.legend()
plt.title('original #' + str(tt)+' & predict')

plt.subplot(3,2,4)  
fft_length = fft_new.shape[-2]
fre = np.linspace(0,Fs/2,fft_length)
for x in range(30):
    plt.plot(fre, fft_new[cc, 0, dd, ii, jj, :, x])
#    plt.xlim([800, 1000])
    plt.ylim([-0.01, 0.02])
plt.title('FFT 800~1000Hz (900pps)')

plt.subplot(3,2,5)  
fft_length = fft_new.shape[-2]
fre = np.linspace(0,Fs/2,fft_length)
for x in range(30):
    plt.plot(fre, fft_new[cc, 1, dd, ii, jj, :, x])
#    plt.xlim([4400, 4600])
    plt.ylim([-0.01, 0.02])
plt.title('FFT 800~1000Hz (4500pps)')

plt.subplot(3,2,6)
plt.plot(SNR_array[cc, 0, dd, ii, jj, :], label = '900pps')
plt.plot(SNR_array[cc, 1, dd, ii, jj, :], label = '4500pps')
plt.legend()
plt.title('SNR ratio')

#%% any SNR ratio differencebetween duration and pulse rate?
def non_zero_mean(np_arr):
    exist = (np_arr!=0)
    num = np_arr.sum()
    den = exist.sum()
    return num/den

SNR_mean = np.zeros((6))
SNR_mean[0] = non_zero_mean(np.reshape(SNR_array[:, 0, 0, :, :, :], (1, 32*3*3*30)))
SNR_mean[1] = non_zero_mean(np.reshape(SNR_array[:, 0, 1, :, :, :], (1, 32*3*3*30)))
SNR_mean[2] = non_zero_mean(np.reshape(SNR_array[:, 0, 2, :, :, :], (1, 32*3*3*30)))
SNR_mean[3] = non_zero_mean(np.reshape(SNR_array[:, 1, 0, :, :, :], (1, 32*3*3*30)))
SNR_mean[4] = non_zero_mean(np.reshape(SNR_array[:, 1, 1, :, :, :], (1, 32*3*3*30)))
SNR_mean[5] = non_zero_mean(np.reshape(SNR_array[:, 1, 2, :, :, :], (1, 32*3*3*30)))

a = SNR_array[:, 1, 0, :, :, :]
b = np.reshape(a, (1,32*3*3*30))
plt.figure()
plt.plot(b.T)

























