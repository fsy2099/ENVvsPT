#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:40:41 2022

@author: shiyi
"""
from sys import platform
import RZ2ephys as ep
import os
import numpy as np
from matplotlib import pyplot as plt
# give recording data path 
if platform == "darwin":  
    os.chdir('/Users/fei/Documents/CI_projects/20220127_Analysis')
    fname = '20220127_ENVvsPT_5_P10_1'   
elif platform == "linux":
    RawDataPath = '/home/colliculus/ephys/3/0_ENVvsPT/data/2022_08_22/'
    results_path = '/media/shiyi/CIdata/2022_08_22/Results_Data/'

file_names = os.listdir(RawDataPath)
sig_names = [file_name for file_name in file_names if all([x in file_name for x in [".csv"]])]
for pp in range(len(sig_names)):
    fname = sig_names[pp][:-4]  
    swps, stm = ep.readEphysFile(RawDataPath+fname)
    if len(swps)-1 == len(stm):
        swps_new=np.delete(swps,0)
    else:
        continue        
    stiDur = np.sort(stm['duration (s)'].unique())
    stiRate = np.sort(stm['clickRate (Hz)'].unique())   
    stiITD = np.sort(stm['ITD (ms)'].unique())
    stienvITD = np.sort(stm['env ITD (ms)'].unique())
    Fs = swps[0].sampleRate
    nchans = swps_new[0].signal.shape[1]
    nsamples = int(Fs*0.5)
    ntrials = np.shape(np.array(stm[(stm['clickRate (Hz)'] == stiRate[0]) & (stm['duration (s)'] == stiDur[0]) & (stm['ITD (ms)'] == stiITD[0]) & (stm['env ITD (ms)'] == stienvITD[0])].index))[0] 
    signal_arrays = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples,ntrials),dtype = 'float32')
    signal_concatenate_array = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples*ntrials), dtype = 'float32')
    ErrorMark_array = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD)), dtype = 'float32')
    for cc in range(nchans):
        print('Chan'+str(cc+1))
        for ff in range(len(stiRate)):        
            for dd in range(len(stiDur)):
                for ii in range(len(stiITD)):
                    for jj in range(len(stienvITD)):
                        stimParam = [stiRate[ff],stiDur[dd],stiITD[ii],stienvITD[jj]]
                        print(stimParam)
                        stm_select = stm[(stm['clickRate (Hz)'] == stimParam[0]) & (stm['duration (s)'] == stimParam[1]) & (stm['ITD (ms)'] == stimParam[2]) & (stm['env ITD (ms)'] == stimParam[3])]
                        stmIdx = np.array(stm_select.index)
                        if len(stmIdx) < ntrials:
                            ErrorMark_array[cc, ff, dd, ii, jj] = 1
                            print(str(len(stmIdx))+' wrong! uneuqal trial length')                            
                            continue
                        elif len(stmIdx) > ntrials:
                            stmIdx = stmIdx[:ntrials]
                        sampleIdx = int(Fs*stimParam[1])
                        for tt in range(ntrials):
                            signal = swps_new[stmIdx[tt]].signal[:nsamples+100, cc]
                            if tt == 0:
                                Ref = signal[:sampleIdx]
                                Reflen = len(Ref)
                                peakIdxRef = np.argmax(np.correlate(Ref-np.mean(Ref),Ref-np.mean(Ref),'full')[(Reflen-5):(Reflen+5)])
                                
                                signal_arrays[cc, ff, dd, ii, jj, :, tt] = signal[: nsamples]
#                                signal_concatenate = signal[: nsamples]
                            if tt >= 1:
                                Temp = signal[:sampleIdx]
                                Templen = len(Temp)
                                peakIdxtemp = np.argmax(np.correlate((Ref-np.mean(Ref)),(Temp-np.mean(Temp)),'full')[(Templen-5):(Templen+5)])
                                peakdiff = peakIdxRef-peakIdxtemp
                                if peakdiff < 0:
                                    signal = np.concatenate((np.zeros((np.abs(peakdiff)), dtype = 'float32'), signal))
                                    signal_arrays[cc, ff, dd, ii, jj, :, tt] = signal[: nsamples]
                                elif peakdiff >= 0:
                                    signal = signal[peakdiff: peakdiff+nsamples]
                                    signal_arrays[cc, ff, dd, ii, jj, :, tt] = signal[: nsamples]
#                                signal_concatenate = np.concatenate((signal_concatenate, signal[: nsamples]))
#                        signal_concatenate_array[cc, ff, dd, ii, jj, :] = signal_concatenate
    
    plt.figure(figsize=(10,15))
    plt.xticks(np.arange(0, 32, 1))
    plt.subplot(1,2,1)
    plt.title('900pps')
    y = 0
    for dd in range(3):
        for ii in range(3):
            for jj in range(3):
                y = y+1
                for cc in range(32):
                    if ErrorMark_array[cc, 0, dd, ii, jj] == 0:                        
                        plt.plot(cc+1,y, 'ko')
                    else:
                        plt.plot(cc+1,y, 'ro')
    plt.subplot(1,2,2)
    plt.title('4500pps')
    y = 0
    for dd in range(3):
        for ii in range(3):
            for jj in range(3):
                y = y+1
                for cc in range(32):
                    if ErrorMark_array[cc, 1, dd, ii, jj] == 0:                        
                        plt.plot(cc+1,y, 'ko')
                    else:
                        plt.plot(cc+1,y, 'ro')
    plt.savefig(results_path+fname)
    if nchans == 32:
        np.save(results_path+fname+'_OriginSigArray.npy',signal_arrays)
        np.save(results_path+fname+'_Stimulus.npy',stm)
#        IC_dic = {'OriginSig_array': signal_arrays, 'OriginSig_concatenate': signal_concatenate_array, 'Stimulus': stm, 'Fs': Fs}
#        np.save(results_path+fname+'_IC.npy', IC_dic)
    if nchans == 96:
        np.save(results_path+fname+'_IC_OriginSigArray.npy',signal_arrays[:32, :, :, :, :, :, :])
        np.save(results_path+fname+'_AC1_OriginSigArray.npy',signal_arrays[32:64, :, :, :, :, :, :])
        np.save(results_path+fname+'_AC2_OriginSigArray.npy',signal_arrays[64:96, :, :, :, :, :, :])
        np.save(results_path+fname+'_Stimulus.npy',stm)
#        IC_dic = {'OriginSig_array': signal_arrays[:32, :, :, :, :, :, :], 'OriginSig_concatenate': signal_concatenate_array[:32, :, :, :, :, :], 'Stimulus': stm, 'Fs': Fs}
#        np.save(results_path+fname+'_IC.npy', IC_dic)
#        AC_dic_1 = {'OriginSig_array': signal_arrays[32:64, :, :, :, :, :, :], 'OriginSig_concatenate': signal_concatenate_array[32:64, :, :, :, :, :], 'Stimulus': stm, 'Fs': Fs}
#        np.save(results_path+fname+'_AC_1.npy', AC_dic_1)
#        AC_dic_2 = {'OriginSig_array': signal_arrays[64:96, :, :, :, :, :, :], 'OriginSig_concatenate': signal_concatenate_array[64:96, :, :, :, :, :], 'Stimulus': stm, 'Fs': Fs}
#        np.save(results_path+fname+'_AC_2.npy', AC_dic_2)
    