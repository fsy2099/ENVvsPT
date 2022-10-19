#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:43:22 2022

@author: shiyi
"""
from sys import platform
import numpy as np
import numpy.matlib
import os 
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window
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
def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x

def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.
    `x` and `y` must be one-dimensional numpy arrays with the same length.
    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]
    The return vaue has length 2*maxlag + 1.
    """
    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                    strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)

def wienerfilt1(X,Y,N,Fs):
    Y1 = Y-np.mean(Y)
    X1 = X-np.mean(X)
    H1 = crosscorrelation(Y1,X1,N)
    H = (H1 / np.var(X))/Fs
    
    H = H - np.mean(H[:N])
    H = H[N:]
    return H

def concatenation(X):
    Y = np.reshape(X.T, [1, X.shape[0]*X.shape[1]])
    Y = Y[0]
    return Y
    
def FindPeak2Peak(X):
    if not int(X[0]) == 0:
        X = X-X[0]
    p2p = np.max(X) - np.min(X)
    return p2p

def CalcSF(X, Y, n):
    idx = np.arange(np.argmax(Y[:n])-12, np.argmax(Y[:n])+13)
    SF = FindPeak2Peak(X[idx])/FindPeak2Peak(Y[idx])
    return SF

# N is the kasier window
def CalcCSD(S1, S2, N, Fs): 
    f, H = csd(S1, S2, Fs, window = N, nperseg = len(N))
    H = 10*np.log10(np.abs(H))
    return H

def CalcARR(S1, S2, C1, C2, N, Fs):
    window = get_window(('kaiser', 5), N)
    ARR = CalcCSD(C1, C2, window, Fs)/CalcCSD(S1, S2, window, Fs)
    return ARR

# O is the kasier window order; F is the fundemental frequence, Fs is the sampling rate
def SelectFre(O, F, Fs):
    step = int(np.around(Fs/(2*int(O/2+1))))
    Fre = np.arange(F, int(Fs/2), F)
    idx = np.around(Fre/step)
    idx = [int(x) for x in idx]
    return idx

def CalcPSD(S, N, Fs):
    f, H = welch(S, Fs, window = N, nperseg = len(N))
    H = 10*np.log10(np.abs(H))
    return H

# S1 is the signal trial signal, S2 is the mean across all trial
#def CalcSNR(S1, S2, N, Fs):
#    window = get_window(('kaiser', 5), N)
#    PSD = CalcPSD(S1, window, Fs)
#    CSD = CalcCSD(S1, S2, window, Fs)
#    SNR = PSD-CSD/CSD
#    return SNR

# A is the artifact length
def CalcFFT(S, A, Fs):
    insig = np.concatenate((S, np.zeros((2, int(Fs/2) - Artifact_length))), axis = 1)
    fft = np.abs(np.fft.rfft(insig))
    return fft

def CalcSNR(S, F, Fs):
    fre = np.around(np.linspace(0,Fs/2,len(S)))
    SNR = np.mean(S[np.where((fre<F+10) & (fre>F-10))[0]])/np.mean(S[np.where((fre<F+100) & (fre>F+20))[0]])
    return SNR


#%% 
#position = ['P1']
file_names = os.listdir(Sig_path)
sig_names = [file_name for file_name in file_names if all([x in file_name for x in ["_OriginSigArray.npy"]])][0:1]
Fs = 24414.0625
WienerFilterOrder = 25
nsamples = int(Fs*0.5)
for sig_name in sig_names:    
#    dic = np.load(Sig_path+sig_name, allow_pickle = True).item()
    sig_array = np.load(results_path+sig_name)
#    sig_array = dic['OriginSig_array']
#    sig_concatenate = dic['OriginSig_concatenate']
    ntrials = sig_array.shape[-1]
#    nsamples = sig_array.shape[-2]
#    Fs = dic['Fs']
#    stm = dic['Stimulus']
    stiDur = [0.01, 0.05, 0.2]
    stiRate = [900, 4500]
    stiITD = [-0.1, 0, 0.1]
    stienvITD = [-0.1, 0, 0.1]
    
    reject_mark = np.zeros((32, len(stiDur),len(stiITD),len(stienvITD)),dtype = 'float32')
    H0_array = np.zeros((32,len(stiDur),len(stiITD),len(stienvITD),WienerFilterOrder+1),dtype = 'float32')
    clean_array = np.zeros((32,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples,ntrials),dtype = 'float32')
    SF_array = np.zeros((32, len(stiRate),len(stiDur),len(stiITD),len(stienvITD)), dtype = 'float32')
#    Predict_array = np.zeros((32, len(stiRate),len(stiDur),len(stiITD),len(stienvITD), nsamples))
    fft_array = np.zeros((32, len(stiRate),len(stiDur),len(stiITD),len(stienvITD),int(np.around(Fs/4)), ntrials), dtype = 'float32')
    SNR_array = np.zeros((32, len(stiRate),len(stiDur),len(stiITD),len(stienvITD), ntrials), dtype = 'float32')
    KaiserWinOrder = 500
    for cc in range(32):
        print('Chan'+str(cc+1))
        for dd in range(3):
            for ii in range(3):
                for jj in range(3):
                    StimParam = [stiDur[dd], stiITD[ii], stienvITD[jj]]
                    print(StimParam)
                    Artifact_length = int(np.around(Fs*stiDur[dd]))
                    # 900pps                        
                    sig_9 = concatenation(sig_array[cc, 0, dd, ii, jj, :, :])
                    stim = StimulusData[0, dd, ii, jj, :, 0]
                    stim_9 = np.matlib.repmat(stim,1,ntrials).T[:, 0]
                    stim_9[stim_9 < 0] = 0
                    # 1. calculate the initnal wiener filter kernel(H0)
                    H0 = wienerfilt1(stim_9, sig_9, WienerFilterOrder, Fs/WienerFilterOrder)
                    H0 = H0-H0[0]
                    # 2. use H0 convolve with stimulus signal to get the predicted artifact
                    predict_9 = np.convolve(H0, stim_9)
                    # 3. there is shift between origin signal and predicted artifact sometimes, use correlate method to aline the signal
                    # use the first artifact of first trial
                    # shift predict artifact to match the original signal
                    correlate_9 = np.correlate(sig_9[: Artifact_length], predict_9[: Artifact_length], 'full')
                    max_idx = np.argmax(correlate_9[Artifact_length-5: Artifact_length+5])
                    # ? How sure the maximum peak is in the range(artifact length-+5), may need to optimize
                    shift = max_idx-(5-1)
                    if shift < 0:
                        predict_9 = predict_9[np.abs(shift): len(sig_9)+np.abs(shift)]
                    elif shift > 0:
                        padding = np.zeros((shift))
                        predict_9 = np.concatenate((padding, predict_9[: len(sig_9)-shift]))     
                    # 4. Calculate the scale factor
                    ScaleFactor_9 = CalcSF(np.mean(sig_array[cc, 0, dd, ii, jj, :, :], 1), predict_9, Artifact_length)
                    print('scale factor: '+str(ScaleFactor_9))
                    SF_array[cc, 0, dd, ii, jj] = ScaleFactor_9
                    # 4. evaluate the shape of kernel H0
                    # continuely use the largest stimulus impulse 
                    # calculate the correlation of the original signal&artifact
                    # if the peak of the correlation results is not at idx(Winerfilterorder - 1), H0 unmatched the artifact shape, reject the data
                    peakidx = np.argmax(predict_9[:Artifact_length])
                    sig_peak = sig_array[cc, 0, dd, ii, jj, peakidx-12:peakidx+13, :] - sig_array[cc, 0, dd, ii, jj, peakidx-12, :]                        
                    predict_9 = predict_9*ScaleFactor_9
                    SigArtifact_list = []
                    CleanArtifact_list = []
                    for tt in range(sig_array.shape[-1]):
                        original_signal = sig_peak[:, tt]
                        predict_signal = predict_9[peakidx-12:peakidx+13]
                        clean_signal = original_signal - predict_signal
                        SigArtifact_list.append(np.argmax(np.correlate(predict_signal,original_signal, 'full')))
                        CleanArtifact_list.append(np.argmax(np.correlate(predict_signal,clean_signal, 'full')))
#                            use variance to evaluate how clean the signal is 
#                            x = np.correlate(predict_signal, clean_signal, 'full')
#                            x = x.astype('float64')
#                            x_var = sta.variance(x)
#                            y = np.correlate(predict_signal, original_signal, 'full')
#                            y = y.astype('float64')
#                            y_var = sta.variance(y)
                    H0_array[cc, dd, ii, jj, :] = H0                            
#                    Predict_array[cc, 0, dd, ii, jj, :Artifact_length] = predict_9[:Artifact_length]
                    clean_9 = np.reshape(sig_9-predict_9[:len(sig_9)], (ntrials, nsamples)).T
                    clean_array[cc, 0, dd, ii, jj, :, : ] = clean_9
                    if not len(np.unique(np.array(SigArtifact_list))) == 1 and np.mean(np.array(SigArtifact_list)) == WienerFilterOrder-1:
                        reject_mark[cc, dd, ii, jj] = 1
                        print('H0 unmatched, reject the data')
                        continue
                    if not len(np.unique(np.array(CleanArtifact_list))) >= int(ntrials/3):
                        reject_mark[cc, dd, ii, jj] = 2
                        print('H0 unmatched, reject the data')
                        continue                    
                    # 5. use H0 to do artifact rejection on 4500pps
                    sig_45 = concatenation(sig_array[cc, 1, dd, ii, jj, :, :])
                    stim = StimulusData[1, dd, ii, jj, :, 0]
                    stim_45 = np.matlib.repmat(stim,1,ntrials).T[:, 0]
                    stim_45[stim_45 < 0] = 0
                    predict_45 = np.convolve(H0, stim_45)
                    correlate_45 = np.correlate(sig_45[:Artifact_length], predict_45[:Artifact_length], 'full')
                    idxCorr_45 = np.argmax(correlate_45[Artifact_length-5:Artifact_length+5])
                    shift = idxCorr_45-(5-1)
                    if shift == 0:
                        predict_45 = predict_45
                    elif shift < 0:
                        predict_45 = predict_45[np.abs(shift):len(sig_45)+np.abs(shift)]
                    elif shift > 0:
                        predict_45 = np.concatenate((np.zeros((shift)), predict_45[:len(sig_45)-shift]))
                    ScaleFactor_45 = CalcSF(np.mean(sig_array[cc, 1, dd, ii, jj, :, :], 1), predict_45, Artifact_length)
                    SF_array[cc, 1, dd, ii, jj] = ScaleFactor_45
                    predict_45 = predict_45*ScaleFactor_45
#                    Predict_array[cc, 1, dd, ii, jj, :Artifact_length] = predict_45[:Artifact_length]
                    clean_45 = np.reshape(sig_45-predict_45[:len(sig_45)], (ntrials, nsamples)).T
                    clean_array[cc, 1, dd, ii, jj, :, : ] = clean_45
                    # 6. calculate post artifact rejection SNR to evaluate how clean the signal is
                    for tt in range(ntrials):
                        fft_array[cc, :, dd, ii, jj, :int(np.around(Fs/4)), tt] = CalcFFT(clean_array[cc, :, dd, ii, jj, :Artifact_length, tt], Artifact_length, Fs)
                        SNR_array[cc, 0, dd, ii, jj, tt] = CalcSNR(fft_array[cc, 0, dd, ii, jj, :], 900, Fs)
                        SNR_array[cc, 1, dd, ii, jj, tt] = CalcSNR(fft_array[cc, 1, dd, ii, jj, :], 4500, Fs)
#                    if np.array(np.where(SNR_array[cc, 0, dd, ii, jj, :]>1.5)).shape[1] > int(ntrials/5):
#                        reject_mark[cc, dd, ii, jj] = 3
#                        print('artifact residue, reject the data')
#                        continue
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
    plt.savefig(results_path+sig_name[:-4]+'_reject mark')
    plt.close('all')
    np.save(results_path+sig_name[:-4]+'_RejectMark.npy',reject_mark)
    np.save(results_path+sig_name[:-4]+'_H0.npy',H0_array)
    np.save(results_path+sig_name[:-4]+'_CleanSig.npy',clean_array)
    np.save(results_path+sig_name[:-4]+'_SNR.npy',SNR_array)
    np.save(results_path+sig_name[:-4]+'_ScaleFactor.npy',SF_array)
    np.save(results_path+sig_name[:-4]+'_FFT.npy',fft_array)
#    ArtifactRejection_dic = {'reject_mark': reject_mark, 'H0_array': H0_array, 'clean_array': clean_array, 'SNR_array': SNR_array, 'SF_array': SF_array, 'fft_array': fft_array, 'Fs': Fs, 'sig_array': sig_array}
#    np.save(results_path+sig_name[:-4]+'_ArtifactRejected.npy', ArtifactRejection_dic)
#%%
stmDur = [0.01, 0.05, 0.2]
stmRate = [900, 4500]
stmITD = [-0.1, 0, 0.1]
stmenvITD = [-0.1, 0, 0.1]
reject_type = 2
# 0 is clean, 1 is H0 shape shift, 2 is H0 residual, 3 is SNR residual
reject_idx = np.array(np.where(reject_mark==reject_type))
print(reject_idx.shape)
#%%
[cc, dd, ii, jj] = reject_idx[:, 32]
StimParam = [stiDur[dd], stiITD[ii], stienvITD[jj]]
print(StimParam)
#[cc, dd, ii, jj] = [31, 2, 2, 2]
Artifact_length = int(np.around(Fs*stmDur[dd]))
tt = 9
stim_9 = StimulusData[0, dd, ii, jj, :, 0]
stim_9[stim_9 < 0] = 0
predict = np.convolve(H0_array[cc, dd, ii, jj, :], stim_9)*SF_array[cc, 0, dd, ii, jj]
idx = np.argmax(predict[:Artifact_length])
plt.figure(figsize=(15, 10))
plt.suptitle(sig_name[:-4]+'_900pps_ch'+str(cc)+'_Dur'+str(stmDur[dd])+'_ptITD'+str(stmITD[ii])+'_envITD'+str(stmenvITD[jj])+'_reject'+str(reject_type))

#plt.subplot(2,2,1)
#plt.title('clean signal [: artifact length]')
#clean_sig = clean_array[cc, 0, dd, ii, jj, 2:Artifact_length, :] - clean_array[cc, 0, dd, ii, jj, 2, :]
#plt.plot(clean_sig)

plt.subplot(2,2,1)
plt.title('clean signal [maxidx-100:maxidx+100]')
plt.plot(clean_array[cc, 0, dd, ii, jj, idx-100:idx+100, :]) 

plt.subplot(2,2,2)
original_sig = sig_array[cc, 0, dd, ii, jj, idx-12:idx+13, :] - sig_array[cc, 0, dd, ii, jj, idx-12, :]
predict_sig = predict[idx-12:idx+13]
plt.plot(original_sig[:, tt], 'r-o', label = 'original')
plt.plot(predict_sig, 'b-o', label = 'predict')
plt.legend()
plt.title('original #' + str(tt)+' & predict')

plt.subplot(2,2,3)  
fft_length = fft_array.shape[-2]
fre = np.linspace(0,Fs/2,fft_length)
for x in range(30):
    plt.plot(fre, fft_array[cc, 0, dd, ii, jj, :, x])
    plt.xlim([800, 1000])
    plt.ylim([-0.01, 0.02])
plt.title('FFT 800~1000Hz')

plt.subplot(2,2,4)
plt.plot(SNR_array[cc, 0, dd, ii, jj, :], label = '900pps')
plt.plot(SNR_array[cc, 1, dd, ii, jj, :], label = '4500pps')
plt.legend()
plt.title('SNR ratio')
#plt.subplot(2,2,4)
#original_2 = sig_array[cc, 0, dd, ii, jj, :Artifact_length, :] - sig_array[cc, 0, dd, ii, jj, 0, :]
#predict_2 = predict[:Artifact_length]
#plt.plot(original_2[:, tt], 'r-o', label = 'original')
#plt.plot(predict_2, 'b-o', label = 'predict')
#plt.legend()
#- clean_array[cc, 0, dd, ii, jj, idx-12, :])
#plt.legend()
#%%
fft_length = fft_array.shape[-2]
fre = np.linspace(0,Fs/2,fft_length)
plt.figure()
for x in range(30):
    plt.plot(fre, fft_array[cc, 0, dd, ii, jj, :, x])
    plt.xlim([800, 1000])
    plt.ylim([-0.01, 0.02])
#%%
#plt.figure(figsize=(10,15))
#plt.xticks(np.arange(0, 32, 1))
#plt.title('reject mark')
##plt.subplot(1,2,1)
#y = 0
#for dd in range(3):
#    for ii in range(3):
#        for jj in range(3):
#            y = y+1
#            for cc in range(32):
#                if reject_mark[cc, dd, ii, jj] == 0:                        
#                    plt.plot(cc+1,y, 'ko')
#                else:
#                    plt.plot(cc+1,y, 'ro')
##%%
#plt.figure(figsize=(10,15))
#y = 0
#for dd in range(3):
#    for ii in range(3):
#        for jj in range(3):
#            y = y+1
#            for cc in range(32):
#                if reject_mark[cc, dd, ii, jj] == 0:                        
#                    plt.plot(cc+1,y, 'ko')
#                else:
#                    plt.plot(cc+1,y, 'ro')
##%%
#np.save(results_path+fname+'_reject_channel.npy',reject_mark)
#np.save(results_path+fname+'_H0.npy',H0_array)
#np.save(results_path+fname+'_clean.npy',clean_array)
#np.save(results_path+fname+'_predict.npy',Predict_array)
#np.save(results_path+fname+'_FilterRatio.npy',FilterRatio_array)
##%%
#cc = 17
#dd = 2
#ii = 0
#jj = 0
#plt.figure
#plt.subplot(1,2,1)
#plt.plot(clean_array[cc, 0, dd, ii, jj, :5000, :5])
#plt.title('900pps')
#plt.subplot(1,2,2)
#plt.plot(clean_array[cc, 1, dd, ii, jj, :5000, :5])
#plt.title('4500pps')                                
##%%
#                        clean_9 = np.reshape(sig_9-predict_scaled[:len(sig_9)], (ntrials, nsamples)).T
#                        sig_1 = np.mean(sig_array[cc, 0, dd, ii, jj, :, :], 1)
#                        clean_1 = np.mean(clean_9, 1)
#                        Fre_idx = SelectFre(KaiserWinOrder, 900, Fs)
#                        ARR_list = []
#                        step = 50
#                        for tt in range(ntrials):
#                            ARR_trial = CalcARR(sig_1, sig_array[cc, 0, dd, ii, jj, :, tt], clean_1, clean_9[:, tt], KaiserWinOrder, Fs)
#                            ARR_list.append(np.mean(ARR_trial[Fre_idx]))
#                        ARR_pre = np.mean(np.array(ARR_list))                        
#                        
#                        scaleFactor = 0
#                        ARR = []
#                        for xx in range(15):
#                            ARR_list = []
#                            scaleFactor = scaleFactor+step
#                            predict_post = predict/scaleFactor
#                            clean_post = np.reshape(sig_9-predict_post[:len(sig_9)], (ntrials, nsamples)).T
#                            clean_1 = np.mean(clean_post, 1)
#                            for tt in range(ntrials):
#                                ARR_trial = CalcARR(sig_1, sig_array[cc, 0, dd, ii, jj, :, tt], clean_1, clean_post[:, tt], KaiserWinOrder, Fs)
#                                ARR_list.append(np.mean(ARR_trial[Fre_idx]))
#                            ARR.append(np.mean(np.array(ARR_list)))
##%%                        
#                        if ARR_post>ARR_pre:
#                            while ARR_post>ARR_pre:
#                                ARR_pre = ARR_post
#                                scaleFactor_post = scaleFactor_post+step
#                                predict_post = predict/scaleFactor_post
#                                clean_post = np.reshape(sig_9-predict_post[:len(sig_9)], (ntrials, nsamples)).T
#                                clean_1 = np.mean(clean_post, 1)
#                                for tt in range(ntrials):
#                                    ARR_trial = CalcARR(sig_1, sig_array[cc, 0, dd, ii, jj, :, tt], clean_1, clean_post[:, tt], KaiserWinOrder, Fs)
#                                    ARR_list.append(np.mean(ARR_trial[Fre_idx]))
#                                ARR_post = np.mean(np.array(ARR_list))
#                                print(scaleFactor_post)
#                        else:
#                            while ARR-post<ARR-pre:
#                                ARR_pre = ARR_post
#                                scaleFactor_post = scaleFactor_post-step
#                                clean_post = np.reshape(sig_9-predict_post[:len(sig_9)], (ntrials, nsamples)).T
#                                clean_1 = np.mean(clean_post, 1)
#                                for tt in range(ntrials):
#                                    ARR_trial = CalcARR(sig_1, sig_array[cc, 0, dd, ii, jj, :, tt], clean_1, clean_post[:, tt], KaiserWinOrder, Fs)
#                                    ARR_list.append(np.mean(ARR_trial[Fre_idx]))
#                                ARR_post = np.mean(np.array(ARR_list))
#                            print(scaleFactor)
#                                
#    #%%
#position = ['P1']
#file_names = os.listdir(Sig_path)
#WienerFilterOrder = 20
#for pp in position:    
#    sig_name = [file_name for file_name in file_names if all([x in file_name for x in [pp, "_IC.npy"]])]
#    for x in sig_name:
#        dic = np.load(Sig_path+x, allow_pickle = True).item()
#        sig_array = dic['OriginSig_array']
#        sig_concatenate = dic['OriginSig_concatenate']
#        ntrials = sig_array.shape[-1]
#        nsamples = sig_array.shape[-2]
#        Fs = dic['Fs']
#        stm = dic['Stimulus']
#        stiDur = np.sort(stm['duration (s)'].unique())
#        stiRate = np.sort(stm['clickRate (Hz)'].unique())   
#        stiITD = np.sort(stm['ITD (ms)'].unique())
#        stienvITD = np.sort(stm['env ITD (ms)'].unique())
#        
#        H0_array = np.zeros((32,len(stiDur),len(stiITD),len(stienvITD),WienerFilterOrder+1),dtype = 'float32')
#        clean_array = np.zeros((32,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples,ntrials),dtype = 'float32')
#        Predict_array = np.zeros((32, len(stiRate),len(stiDur),len(stiITD),len(stienvITD), nsamples))
#        FilterRatio_array = np.zeros((32, len(stiDur),len(stiITD),len(stienvITD)))
#        ARR_array = np.zeros((32, len(stiDur),len(stiITD),len(stienvITD)))
#        kaiser_window = get_window(('kaiser', 5), 500)
#        for cc in range(32):
#            print('Chan'+str(cc+1))
#            for dd in range(3):
#                for ii in range(3):
#                    for jj in range(3):
#                        Artifact_length = int(np.around(Fs*stiDur[dd]))
#                        # 900pps                        
#                        sig_9 = sig_concatenate[cc, 0, dd, ii, jj, :]
#                        stim = StimulusData[0, dd, ii, jj, :, 0]
#                        stim_9 = np.matlib.repmat(stim,1,ntrials).T[:, 0]
#                        stim_9[stim_9 < 0] = 0
#                        # find the best filter ratio
#                        FactorStep = 0.01
#                        ARR_pre = 0
#                        for nn in range(500):
#                            FilterRatio = 2.5+nn*FactorStep
#                            H0 = wienerfilt1(stim_9, sig_9, WienerFilterOrder, Fs/FilterRatio*WienerFilterOrder) 
#                            H0 = H0-H0[0]
#                            Predict_9 = np.convolve(H0, stim_9)
#                            clean = np.reshape(sig_9-Predict_9[:len(sig_9)], (ntrials, nsamples)).T
#                            
#                            sig_1 = np.mean(sig_array[cc, 0, dd, ii, jj, :, :], 1)
#                            clean_1 = np.mean(clean, 1)
#                            ARR = []
#                            for tt in range(ntrials):
#                                sig_2 = sig_array[cc, 0, dd, ii, jj, :, tt]
#                                clean_2 = clean[:, tt]             
#                                f, sig_csd_pre = csd(sig_1, sig_2, Fs, window = kaiser_window, nperseg = len(kaiser_window))
#                                sig_csd_pre = 10*np.log10(np.abs(sig_csd_pre))
#                                f, sig_csd_post = csd(clean_1, clean_2, Fs, window = kaiser_window, nperseg = len(kaiser_window))
#                                sig_csd_post = 10*np.log10(np.abs(sig_csd_post))
#                                ARR.append(sig_csd_post/sig_csd_pre)
#                            ARR = np.array(ARR).T
#                            step = int(np.around(Fs/(2*len(sig_csd_pre))))
#                            Fre9 = np.arange(900, int(Fs/2), 900)
#                            Fre_idx = Fre9/step
#                            ARR_selected = np.zeros((len(Fre_idx), ntrials))
#                            for ff in range(len(Fre_idx)):
#                                fre = int(np.around(Fre_idx[ff]))
#                                ARR_selected[ff, :] = ARR[fre]
#                            ARR_post = np.mean(np.mean(ARR_selected, 0))
#                            if ARR_post-ARR_pre > 0:
#                                ARR_pre = ARR_post
#                            else:
#                                break
#                        ARR_array[cc, dd, ii, jj] = ARR_pre
#                        FilterRatio = 2.5+(nn-1)*FactorStep
#                        print(FilterRatio)
#                        FilterRatio_array[cc, dd, ii, jj] = FilterRatio
#                        H0 = wienerfilt1(stim_9, sig_9, WienerFilterOrder, Fs/FilterRatio*WienerFilterOrder) 
#                        H0= H0-H0[0]
#                        H0_array[cc, dd, ii, jj, :] = H0
#                        Predict_9 = np.convolve(H0, stim_9)
#                        Predict_array[cc, 0, dd, ii, jj, :Artifact_length] = Predict_9[:Artifact_length]
#                        clean = np.reshape(sig_9-Predict_9[:len(sig_9)], (ntrials, nsamples)).T
#                        clean_array[cc, 0, dd, ii, jj, :, :] = clean
#                        
#                        #4500pps
#                        sig_45 = sig_concatenate[cc, 1, dd, ii, jj, ]
#                        stim = StimulusData[1, dd, ii, jj, :, 0]
#                        stim_45 = np.matlib.repmat(stim,1,ntrials).T[:, 0]
#                        stim_45[stim_45 < 0] = 0
#                        predict_45 = np.convolve(H0, stim_45)
#                        correlate_45 = np.correlate(sig_45[:Artifact_length], Predict_45[:Artifact_length], 'full')
#                        idxCorr_45 = np.argmax(correlate_45[Artifact_length-5:Artifact_length+5])
#                        shift = idxCorr_45-(5-1)
#                        if shift == 0:
#                            Predict_45 = Predict_45
#                        elif shift < 0:
#                            Predict_45 = Predict_45[np.abs(shift):len(sig_45)+np.abs(shift)]
#                        elif shift > 0:
#                            Predict_45 = np.concatenate((np.zeros((shift)), Predict_45[:len(sig_45)-shift]))
#                        Predict_array[cc, 1, dd, ii, jj, :Artifact_length] = Predict_45[:Artifact_length]
#                        clean_45 = np.reshape(sig_45-Predict_45[:len(sig_45)], (ntrials, nsamples)).T
#                        clean_array[cc, 1, dd, ii, jj, :, : ] = clean_45
#        dic = {'H0': H0_array, 'clean' : clean_array, 'predict': Predict_array, 'filterRatio': FilterRatio_array, 'ARR_Ratio': ARR_array}
#        np.save(results_path+x[:-7]+'_ArtifactRejection.npy', dic)    
##%%
#[cc, ff, dd, ii, jj, tt] = [5, 1, 2, 1, 1, 10]
#Artifact_length = int(Fs*stiDur[dd])
#padding = np.zeros((int(Fs/2)-Artifact_length))
#clean_fft = np.abs(np.fft.rfft(np.concatenate((clean_array[cc, ff, dd, ii, jj, :Artifact_length, tt]-abs(clean_array[cc, ff, dd, ii, jj, Artifact_length, tt]), padding), 0)))
#sig_fft = np.abs(np.fft.rfft(np.concatenate((sig_array[cc, ff, dd, ii, jj, :Artifact_length, tt]-abs(sig_array[cc, ff, dd, ii, jj, Artifact_length, tt]), padding), 0)))
#fre = np.linspace(0,Fs/2,len(sig_fft))
#plt.figure()
#plt.subplot(2,2,1)
#plt.plot(sig_array[cc, ff, dd, ii, jj, :5000, :])
#plt.subplot(2,2,2)
#plt.plot(fre, sig_fft)
#plt.subplot(2,2,3)
#plt.plot(clean_array[cc, ff, dd, ii, jj, :5000, :])
#plt.subplot(2,2,4)
#plt.plot(fre, clean_fft)
#
#
##%%
#position = ['P1']
#file_names = os.listdir(Sig_path)
#WienerFilterOrder = 20
#for pp in position:    
#    sig_name = [file_name for file_name in file_names if all([x in file_name for x in [pp+"_IC"]])]
#    for x in sig_name:
#        dic = np.load(Sig_path+x, allow_pickle = True).item()
#        sig_array = dic['OriginSig_array']
#        sig_concatenate = dic['OriginSig_concatenate']
#        ntrials = sig_array.shape[-1]
#        nsamples = sig_array.shape[-2]
#        Fs = dic['Fs']
#        stm = dic['Stimulus']
#        stiDur = np.sort(stm['duration (s)'].unique())
#        stiRate = np.sort(stm['clickRate (Hz)'].unique())   
#        stiITD = np.sort(stm['ITD (ms)'].unique())
#        stienvITD = np.sort(stm['env ITD (ms)'].unique())
#        
#        H0_array = np.zeros((32,len(stiDur),len(stiITD),len(stienvITD),WienerFilterOrder+1),dtype = 'float32')
#        clean_array = np.zeros((32,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples,ntrials),dtype = 'float32')
#        for cc in range(32):
#            print('Chan'+str(cc+1))
#            for dd in range(3):
#                for ii in range(3):
#                    for jj in range(3):
#                        Artifact_length = int(Fs*stiDur[dd])
#                        padding = np.zeros((int(Fs/2)-Artifact_length))
#                        sig_9 = sig_concatenate[cc, 0, dd, ii, jj, :]
#                        stim = StimulusData[0, dd, ii, jj, :, 0]
#                        stim_9 = np.matlib.repmat(stim,1,ntrials).T[:, 0]
#                        stim_9[stim_9 < 0] = 0
#                        # find the beat filter ratio
#                        FactorStep = 0.01
#                        SNR_pre = 10
#                        SNR_array = []
#                        for nn in range(500):
#                            FilterRatio = 1+nn*FactorStep
#                            print(FilterRatio)
#                            H0 = wienerfilt1(stim_9, sig_9, WienerFilterOrder, Fs/FilterRatio*WienerFilterOrder) 
#                            H0 = H0-H0[0]
#                            Predict = np.convolve(H0, stim_9)
#                            clean = np.reshape(sig_9-Predict[:len(sig_9)], (ntrials, nsamples)).T
#                            SNR_trials = []
#                            for tt in range(ntrials):                                
#                                sig_fft = np.abs(np.fft.rfft(np.concatenate((sig_array[cc, 0, dd, ii, jj, :, tt], padding), 0)))
#                                clean_fft = np.abs(np.fft.rfft(np.concatenate((clean[:, tt], padding), 0)))
#                                fre = np.linspace(0,Fs/2,len(sig_fft))
#                                f_step = int(np.around(Fs/(2*len(sig_fft))))
#                                f_idx = np.arange(900, int(Fs/2), 900)/f_step
#                                SNR_fre = []
#                                for ff in range(len(f_idx)):
#                                    f = int(np.around(f_idx[ff]))
#                                    SNR = np.mean(clean_fft[f-20:f+20])/np.mean(clean_fft[f+200:f+400])                                
#                                    SNR_fre.append(SNR)
#                                SNR_trials.append(max(np.array(SNR_fre)))
#                            SNR_post = np.min(np.array(SNR_trials))
#                            SNR_array.append(SNR_post)
#                            if SNR_post-SNR_pre < 0:
#                                SNR_pre = SNR_post
#                            else:
#                                break
#                        FilterRatio = 1+(nn-1)*FactorStep
#                        H0 = wienerfilt1(stim_9, sig_9, WienerFilterOrder, Fs/FilterRatio*WienerFilterOrder) 
#                        H0 = H0-H0[0]
#                        Predict = np.convolve(H0, stim_9)
#                        clean = np.reshape(sig_9-Predict[:len(sig_9)], (ntrials, nsamples)).T
#                        Predict_temp = Predict-np.abs(sig_9[0])
#                        clean_temp = np.reshape(sig_9-Predict_temp[:len(sig_9)], (ntrials, nsamples)).T
##%% 8/30 re-write base on Fei0817
#position = ['P1']
#file_names = os.listdir(Sig_path)
#WienerFilterOrder = 25
#for pp in position:    
#    sig_name = [file_name for file_name in file_names if all([x in file_name for x in [pp+"_IC"]])]
#    # ? bugs, need to be fixed
#    for x in sig_name:
#        dic = np.load(Sig_path+x, allow_pickle = True).item()
#        sig_array = dic['OriginSig_array']
#        sig_concatenate = dic['OriginSig_concatenate']
#        ntrials = sig_array.shape[-1]
#        nsamples = sig_array.shape[-2]
#        Fs = dic['Fs']
#        stm = dic['Stimulus']
#        stiDur = np.sort(stm['duration (s)'].unique())
#        stiRate = np.sort(stm['clickRate (Hz)'].unique())   
#        stiITD = np.sort(stm['ITD (ms)'].unique())
#        stienvITD = np.sort(stm['env ITD (ms)'].unique())
#        
#        H0_array = np.zeros((32,len(stiDur),len(stiITD),len(stienvITD),WienerFilterOrder+1),dtype = 'float32')
#        clean_array = np.zeros((32,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples,ntrials),dtype = 'float32')        
#        for cc in range(32):
#            print('Chan'+str(cc+1))
#            for dd in range(3):
#                for ii in range(3):
#                    for jj in range(3):
#                        Artifact_length = int(Fs*stiDur[dd])
#                        sig_9 = sig_concatenate[cc, 0, dd, ii, jj, :]
#                        stim = StimulusData[0, dd, ii, jj, :, 0]
#                        stim_9 = np.matlib.repmat(stim,1,ntrials).T[:, 0]
#                        stim_9[stim_9 < 0] = 0
#                        # 1. calculate the wiener filter kernel(H0)
#                        H0 = wienerfilt1(stim_9, sig_9, WienerFilterOrder, Fs/WienerFilterOrder)
#                        H0 = H0-H0[0]
#                        H0_array[cc,dd,ii,jj,:] = H0
#                        # 2. use H0 convolve with stimulus signal to get the predicted artifact
#                        predict = np.convolve(H0, stim_9)
#                        # 3. there is shift between origin signal and predicted artifact sometimes, use correlate method to aline the signal
#                        # use the first artifact of first trial
#                        # shift predict artifact to match the original signal
#                        correlate_9 = np.correlate(sig_9[: Artifact_length], predict[: Artifact_length])
#                        max_idx = np.argmax(correlate_9[Artifact_length-5: Artifact_length+5])
#                        # ? How sure the maximum peak is in the range(artifact length-+5), may need to optimize
#                        shift = max_idx-(5-1)
#                        if shift < 0:
#                            predict = predict[np.abs(shift): len(sig_9)+np.abs(shift)]
#                        elif shift > 0:
#                            padding = np.zeros((shift))
#                            predict = np.concatenate((padding, predict[: len(sig_9)-shift]))
#                        # 4. use average of the largest stimulus impulse across all trials and the artifact roughly calculate the scale factor
#                        peakidx = np.argmax(predict[:Artifact_length])
#                        predict_peak = predict[peakidx-12:peakidx+13]
#                        predict_peaktopeak = np.max(predict_peak) - np.min(predict_peak)
#                        sig_peak = np.mean(sig_array[cc, 0, dd, ii, jj, peakidx-12:peakidx+13, :] - sig_array[cc, 0, dd, ii, jj, peakidx-12, :], 1)
#                        sig_peaktopeak = np.max(sig_peak) - np.min(sig_peak)
#                        scaleFactor = predict_peaktopeak/sig_peaktopeak
#                        predict_peak = predict_peak/scaleFactor
#                        # 5. evaluate the shape of kernel H0
#                        # continuely use the largest stimulus impulse 
#                        # calculate the correlation of the original signal&artifact, clean signal&artifact, clean signal&original signal 
#                        # 
#                        SigArtiPeak_list = []
#                        CleanArtiPeak_list = []
#                        SigCleanPeak_list = []
#                        for tt in range(sig_array.shape[-1]):
#                            sig_peakSingle = sig_peak[:, tt]
#                            clean_peak = sig_peakSingle-predict_peak
#                            sig_artifact = np.correlate(predict_peak, sig_peaksingle, 'full')
#                            clean_artifact = np.correlate(predict_peak, clean_peak, 'full')                            
#                            sig_clean = np.correlate(sig_peaksingle, clean_peak, 'full')
#                            SigArtiPeak_list.append(np.argmax(np.abs(sig_artifact)))
#                            CleanArtiPeak_list.append(np.argmax(np.abs(clean_artifact)))
#                            SigCleanPeak_list.append(np.argmax(np.abs(sig_clean)))
#                        SigArtiPeak_idx = np.unique(np.array(SigArtiPeak_list))
#                        CleanArtiPeak_array = np.unique(np.array(CleanArtiPeak_list))
#                        SigCleanPeak_array = np.unique(np.array(SigCleanPeak_list))
#                        if len(SigArtiPeak_idx) == 1 and SigArtiPeak_idx == WienerFilterOrder - 1:
#                            if len(CleanArtiPeak_array) >= 5:
#                                clean_array = sig_9 - 
