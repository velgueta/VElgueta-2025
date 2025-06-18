import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import sys
import glob
import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, detrend # change from filtfilt to sosfiltfilt
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import datetime
import time
from tqdm import tqdm

def get_tstamp(fname):
    datestr = fname.split('_')[1].split('-')
    y = int(datestr[0])
    m = int(datestr[1])
    d = int(datestr[2])
    timestr = fname.split('_')[2].split('.')
    H = int(timestr[0])
    M = int(timestr[1])
    S = int(timestr[2])
    return UTCDateTime('%04d-%02d-%02dT%02d:%02d:%02d' % (y,m,d,H,M,S))

start_timer = time.time()

#=============================get file path and sort by time==========================
t_start = UTCDateTime(2024, 1, 23, 0, 0, 0)
t_end = UTCDateTime(2024, 1, 24, 0, 0, 0)

fdir = f'/{t_start.year}/{t_start.month:02}/{t_start.day:02}/'

flist = np.array([f for f in os.listdir(fdir)]) # if os.path.isfile(os.path.join(fdir, f))
ftime = np.array([get_tstamp(fname) for fname in flist]) # UTCDateTimes from file names
index = np.argsort(np.array(ftime)-ftime[0])
flist = flist[index]
ftime = ftime[index]

# Create a mask to filter elements within the specified time range
mask = np.logical_and(ftime >= t_start, ftime <= t_end)

# Apply the mask to filter the array
flist = flist[mask] # only file names within the time range
ftime = ftime[mask]

fdir = fdir
flist = [os.path.join(fdir, fname) for fname in flist]
nf = len(flist)

#=============================correlation parameter==========================
# dictionary for the Xcorr job.
fmin = 0.5 # minimal frequency
fmax = 15. # maximal frequency
whiten = True # whiten the data 
onebit = True # one bit normalization
nns = 3000 # number of samples for the xcorr.
srcx = np.arange(100, 3200+1, 1) # channel source location
n_cha = 100 # number of channels outside of source channels to be used as receivers (int)

pdict = {'srcx': srcx, 'n_cha': n_cha, 'nns': nns, 'fmin': fmin, 'fmax': fmax, 'whiten': whiten, 'onebit': onebit}

#=============================interogator parameters==========================
fs = 200. # sample rate
nx = 4494 # number of channels
ns = 12000 # number of samples 
ns_tolerance = int(0.005 * ns) # tolerance for number of samples, 0.5% of ns

with h5py.File(os.path.join(fdir, flist[0]), 'r') as fp:
    dx = fp['Acquisition'].attrs['SpatialSamplingInterval']
    # fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']
    # nx = fp['Acquisition']['Raw[0]'].attrs['NumberOfLoci']
    # ns = len(fp['Acquisition']['Raw[0]']['RawDataTime'][:])  # number of samples

nsrc = len(srcx) # number of sources
min_cha = max(0, min(srcx) - n_cha) # min channel
max_cha = min(nx-1, max(srcx) + n_cha) # max channel
recx = np.arange(min_cha, max_cha+1) # array of receiver channels
nrec = recx.shape[0] # number of channels
nw = nns//2 + 1 # number of sample to one side in time -> number of freq bins after fft
nwin = int(ns//nns) # number of full cc windows per file
xc_arr = np.zeros((nsrc, n_cha*2+1, nw), dtype=np.complex_) # initzalise matrix shape(channels, samples on one side)

sos = butter(4, (fmin/(fs*0.5), fmax/(fs*0.5)),'bandpass', output='sos') # define banpass filter, decay, fmin, fmax

try:
    for fname in tqdm(flist):
        try:
            with h5py.File(fname, 'r') as fp:
                fs_check = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate'] # sampling rate
                nx_check = fp['Acquisition']['Raw[0]'].attrs['NumberOfLoci'] # number of locations
                ns_check = len(fp['Acquisition']['Raw[0]']['RawDataTime'][:])  # number of samples

                if fs == fs_check and nx == nx_check and abs(ns - ns_check) <= ns_tolerance:
                    data = np.asarray(fp['Acquisition']['Raw[0]']['RawData'], dtype='float32')[:,min_cha:max_cha+1].T # read data, cut to recx range
                    if ns_check < ns: # samples missing --> pad with zeros
                        pad_width = ((0, 0), (0, ns - ns_check))  # pad time axis only
                        data = np.pad(data, pad_width, mode='constant', constant_values=0)
                        # print(f'Warning: file {fname} is short by {ns - ns_check} samples, padded with zeros.')
                    elif ns_check > ns: # samples too long --> truncate
                        data = data[:, :ns]  # truncate extra samples if slightly too long
                        # print(f'Note: file {fname} has {ns_check - ns} extra samples, truncated to {ns}.')
                else:
                    print('\nError: file parameters do not match')
                    continue
        except Exception as e:
            print(f'\nError reading file {fname}: {e}')
            continue

        for iwin in range(nwin): # loop over time window indeces per minute
            try:
                data_cut = data[:, iwin*nns:(iwin+1)*nns] # cut the data in time to the cc window lenght (nns)
                data_cut = detrend(data_cut, axis=1, type = 'constant') # linear, constant detrend
                data_cut *= np.tile(np.hamming(nns), (nrec, 1)) # multiply hamming window length nns with all receivers
                data_cut = sosfiltfilt(sos, data_cut, axis=1) # bandpass filter data
                sp = np.fft.rfft(data_cut, axis=1) # fourie transformation, (nns/2)+1 frequency bands

                if whiten:
                    #=============================whiten==========================
                    i1 = int(np.ceil(fmin/(fs/nns))) # index of fmin after fft
                    i2 = int(np.ceil(fmax/(fs/nns))) # index of fmax after fft
                    sp[:, i1:i2] = np.exp(1j*np.angle(sp[:,i1:i2])) # r = 1 for frequencies fmin-fmax
                    sp[:, :i1] = np.cos(np.linspace(np.pi*0.5, np.pi, i1))**2 *\
                                            np.exp(1j*np.angle(sp[:,:i1])) # r <= 1 for frequencies <fmin
                    sp[:, i2:] = np.cos(np.linspace(np.pi,np.pi*0.5,nw-i2))**2 *\
                                        np.exp(1j*np.angle(sp[:,i2:]))  # r <= 1 for frequencies >fmax
                    
                if onebit:
                    #=============================onebit==========================
                    data_cut = np.fft.irfft(sp, axis=1) # inverse fourie transform (back in time domain)
                    data_cut = np.sign(data_cut) # one-bit normalization (amplitudes either 1 or -1)
                    sp = np.fft.rfft(data_cut, axis=1) # fourie transformation, (nns/2)+1 frequency bands

            except Exception as e:
                print(f'\nError in preprocessing window {iwin} of file {fname}: {e}')
                continue

    # HERE WE NEED TO SELECT THE DATA
            for idx, src in enumerate(srcx):
                try:
                    srcid = np.where(recx == src)[0][0] # this gives the index of the source channel for the cutted data
                    recx_full = np.arange(src - n_cha, src + n_cha + 1) # Fixed receiver range: always 2*n_cha + 1
                    valid_mask = (recx_full >= 0) & (recx_full < nx) # Identify valid receiver channels (within cable)
                    valid_recx = recx_full[valid_mask] # Valid receiver channels
                    valid_recids = np.searchsorted(recx, valid_recx) # Find indices in recx for valid receivers
                    target_ids = np.where(valid_mask)[0] # Find target indices in output array where these will be written

                    # Perform correlation only on valid receiver indices
                    xc_arr[idx, target_ids, :] += sp[valid_recids,:] * np.tile(np.conj(sp[srcid,:]), (len(valid_recids), 1))
                    
                except Exception as e:
                    print(f'\nError in correlation (src={src}) in file {fname}: {e}')
                    continue
    print('\nIntermediate elapsed time: %.2f minutes' % ((time.time() - start_timer) / 60))
    #%% ====================== Final normalization and saving ======================
    try:
        xc_arr/= nf * nwin # normalization, divide the stack by number of time windows in total
        mask = np.any(xc_arr != 0, axis=2)  # shape (nsrc, nrec), True for valid receivers
        mean_vals = np.sum(xc_arr, axis=1, keepdims=True) / np.maximum(mask.sum(axis=1, keepdims=True), 1)[..., None] # mean values for each source channel
        xc_arr -= mean_vals * mask[..., None] # subtract mean values from each source channel

        data_xc = np.fft.irfft(xc_arr, axis=2)# inverse fourie transform (back in time domain), 0 lag at the beginning
        data_xc = np.roll(data_xc, -nns//2, axis=2) # roll the data to have 0 lag in the middle
        data_xc = sosfiltfilt(sos, data_xc, axis=2) # bandpass filter correlations

        offset = np.linspace(-(n_cha*dx), n_cha*dx, data_xc.shape[1], dtype=np.float32) # offset in meters
        lags = np.linspace(-(nns*0.5/fs), nns*0.5/fs, data_xc.shape[2], dtype=np.float32) # lag time in seconds

        np.savez_compressed(f'/x_corr/{t_start.strftime("%Y%m%d_%H%M%S")}.npz',
                            pdict=pdict, offset=offset, lags=lags, data_xc=data_xc)

    except Exception as e:
        print(f'\nError in post-processing or saving final results: {e}')
        np.savez_compressed(f'/x_corr/FAILED_raw_{t_start.strftime("%Y%m%d_%H%M%S")}.npz',
                            pdict=pdict, xc_arr=xc_arr)

except Exception as e:
    print(f'\nProcessing failed at file {fname}: {e}')
    np.savez_compressed(f'/x_corr/FAILED_partial_{t_start.strftime("%Y%m%d_%H%M%S")}.npz',
                        pdict=pdict, xc_arr=xc_arr)
finally:
    print('\nElapsed time: %.2f minutes' % ((time.time() - start_timer) / 60))