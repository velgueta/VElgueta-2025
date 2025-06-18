
import sys
# The following needs to be installed through: https://github.com/niyiyu/NoisePy4DAS-SeaDAS
sys.path.append("/NoisePy4DAS-SeaDAS/src")
sys.path.append("/NoisePy4DAS-SeaDAS/DASstore")

import os
import gc
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import glob
import numpy as np

from tqdm import tqdm
from obspy import UTCDateTime
import DAS_module

def get_tstamp(fname):
    datestr = fname.split('_')[-3].split('-')
    y = int(datestr[0])
    m = int(datestr[1])
    d = int(datestr[2])
    timestr = fname.split('_')[-2].split('.')
    H = int(timestr[0])
    M = int(timestr[1])
    S = int(timestr[2])
    return UTCDateTime('%04d-%02d-%02dT%02d:%02d:%02d' % (y,m,d,H,M,S))

#%% =================================== parameters ==========================================
original_freq = 200 # Hz
start_time = UTCDateTime('2023-10-02T00:00:00')
end_time = UTCDateTime('2023-10-02T01:00:00')

cc_len = 15                   # correlate length in second
num_sample = int(cc_len*original_freq) # seconds * original sampling rate, time window for correlation in samples

samp_freq = 200                # targeted sampling rate
freq_norm   = 'no'             # 'no' for no whitening, or 'rma' for running-mean average, 'phase_only' for sign-bit normalization in freq domain.
time_norm   = 'no'             # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain
cc_method   = 'xcorr'          # 'xcorr' for pure cross correlation, 'deconv' for deconvolution; FOR "COHERENCY" PLEASE set freq_norm to "rma", time_norm to "no" and cc_method to "xcorr"
smooth_N    = 50               # moving window length for time domain normalization if selected (points)
smoothspect_N  = 50            # moving window length to smooth spectrum amplitude (points)
maxlag      = 5                # lags of cross-correlation to save (sec)

# criteria for data selection
max_over_std = 10 *9               # threshold to remove window of bad signals: set it to 10*9 if prefer not to remove them

n_lag = maxlag * samp_freq * 2 + 1 # lags of cross-correlation to save (samples)

freqmin = 1
freqmax = int(samp_freq/2 - 1)

auto_ch = 1030

prepro_para = {'freqmin':freqmin,
                'freqmax':freqmax,
                'sps':original_freq,
                'npts_chunk':num_sample,
                'nsta':1,
                'cha_list':1,
                'samp_freq':samp_freq,
                'freq_norm':freq_norm,
                'time_norm':time_norm,
                'cc_method':cc_method,
                'smooth_N':smooth_N,
                'smoothspect_N':smoothspect_N,
                'maxlag':maxlag,
                'max_over_std':max_over_std,
                'cc_len':cc_len,
                'n_lag':n_lag,
                }
#%% =================================== inputh and output path ==========================================
data_dir = f'/{start_time.year}/{start_time.month:02d}/*/*'
# Use glob to find all files in the directory and subdirectories
file_path = sorted(glob.glob(data_dir))
file_list = np.array([f_path.split('/')[-1] for f_path in file_path])

acqu_time = np.array([get_tstamp(i) for i in file_list])

mask = (acqu_time >= start_time) & (acqu_time <= end_time)

file_list = np.array(file_path)[mask]

outdir = f'/autocorr/{cc_len}sec_norm/{auto_ch}/'
os.makedirs(outdir, exist_ok=True)


# First, calculate total number of windows
total_windows = 0
samples_per_file = 60 * samp_freq  # adjust as needed

for file in file_list:
    n_windows = samples_per_file // num_sample
    total_windows += n_windows

# Initialize progress bar for total number of windows
pbar = tqdm(total=total_windows)

for file in file_list:

    with h5py.File(file, 'r',locking=False) as f:
        full_data = np.asarray(f['Acquisition']['Raw[0]']['RawData'], dtype='float32')[:samples_per_file, auto_ch:auto_ch+1]

    n_windows = samples_per_file // num_sample
    start_time = get_tstamp(file)

    for w in range(n_windows):
        start_idx = w * num_sample
        end_idx = start_idx + num_sample
        if end_idx > full_data.shape[0]:
            print(f'Window {w} exceeds data length for file {file}. Skipping...')
            pbar.update(1)
            continue

        minute_data = full_data[start_idx:end_idx,:]
        st = start_time + w * (num_sample / samp_freq)
        et = st + (num_sample / samp_freq)

        trace_stdS, dataS = DAS_module.preprocess_raw_make_stat(minute_data, prepro_para)
        white_spect = DAS_module.noise_processing(dataS, prepro_para)
        Nfft = white_spect.shape[1]
        data = white_spect[:, :(Nfft // 2)]
        del dataS, white_spect
        gc.collect()

        ind = np.where((trace_stdS < prepro_para['max_over_std']) & 
                    (trace_stdS > 0) & 
                    (~np.isnan(trace_stdS)))[0]

        if len(ind) == 0:
            pbar.update(1)
            continue

        white_spect = data[ind]
        sfft1 = DAS_module.smooth_source_spect(white_spect, prepro_para)
        corr, tindx = DAS_module.correlate(sfft1, white_spect, prepro_para, Nfft)
        corr = corr/np.var(sfft1)/len(corr)

        filename_stime = st.strftime('%Y%m%dT%H%M%S')
        filename_etime = et.strftime('%Y%m%dT%H%M%S')
        f_name = f'autocorr_{filename_stime}_{filename_etime}.npz'
        np.savez_compressed(outdir + f_name, autocorr=corr[:, int(corr.shape[1] * 0.5):])

        pbar.update(1)
pbar.close()