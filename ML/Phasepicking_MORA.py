"""Example DAS waveform-to-pick workflow; not the final event catalog.

Step 1: edit the helper-module paths, waveform directory, output directory and
date/hour loop below. This preserved example processes two December 2023 hours.
Step 2: load six pretrained EQTransformer models; no local training is performed.
Step 3: filter and normalize each minute, then downsample to the picker rate.
Step 4: apply ELEP to every fourth channel and export candidate P/S arrival times.
Step 5: run das_event_catalog.py separately to evaluate S-pick coherence/moveout.

Requires the external das_util, detect_util and denodas_model modules as well as
ELEP, SeisBench and the imported scientific packages. Model downloads may require
network access. Create the output directory first. No denoiser is executed: its
initialization is commented out. Server paths are examples, not portable defaults.
The exported sampling-rate metadata remains 200 although picker input is 100 Hz;
verify this convention before interpreting sample-based fields downstream.
"""

import sys
sys.path.append('/home/velgueta/notebooks/RainierDas/Qibin-repo/Shi_etal_2023_denoiseDAS/src/denoiser/')
sys.path.append('/home/velgueta/notebooks/RainierDas/Qibin-repo/Shi_etal_2023_denoiseDAS/src/ensemble_picker/')

import gc
import glob
import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import seisbench.models as sbm
from ELEP.elep.ensemble_coherence import ensemble_semblance
from ELEP.elep.trigger_func import picks_summary_simple

from datetime import datetime, timedelta
from das_util import *
from detect_util import *
from denodas_model import unet
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.signal import filtfilt, butter
from scipy.interpolate import interp1d

import obspy
from obspy import UTCDateTime


## Paths and formats

#filepath = '/data/data4/veronica-scratch-rainier/picking-denoising/2024-08-28_day-10chan/' #create my folder for save my picks
filepath = '/data/data4/veronica-scratch-rainier/picking-denoising/avalanche-picks/' #create my folder for save my picks
rainier_dir = '/1-fnp/petasaur/p-wd07/rainier/'    #just go to my data (avalanche)
#rainier_dir = '/1-fnp/petasaur/p-wd15/rainier-10-14-2023-drive1/'    #just go to my data (swarm)
model_dir = '../models/checkpoint_noatt_LRdecays0.8_mask0.5_raw2raw_chmax4500.pt' #denoiser
format_part = 'decimator_%Y-%m-%d_%H.??.??_UTC.h5'

## Download the models from Qibin repo

# filtering the data

############################################# Loop for hours

### Bandpass filter
b, a = butter(4, (0.5, 49), fs=200, btype='bandpass') #0.5 to 49 to avoid aliasing

### Initialize the U-net, denoiser 
devc = try_gpu(i=0)
#model_1 = unet(1, 16, 1024, factors=(5, 3, 2, 2), use_att=False)
#model_1 = nn.DataParallel(model_1, device_ids=[0])
#model_1.to(devc)
### Load the pretrained weights
#model_1.load_state_dict(torch.load(model_dir))
#model_1.eval() 

### ELEP models, phase picker
pn_ethz_model = sbm.EQTransformer.from_pretrained("ethz")
pn_neic_model = sbm.EQTransformer.from_pretrained("neic")
pn_scedc_model = sbm.EQTransformer.from_pretrained("scedc")
pn_stead_model = sbm.EQTransformer.from_pretrained("stead")
pn_geofon_model = sbm.EQTransformer.from_pretrained("geofon")
pn_instance_model = sbm.EQTransformer.from_pretrained("instance")
pn_ethz_model.to(devc); pn_ethz_model.eval()
pn_neic_model.to(devc); pn_neic_model.eval()
pn_scedc_model.to(devc); pn_scedc_model.eval()
pn_stead_model.to(devc); pn_stead_model.eval()
pn_geofon_model.to(devc); pn_geofon_model.eval()
pn_instance_model.to(devc); pn_instance_model.eval()
list_models = [pn_ethz_model,
            pn_neic_model,
            pn_scedc_model,
            pn_stead_model,
            pn_geofon_model,
            pn_instance_model]
fs=100; ch_itv=4 #ch_itv=500  # data are downsampled to pick


## Loop to calculate the phase picking for the number of hours that I said

#just 1 day hrs to try
for ihour in tqdm(range(0,2)):
    ############################################# Read hour data
    print('---- Read {ihour}th hour ----')
    t0 = UTCDateTime("2023-12-10_12") + ihour*3600
    fname = UTCDateTime.strftime(t0, format=format_part)
    rainier_files = sorted(glob.glob(rainier_dir+fname))
    #print(rainier_dir+fname)

    rainier_data = np.zeros((len(rainier_files), 4430, 12000), dtype=np.float32)
    
    rainier_btimes = np.zeros(len(rainier_files), dtype=object)
    

    for i, rainier_file in enumerate(rainier_files):
        with h5py.File(rainier_file, 'r') as f:
            #time_data = f['Acquisition']['Raw[0]']['RawData'][:12000, 500:3000]
            time_data = f['Acquisition']['Raw[0]']['RawData'][:12000, 0:4430]
            rainier_data[i, :time_data.shape[1], :time_data.shape[0]] = time_data.T
            rainier_btimes[i] = datetime.utcfromtimestamp(f['Acquisition']['Raw[0]']['RawDataTime'][0]/1e6)
        del time_data
        gc.collect()


    ### merge two arrays and filter
    
    rawdata = np.nan_to_num(rainier_data)
    # Filter before decimation; the 49 Hz upper corner is below the new Nyquist.
    # Scale each minute by a single standard deviation over channels and time.
    filt = filtfilt(b, a, rawdata, axis=2)
    rawdata = filt / np.std(filt, axis=(1,2), keepdims=True)
    rawdata = rawdata[:,:,::2]
    len_cat = len(rawdata)
    print(rawdata.shape)

    

    ############################################# Denoise
    #:c

    ############################################# Pick
    print('---- Picking ----')
    ### ELEP parameters
    paras_semblance = {'dt':0.01, 
                    'semblance_order':2, 
                    'window_flag':True, 
                    'semblance_win':0.5, 
                    'weight_flag':'max'}
    
    sel_ch = np.arange(int(ch_itv/2), rawdata.shape[1], ch_itv)
    # Sampling starts at channel 2 with stride 4. Each selected trace is passed
    # to the external ELEP helper; spatial event screening happens downstream.
    nsta = len(sel_ch)
    mul_picks = np.zeros([len_cat, nsta, 2, 2], dtype = np.float32)
    for imin in np.arange(len_cat):
        mul_picks[imin,:,:,:] = apply_elep(rawdata[imin,sel_ch,:], 
                                           list_models, fs, paras_semblance, devc)
    
    
    ############################################# Save
    print('---- Saving ----')
    df_deno = pd.DataFrame(columns=[
            'event_id',
            'source_type',
            'station_network_code',
            'station_channel_code',
            'station_code',
            'station_location_code',
            'station_latitude_deg',
            'station_longitude_deg',
            'station_elevation_m',
            'trace_name',
            'trace_sampling_rate_hz',
            'trace_start_time',
            'trace_S_arrival_sample',
            'trace_P_arrival_sample',
            'trace_S_onset',
            'trace_P_onset',
            'trace_snr_db',
            'trace_s_arrival',
            'trace_p_arrival'])

    for ch in np.arange(nsta):
    
        ch1 = int(sel_ch[ch]) 
        #longitude = terra.loc[ch1, 'lon']
        #latitude = terra.loc[ch1, 'lat']
        #elevation = terra.loc[ch1, 'dep']
        b_t = rainier_btimes
       

        p_den = [b_t[i] + timedelta(seconds=np.float64(mul_picks[i, ch, 0, 0])) if mul_picks[i, ch, 0, 1] > 0.10 else np.nan for i in range(len_cat)]
        s_den = [b_t[i] + timedelta(seconds=np.float64(mul_picks[i, ch, 1, 0])) if mul_picks[i, ch, 1, 1] > 0.05 else np.nan for i in range(len_cat)]
        
        df_deno = pd.concat([df_deno, pd.DataFrame(data={
            'event_id': [' '] * len_cat,
            'source_type': [' '] * len_cat,
            'station_network_code': ['MORADAS'] * len_cat,
            'station_channel_code': [' '] * len_cat,
            'station_code': ['das'+str(sel_ch[ch])] * len_cat,
            'station_location_code': [' '] * len_cat,
            'station_latitude_deg': [''] * len_cat,
            'station_longitude_deg': [''] * len_cat,
            'station_elevation_m': [''] * len_cat,
            'trace_name': [' '] * len_cat,
            'trace_sampling_rate_hz': [200] * len_cat,
            'trace_start_time': b_t,
            'trace_S_arrival_sample': [' '] * len_cat,
            'trace_P_arrival_sample': [' '] * len_cat,
            'trace_S_onset': [' '] * len_cat,
            'trace_P_onset': [' '] * len_cat,
            'trace_snr_db': [' '] * len_cat,
            'trace_s_arrival': s_den,
            'trace_p_arrival': p_den})], ignore_index=True)
        
    df_deno.to_csv(filepath + 'MORADAS_' + t0.strftime('%Y%m%d_%H') + '.csv') #create the csv for each hour
