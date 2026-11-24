import sys

import gc
import glob
import h5py
import numpy as np
import pandas as pd
import scipy.ndimage as nd

import torch
import seisbench.models as sbm
from ELEP.elep.ensemble_coherence import ensemble_semblance
from ELEP.elep.trigger_func import picks_summary_simple

from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm

import obspy
from obspy import UTCDateTime, read_events, read, read_inventory
from obspy.clients.fdsn.client import Client

client = Client("IRIS")

### Functions to pick many windows
def apply_elep_smb(data, list_models, paras_semblance, device):
    """
    Input: data of 1 station, 3 components, many time windows
           several eqT models (already to device and eval mode)
    Output: semblance for P or S
    """

    ### Normalize data
    tmp = data - np.mean(data, axis=-1, keepdims= True)
    mmax = np.max(np.abs(tmp), axis=-1, keepdims=True)
    data_max = np.divide(tmp , mmax, out=np.zeros_like(tmp), where=(mmax!=0))
    data_tt = torch.from_numpy(data_max).to(device, dtype=torch.float32)
    
    ### ELEP workflow
    twin = 6000  ## constant for EQTransformer
    nwin = data.shape[0]
   
    # predictions from all models
    batch_pred_P = np.zeros((len(list_models),nwin,twin)) 
    batch_pred_S = np.zeros((len(list_models),nwin,twin))
    smb = np.zeros((nwin,2,twin), dtype = np.float32)
    
    for ii, imodel in enumerate(list_models):
        with torch.no_grad():
            batch_pred_P[ii, :, :] = imodel(data_tt)[1].cpu().numpy()[:, :]
            batch_pred_S[ii, :, :] = imodel(data_tt)[2].cpu().numpy()[:, :]
    
    # semblance of all predictions
    smb[:,0,:] =np.array(Parallel(n_jobs=100)(delayed(ensemble_semblance)(batch_pred_P[:, iwin, :], paras_semblance) 
                                                    for iwin in range(nwin)))
    smb[:,1,:] =np.array(Parallel(n_jobs=100)(delayed(ensemble_semblance)(batch_pred_S[:, iwin, :], paras_semblance) 
                                                    for iwin in range(nwin)))
    
    del batch_pred_P, batch_pred_S, data_tt, data_max, mmax, tmp
    gc.collect()
    torch.cuda.empty_cache()
    
    return smb

def semblance_vec(sig, dt, order=2, win=0.5):
    """
    ELEP ensemble_semblance (window_flag=True, weight_flag='max') for ALL windows at once.

    sig: [nmodels, nwin, npts] model predictions -> returns [nwin, npts] float32.

    Same maths as ELEP.elep.ensemble_coherence.ensemble_semblance -- Marfurt et al. (1998)
    semblance of the nmodels traces in a boxcar of `win` seconds, raised to `order`, then
    weighted by the per-sample max over models -- but the boxcar sum is a uniform_filter1d
    over the whole [nwin, npts] block instead of a python loop over windows. Copied from
    denopick_dasbb_2026/scripts/continuous/elep_continuous_vec.py (Denolle-Lab
    cascadia_obs_ensemble dryrun_models.py::semblance_vec), where it was checked against
    ensemble_semblance to 2e-7. Replaces Parallel(n_jobs=100) over ~1400 windows per day.
    """
    ntr, n = sig.shape[0], int(win / dt)
    sq = np.sum(sig, axis=0) ** 2
    ss = np.sum(sig ** 2, axis=0)
    num = nd.uniform_filter1d(sq.astype(np.float64), n, axis=-1, mode="constant") * n
    den = ntr * nd.uniform_filter1d(ss.astype(np.float64), n, axis=-1, mode="constant") * n
    with np.errstate(invalid="ignore", divide="ignore"):
        s0 = (num / den).astype(np.float32)
    return s0 ** order * np.amax(sig, axis=0)


def apply_elep_smb_vec(data, list_models, paras_semblance, device, batch=256):
    """
    apply_elep_smb with (a) the GPU forward chunked `batch` windows at a time instead of the
    whole day in one call, and (b) semblance_vec instead of the joblib loop. Same input
    ([nwin, 3, npts] windows of 1 station), same output ([nwin, 2, npts] P/S semblance),
    same normalisation (demean + peak-normalise per window and component, in float32).
    Eval-mode EQTransformer outputs are batch-independent up to float rounding.
    """
    tmp = data - np.mean(data, axis=-1, keepdims=True)
    mmax = np.max(np.abs(tmp), axis=-1, keepdims=True)
    data_max = np.divide(tmp, mmax, out=np.zeros_like(tmp), where=(mmax != 0)).astype(np.float32)

    twin = data.shape[-1]
    nwin = data.shape[0]
    batch_pred_P = np.zeros((len(list_models), nwin, twin), dtype=np.float32)
    batch_pred_S = np.zeros((len(list_models), nwin, twin), dtype=np.float32)

    with torch.no_grad():
        for b in range(0, nwin, batch):
            x = torch.from_numpy(data_max[b:b + batch]).to(device, dtype=torch.float32)
            for ii, imodel in enumerate(list_models):
                out = imodel(x)
                batch_pred_P[ii, b:b + batch] = out[1].cpu().numpy()
                batch_pred_S[ii, b:b + batch] = out[2].cpu().numpy()
            del x

    dt, order, win = paras_semblance['dt'], paras_semblance['semblance_order'], paras_semblance['semblance_win']
    smb = np.zeros((nwin, 2, twin), dtype=np.float32)
    smb[:, 0, :] = semblance_vec(batch_pred_P, dt, order, win)
    smb[:, 1, :] = semblance_vec(batch_pred_S, dt, order, win)

    del batch_pred_P, batch_pred_S, data_max, mmax, tmp
    gc.collect()
    torch.cuda.empty_cache()

    return smb

def detect_on_fly(network, station, t1, t2, filepath, width, stride, list_models, devcc):
    '''
    The workflow: 
    1) download data for 1 station
    2) window the data with stride
    3) apply ELEP
    4) save picks
    '''
   ########################
   ### Download waveforms
    try:
        sdata = client.get_waveforms(network=network, 
                                     station=station,
                                     location="*", 
                                     channel="BH?", 
                                     starttime=t1,
                                     endtime=t2)

        inventory = client.get_stations(network=network,
                                        station=station,
                                        location="*", 
                                        channel="BH?", 
                                        starttime=t1,
                                        endtime=t2)
        
    except obspy.clients.fdsn.header.FDSNNoDataException:
        print(f"--- No data for {network}.{station} from {t1} to {t2} ---")
        return
    
    fs_all = [tr.stats.sampling_rate for tr in sdata]
    fs = np.round(fs_all[0])
    if len(np.unique(np.array(fs_all))) > 1:      
        print(f"--- Sampling rates are different for {network}.{station} on {t1} ---")
        sdata = sdata.resample(fs)
    
    sdata.merge(fill_value='interpolate')  # fill gaps
    sdata.filter(type='bandpass',freqmin=0.5,freqmax=12)
    btime = sdata[0].stats.starttime
    

    # align 3 components
    max_b = max([tr.stats.starttime for tr in sdata])
    min_e = min([tr.stats.endtime for tr in sdata])
    for tr in sdata:
        tr.trim(starttime=max_b, endtime=min_e, nearest_sample=True)    
        
    ########################
    ### Window data
    arr_sdata = np.array(sdata)
    if len(arr_sdata.shape) == 1:
        arr_sdata = arr_sdata[np.newaxis, :] # add dimension
    if arr_sdata.shape[0] == 1:
        arr_sdata = np.repeat(arr_sdata, 3, axis=0)
    elif arr_sdata.shape[0] == 2:
        arr_sdata = np.vstack((arr_sdata, arr_sdata[1]))
    elif arr_sdata.shape[0] > 3:
        arr_sdata = arr_sdata[:3]

    nwin = (arr_sdata.shape[1] - width) // stride
    print(arr_sdata.shape[1], width)
    if nwin < 1: 
        print(f"--- Data too short for {network}.{station} on {t1} ---")
        return
    arr_sdata = arr_sdata[:, :int(nwin * stride + width)]
    
    win_idx = np.zeros(nwin, dtype=np.int32)
    windows = np.zeros((nwin, 3, width), dtype= np.float32)

    for iwin in range(nwin):
        idx = iwin * stride
        win_idx[iwin] = idx
        windows[iwin,:,:] = arr_sdata[:, idx:idx+width]
        
    ########################
    ### Apply ELEP
    paras_semblance = {'dt':1/fs, 
                       'semblance_order':2, 
                       'window_flag':True, 
                       'semblance_win':0.5, 
                       'weight_flag':'max'}
    
    smb = apply_elep_smb(windows, list_models, paras_semblance, devcc)
    smb_all = np.zeros_like(arr_sdata[0:2])

    for iwin in range(nwin):
        idx = iwin * stride
        smb_all[0, idx+stride:idx+width] = smb[iwin,0,stride:]
        smb_all[1, idx+stride:idx+width] = smb[iwin,1,stride:]

    p_picks = picks_summary_simple(smb_all[0], 0.10)
    s_picks = picks_summary_simple(smb_all[1], 0.05)

    ########################
    ### Save picks
    len_picks = len(p_picks + s_picks)
    df = pd.DataFrame({
        'event_id': [' '] * len_picks,
        'source_type': [' '] * len_picks,
        'station_network_code': [network] * len_picks,
        'station_channel_code': [' '] * len_picks,
        'station_code': [station] * len_picks,
        'station_location_code': [sdata[0].stats.location] * len_picks,
        'station_latitude_deg': [inventory[0][0].latitude] * len_picks,
        'station_longitude_deg': [inventory[0][0].longitude] * len_picks,
        'station_elevation_m': [inventory[0][0].elevation] * len_picks,
        'trace_name': [' '] * len_picks,
        'trace_sampling_rate_hz': [sdata[0].stats.sampling_rate] * len_picks,
        'trace_start_time': [sdata[0].stats.starttime] * len_picks,
        'trace_S_arrival_sample': [' '] * len_picks,
        'trace_P_arrival_sample': [' '] * len_picks,
        'trace_S_onset': [' '] * len_picks,
        'trace_P_onset': [' '] * len_picks,
        'trace_snr_db': [' '] * len_picks,
        'trace_s_arrival': [np.nan] * len(p_picks) + [str(btime + idx / fs) for idx in s_picks],
        'trace_p_arrival': [str(btime + idx / fs) for idx in p_picks] + [np.nan] * len(s_picks)
    })

    df.to_csv(filepath+'/' + network + '_' + station + '_' + t1.strftime('%Y%m%d') + '.csv')

def detect_on_fly2(network, station, t1, t2, filepath, width, stride, list_models, devcc):
    '''
    The workflow: 
    1) download data for 1 station
    2) window the data with stride
    3) apply ELEP
    4) save picks
    '''
   ########################
   ### Download waveforms
    try:
        sdata = client.get_waveforms(network=network, 
                                     station=station,
                                     location="*", 
                                     channel="HH?", 
                                     starttime=t1,
                                     endtime=t2)

        inventory = client.get_stations(network=network,
                                        station=station,
                                        location="*", 
                                        channel="HH?", 
                                        starttime=t1,
                                        endtime=t2)
        
    except obspy.clients.fdsn.header.FDSNNoDataException:
        print(f"--- No data for {network}.{station} from {t1} to {t2} ---")
        return
    
    fs_all = [tr.stats.sampling_rate for tr in sdata]
    fs = np.round(fs_all[0])
    if len(np.unique(np.array(fs_all))) > 1:      
        print(f"--- Sampling rates are different for {network}.{station} on {t1} ---")
        sdata = sdata.resample(fs)
    
    sdata.merge(fill_value='interpolate')  # fill gaps
    sdata.filter(type='bandpass',freqmin=2,freqmax=9)
    btime = sdata[0].stats.starttime
    

    # align 3 components
    max_b = max([tr.stats.starttime for tr in sdata])
    min_e = min([tr.stats.endtime for tr in sdata])
    for tr in sdata:
        tr.trim(starttime=max_b, endtime=min_e, nearest_sample=True)    
        
    ########################
    ### Window data
    arr_sdata = np.array(sdata)
    if len(arr_sdata.shape) == 1:
        arr_sdata = arr_sdata[np.newaxis, :] # add dimension
    if arr_sdata.shape[0] == 1:
        arr_sdata = np.repeat(arr_sdata, 3, axis=0)
    elif arr_sdata.shape[0] == 2:
        arr_sdata = np.vstack((arr_sdata, arr_sdata[1]))
    elif arr_sdata.shape[0] > 3:
        arr_sdata = arr_sdata[:3]

    nwin = (arr_sdata.shape[1] - width) // stride
    print(arr_sdata.shape[1], width)
    if nwin < 1: 
        print(f"--- Data too short for {network}.{station} on {t1} ---")
        return
    arr_sdata = arr_sdata[:, :int(nwin * stride + width)]
    
    win_idx = np.zeros(nwin, dtype=np.int32)
    windows = np.zeros((nwin, 3, width), dtype= np.float32)

    for iwin in range(nwin):
        idx = iwin * stride
        win_idx[iwin] = idx
        windows[iwin,:,:] = arr_sdata[:, idx:idx+width]
        
    ########################
    ### Apply ELEP
    paras_semblance = {'dt':1/fs, 
                       'semblance_order':2, 
                       'window_flag':True, 
                       'semblance_win':0.5, 
                       'weight_flag':'max'}
    
    smb = apply_elep_smb(windows, list_models, paras_semblance, devcc)
    smb_all = np.zeros_like(arr_sdata[0:2])

    for iwin in range(nwin):
        idx = iwin * stride
        smb_all[0, idx+stride:idx+width] = smb[iwin,0,stride:]
        smb_all[1, idx+stride:idx+width] = smb[iwin,1,stride:]

    p_picks = picks_summary_simple(smb_all[0], 0.10)
    s_picks = picks_summary_simple(smb_all[1], 0.05)

    ########################
    ### Save picks
    len_picks = len(p_picks + s_picks)
    df = pd.DataFrame({
        'event_id': [' '] * len_picks,
        'source_type': [' '] * len_picks,
        'station_network_code': [network] * len_picks,
        'station_channel_code': [' '] * len_picks,
        'station_code': [station] * len_picks,
        'station_location_code': [sdata[0].stats.location] * len_picks,
        'station_latitude_deg': [inventory[0][0].latitude] * len_picks,
        'station_longitude_deg': [inventory[0][0].longitude] * len_picks,
        'station_elevation_m': [inventory[0][0].elevation] * len_picks,
        'trace_name': [' '] * len_picks,
        'trace_sampling_rate_hz': [sdata[0].stats.sampling_rate] * len_picks,
        'trace_start_time': [sdata[0].stats.starttime] * len_picks,
        'trace_S_arrival_sample': [' '] * len_picks,
        'trace_P_arrival_sample': [' '] * len_picks,
        'trace_S_onset': [' '] * len_picks,
        'trace_P_onset': [' '] * len_picks,
        'trace_snr_db': [' '] * len_picks,
        'trace_s_arrival': [np.nan] * len(p_picks) + [str(btime + idx / fs) for idx in s_picks],
        'trace_p_arrival': [str(btime + idx / fs) for idx in p_picks] + [np.nan] * len(s_picks)
    })

    df.to_csv(filepath+'/' + network + '_' + station + '_' + t1.strftime('%Y%m%d') + '.csv')

def detect_on_fly3(network, station, t1, t2, filepath, width, stride, list_models, devcc, channel="HH?",
                   batch=256):
    '''
    detect_on_fly2 + the component-order fix. Same workflow:
    1) download data for 1 station
    2) window the data with stride
    3) apply ELEP
    4) save picks

    CHANGE vs detect_on_fly / detect_on_fly2 (2026-10-28):
      The 3 traces are sorted into Z,N,E before np.array(sdata). The pretrained
      EQTransformer weights expect Z,N,E (the ELEP tutorials do
      np.array(s2)[[2,1,0], :] for an E,N,Z stream). detect_on_fly/detect_on_fly2
      used the stream as IRIS returned it (E,N,Z, and arbitrary after merging gappy
      data), so the vertical sat in the E slot. Same fix as
      Shi_etal_2023_denoiseDAS/src/ensemble_picker/detect_util.py (2026-10-20).
      Also takes `channel` ("BH?" or "HH?") instead of hardcoding it.

    CHANGE (2026-10-29): semblance + forward via apply_elep_smb_vec -- the GPU forward
      is chunked `batch` windows at a time and the semblance is the vectorized
      semblance_vec (same maths as ELEP ensemble_semblance, no joblib loop).
      Picks are unchanged up to float rounding.
    '''
   ########################
   ### Download waveforms
    try:
        sdata = client.get_waveforms(network=network, 
                                     station=station,
                                     location="*", 
                                     channel=channel, 
                                     starttime=t1,
                                     endtime=t2)

        inventory = client.get_stations(network=network,
                                        station=station,
                                        location="*", 
                                        channel=channel, 
                                        starttime=t1,
                                        endtime=t2)
        
    except obspy.clients.fdsn.header.FDSNNoDataException:
        print(f"--- No data for {network}.{station} from {t1} to {t2} ---")
        return
    
    fs_all = [tr.stats.sampling_rate for tr in sdata]
    fs = np.round(fs_all[0])
    if len(np.unique(np.array(fs_all))) > 1:      
        print(f"--- Sampling rates are different for {network}.{station} on {t1} ---")
        sdata = sdata.resample(fs)
    
    sdata.merge(fill_value='interpolate')  # fill gaps
    sdata.filter(type='bandpass',freqmin=2,freqmax=9)
    btime = sdata[0].stats.starttime
    

    # align 3 components
    max_b = max([tr.stats.starttime for tr in sdata])
    min_e = min([tr.stats.endtime for tr in sdata])
    for tr in sdata:
        tr.trim(starttime=max_b, endtime=min_e, nearest_sample=True)    
        
    ########################
    ### Window data
    # Components in Z,N,E (see docstring). Z/3 -> 0, N/1 -> 1, E/2 -> 2; anything else last.
    _rank = {'Z': 0, '3': 0, 'N': 1, '1': 1, 'E': 2, '2': 2}
    sdata = obspy.Stream(sorted(sdata, key=lambda tr: (_rank.get(tr.stats.channel[-1].upper(), 9), tr.stats.channel)))
    arr_sdata = np.array(sdata)
    if len(arr_sdata.shape) == 1:
        arr_sdata = arr_sdata[np.newaxis, :] # add dimension
    if arr_sdata.shape[0] == 1:
        arr_sdata = np.repeat(arr_sdata, 3, axis=0)
    elif arr_sdata.shape[0] == 2:
        arr_sdata = np.vstack((arr_sdata, arr_sdata[1]))
    elif arr_sdata.shape[0] > 3:
        arr_sdata = arr_sdata[:3]

    nwin = (arr_sdata.shape[1] - width) // stride
    print(arr_sdata.shape[1], width)
    if nwin < 1: 
        print(f"--- Data too short for {network}.{station} on {t1} ---")
        return
    arr_sdata = arr_sdata[:, :int(nwin * stride + width)]
    
    win_idx = np.zeros(nwin, dtype=np.int32)
    windows = np.zeros((nwin, 3, width), dtype= np.float32)

    for iwin in range(nwin):
        idx = iwin * stride
        win_idx[iwin] = idx
        windows[iwin,:,:] = arr_sdata[:, idx:idx+width]
        
    ########################
    ### Apply ELEP
    paras_semblance = {'dt':1/fs, 
                       'semblance_order':2, 
                       'window_flag':True, 
                       'semblance_win':0.5, 
                       'weight_flag':'max'}
    
    smb = apply_elep_smb_vec(windows, list_models, paras_semblance, devcc, batch=batch)
    smb_all = np.zeros_like(arr_sdata[0:2])

    for iwin in range(nwin):
        idx = iwin * stride
        smb_all[0, idx+stride:idx+width] = smb[iwin,0,stride:]
        smb_all[1, idx+stride:idx+width] = smb[iwin,1,stride:]

    p_picks = picks_summary_simple(smb_all[0], 0.10)
    s_picks = picks_summary_simple(smb_all[1], 0.05)

    ########################
    ### Save picks
    len_picks = len(p_picks + s_picks)
    df = pd.DataFrame({
        'event_id': [' '] * len_picks,
        'source_type': [' '] * len_picks,
        'station_network_code': [network] * len_picks,
        'station_channel_code': [' '] * len_picks,
        'station_code': [station] * len_picks,
        'station_location_code': [sdata[0].stats.location] * len_picks,
        'station_latitude_deg': [inventory[0][0].latitude] * len_picks,
        'station_longitude_deg': [inventory[0][0].longitude] * len_picks,
        'station_elevation_m': [inventory[0][0].elevation] * len_picks,
        'trace_name': [' '] * len_picks,
        'trace_sampling_rate_hz': [sdata[0].stats.sampling_rate] * len_picks,
        'trace_start_time': [sdata[0].stats.starttime] * len_picks,
        'trace_S_arrival_sample': [' '] * len_picks,
        'trace_P_arrival_sample': [' '] * len_picks,
        'trace_S_onset': [' '] * len_picks,
        'trace_P_onset': [' '] * len_picks,
        'trace_snr_db': [' '] * len_picks,
        'trace_s_arrival': [np.nan] * len(p_picks) + [str(btime + idx / fs) for idx in s_picks],
        'trace_p_arrival': [str(btime + idx / fs) for idx in p_picks] + [np.nan] * len(s_picks)
    })

    df.to_csv(filepath+'/' + network + '_' + station + '_' + t1.strftime('%Y%m%d') + '.csv')

#added later for mora nodes

# take Qibin's detect_on_fly function and rewrite it to work with the mseed files instead of downloading data from IRIS
def detect_mora_nodes(file, output_dir, width, stride, list_models, devcc):
    event_xml_path = '/1-fnp/psound/psound-wd5/mora_nodes/stationXML/Z5'
    '''
    The workflow: 
    1) download data for 1 station
    2) window the data with stride
    3) apply ELEP
    4) save picks
    '''
   ########################
   ### Download waveforms
    
    #read downloaded data from mseed file
    sdata = read(file)

    base = file.split("/")[-1]
    node_id = base.split(".")[0]
    event = base.split(".")[-1]

    #take the inventory from the .xml file
    inventory = read_inventory(f'{event_xml_path}/Z5.{node_id}.xml')

    fs_all = [tr.stats.sampling_rate for tr in sdata]
    fs = np.round(fs_all[0])
    if len(np.unique(np.array(fs_all))) > 1:      
        print(f"--- Sampling rates are different for {node_id} during {event}---")
        sdata = sdata.resample(fs)
    
    sdata.merge(fill_value='interpolate')  # fill gaps
    sdata.filter(type='bandpass',freqmin=0.5,freqmax=12)
    btime = sdata[0].stats.starttime
    

    # align 3 components
    max_b = max([tr.stats.starttime for tr in sdata])
    min_e = min([tr.stats.endtime for tr in sdata])
    for tr in sdata:
        tr.trim(starttime=max_b, endtime=min_e, nearest_sample=True)    
        
    ########################
    ### Window data
    arr_sdata = np.array(sdata)
    if len(arr_sdata.shape) == 1:
        arr_sdata = arr_sdata[np.newaxis, :] # add dimension
    if arr_sdata.shape[0] == 1:
        arr_sdata = np.repeat(arr_sdata, 3, axis=0)
    elif arr_sdata.shape[0] == 2:
        arr_sdata = np.vstack((arr_sdata, arr_sdata[1]))
    elif arr_sdata.shape[0] > 3:
        arr_sdata = arr_sdata[:3]

    nwin = (arr_sdata.shape[1] - width) // stride
    print(arr_sdata.shape[1], width)
    if nwin < 1: 
        print(f"--- Data too short for {node_id} during {event} ---")
        return
    arr_sdata = arr_sdata[:, :int(nwin * stride + width)]
    
    win_idx = np.zeros(nwin, dtype=np.int32)
    windows = np.zeros((nwin, 3, width), dtype= np.float32)

    for iwin in range(nwin):
        idx = iwin * stride
        win_idx[iwin] = idx
        windows[iwin,:,:] = arr_sdata[:, idx:idx+width]
        
    ########################
    ### Apply ELEP
    paras_semblance = {'dt':1/fs, 
                       'semblance_order':2, 
                       'window_flag':True, 
                       'semblance_win':0.5, 
                       'weight_flag':'max'}
    
    smb = apply_elep_smb(windows, list_models, paras_semblance, devcc)
    smb_all = np.zeros_like(arr_sdata[0:2])

    for iwin in range(nwin):
        idx = iwin * stride
        smb_all[0, idx+stride:idx+width] = smb[iwin,0,stride:]
        smb_all[1, idx+stride:idx+width] = smb[iwin,1,stride:]

    p_picks = picks_summary_simple(smb_all[0], 0.10)
    s_picks = picks_summary_simple(smb_all[1], 0.05)

    ########################
    ### Save picks
    len_picks = len(p_picks + s_picks)
    df = pd.DataFrame({
        'event_id': [' '] * len_picks,
        'source_type': [' '] * len_picks,
        'station_network_code': ['Z5'] * len_picks,
        'station_channel_code': [' '] * len_picks,
        'station_code': [node_id] * len_picks,
        'station_location_code': [sdata[0].stats.location] * len_picks,
        'station_latitude_deg': [inventory[0][0].latitude] * len_picks,
        'station_longitude_deg': [inventory[0][0].longitude] * len_picks,
        'station_elevation_m': [inventory[0][0].elevation] * len_picks,
        'trace_name': [' '] * len_picks,
        'trace_sampling_rate_hz': [sdata[0].stats.sampling_rate] * len_picks,
        'trace_start_time': [sdata[0].stats.starttime] * len_picks,
        'trace_S_arrival_sample': [' '] * len_picks,
        'trace_P_arrival_sample': [' '] * len_picks,
        'trace_S_onset': [' '] * len_picks,
        'trace_P_onset': [' '] * len_picks,
        'trace_snr_db': [' '] * len_picks,
        'trace_s_arrival': [np.nan] * len(p_picks) + [str(btime + idx / fs) for idx in s_picks],
        'trace_p_arrival': [str(btime + idx / fs) for idx in p_picks] + [np.nan] * len(s_picks)
    })

    df.to_csv(output_dir + '/' + node_id + '.Z5.2025.' + event + '.csv')

def detect_badger_nodes(file, output_dir, width, stride, list_models, devcc):
    '''
    The workflow: 
    1) download data for 1 station
    2) window the data with stride
    3) apply ELEP
    4) save picks
    '''
   ########################
   ### Download waveforms
    
    #read downloaded data from mseed file
    sdata = read(file)

    base = file.split("/")[-1]
    node_id = base.split(".")[0]
    event = base.split("/")[-1]

    

    #take the inventory from the .xml file
    #inventory = read_inventory(f'{event_xml_path}/Z5.{node_id}.xml')

    fs_all = [tr.stats.sampling_rate for tr in sdata]
    fs = np.round(fs_all[0])
    if len(np.unique(np.array(fs_all))) > 1:      
        print(f"--- Sampling rates are different for {node_id} during {event}---")
        sdata = sdata.resample(fs)
    
    sdata.merge(fill_value='interpolate')  # fill gaps
    sdata.filter(type='bandpass',freqmin=0.5,freqmax=12)
    btime = sdata[0].stats.starttime
    

    # align 3 components
    max_b = max([tr.stats.starttime for tr in sdata])
    min_e = min([tr.stats.endtime for tr in sdata])
    for tr in sdata:
        tr.trim(starttime=max_b, endtime=min_e, nearest_sample=True)    
        
    ########################
    ### Window data
    arr_sdata = np.array(sdata)
    if len(arr_sdata.shape) == 1:
        arr_sdata = arr_sdata[np.newaxis, :] # add dimension
    if arr_sdata.shape[0] == 1:
        arr_sdata = np.repeat(arr_sdata, 3, axis=0)
    elif arr_sdata.shape[0] == 2:
        arr_sdata = np.vstack((arr_sdata, arr_sdata[1]))
    elif arr_sdata.shape[0] > 3:
        arr_sdata = arr_sdata[:3]

    nwin = (arr_sdata.shape[1] - width) // stride
    print(arr_sdata.shape[1], width)
    if nwin < 1: 
        print(f"--- Data too short for {node_id} during {event} ---")
        return
    arr_sdata = arr_sdata[:, :int(nwin * stride + width)]
    
    win_idx = np.zeros(nwin, dtype=np.int32)
    windows = np.zeros((nwin, 3, width), dtype= np.float32)

    for iwin in range(nwin):
        idx = iwin * stride
        win_idx[iwin] = idx
        windows[iwin,:,:] = arr_sdata[:, idx:idx+width]
        
    ########################
    ### Apply ELEP
    paras_semblance = {'dt':1/fs, 
                       'semblance_order':2, 
                       'window_flag':True, 
                       'semblance_win':0.5, 
                       'weight_flag':'max'}
    
    smb = apply_elep_smb(windows, list_models, paras_semblance, devcc)
    smb_all = np.zeros_like(arr_sdata[0:2])

    for iwin in range(nwin):
        idx = iwin * stride
        smb_all[0, idx+stride:idx+width] = smb[iwin,0,stride:]
        smb_all[1, idx+stride:idx+width] = smb[iwin,1,stride:]

    p_picks = picks_summary_simple(smb_all[0], 0.10)
    s_picks = picks_summary_simple(smb_all[1], 0.05)

    ########################
    ### Save picks
    len_picks = len(p_picks + s_picks)
    df = pd.DataFrame({
        'event_id': [' '] * len_picks,
        'source_type': [' '] * len_picks,
        'station_network_code': ['Z5'] * len_picks,
        'station_channel_code': [' '] * len_picks,
        'station_code': [node_id] * len_picks,
        'station_location_code': [sdata[0].stats.location] * len_picks,
        'station_latitude_deg': [np.nan] * len_picks,
        'station_longitude_deg': [np.nan] * len_picks,
        'station_elevation_m': [np.nan] * len_picks,
        'trace_name': [' '] * len_picks,
        'trace_sampling_rate_hz': [sdata[0].stats.sampling_rate] * len_picks,
        'trace_start_time': [sdata[0].stats.starttime] * len_picks,
        'trace_S_arrival_sample': [' '] * len_picks,
        'trace_P_arrival_sample': [' '] * len_picks,
        'trace_S_onset': [' '] * len_picks,
        'trace_P_onset': [' '] * len_picks,
        'trace_snr_db': [' '] * len_picks,
        'trace_s_arrival': [np.nan] * len(p_picks) + [str(btime + idx / fs) for idx in s_picks],
        'trace_p_arrival': [str(btime + idx / fs) for idx in p_picks] + [np.nan] * len(s_picks)
    })

    #print(output_dir + '/' + event + '.csv')
    df.to_csv(output_dir + '/' + event + '.csv')