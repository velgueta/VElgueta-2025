import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
import datetime
import pandas as pd
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import time
import glob
import os
import pytz
from tqdm import tqdm
import csv
import matplotlib.dates as mdates
import csv
import re
from templatematching import *

#loading data to run

def get_file_list(base_path,start_date,end_date):
    file_list = []
    actual_date = start_date
    
    while actual_date <= end_date:
        #new_decimator is because is downsample data, usually is just decimator
        #search_pattern = f"{base_path}/new_decimator_{actual_date}*"
        search_pattern = f"{base_path}/decimator_{actual_date}*"
        file_list.extend(glob.glob(search_pattern))
        actual_date = increase_date(actual_date)
        file_list.sort()
    return file_list
    
# defining the other fuction to increment the date
def increase_date(date_str):
    from datetime import datetime,timedelta
    date_format = "%Y-%m-%d_%H.%M"
    date = datetime.strptime(date_str,date_format)
    next_date = date + timedelta(minutes=1)
    return next_date.strftime(date_format)

#def increase_date(date_str):
#    date_format = "%Y-%m-%d_%H-%M"
#    date = datetime.strptime(date_str, date_format)
#    next_date = date + timedelta(minutes=1)
#    return next_date.strftime(date_format)

# Buiding outputfiles and correlations for each template on the list

# Fuctions
def loading_data(file, tem, chan_min, chan_max):
    '''
    Load data from an HDF5 file and a template file, and return the template, 
    raw data filtered by channel, and timestamps.

    Args:
        file (str): Path to the raw data file.
        tem (str): Path to the template file.
        chan_min (int): Index of the minimum channel to consider.
        chan_max (int): Index of the maximum channel to consider (exclusive).

    Returns:
        tuple: A tuple containing:
            - template (np.ndarray): Template data.
            - raw_data (np.ndarray): Raw data filtered by channel.
            - timestamps (np.ndarray): Timestamps of the raw data.

    Raises:
        RuntimeError: If there is an error while loading the data.
    '''
    try:
        with h5py.File(file, "r") as f, h5py.File(tem, "r") as d:
            template = np.array(d['Acquisition/Raw[0]/RawData'][:, 0:-1])
            raw_data = np.array(f['Acquisition/Raw[0]/RawData'][:, chan_min:chan_max])
            timestamps = np.array(f['Acquisition/Raw[0]/RawDataTime'])
        print(f"Loaded template shape: {template.shape}, raw_data shape: {raw_data.shape}, timestamps shape: {timestamps.shape}")
        return template, raw_data, timestamps
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
        
        
def filter_data(raw_data, b, a):
    """
    Filtering the raw_data using filtfilt
    
    b,a comes from butter 
    """
    try:
        data_filt = filtfilt(b, a, raw_data, axis=0)
        return data_filt
    except Exception as e:
        raise RuntimeError(f"Error filtering data: {e}")

def compute_correlations(template, data_filt, samples_per_file, channel_number):
    """
    This function calculates the correlation between the template and the filtered data.
    The correlation is the size of the template.
    """
    try:
        # Ensure template and data_filt have the same number of channels
        if template.shape[1] != data_filt.shape[1]:
            raise ValueError(f"Channel mismatch: template channels = {template.shape[1]}, data_filt channels = {data_filt.shape[1]}")

        corrs = window_and_correlate(template, data_filt)
        print(f"corrs.size: {corrs.size}, expected size: {samples_per_file * template.shape[1]}")
        print(f"template.shape: {template.shape}, data_filt.shape: {data_filt.shape}")

        # Ensure the size of correlations matches the expected size
        expected_size = samples_per_file * template.shape[1]
        if corrs.size != expected_size:
            raise ValueError(f"Size mismatch: corrs.size = {corrs.size}, expected size = {expected_size}")

        corrs2 = corrs.reshape((samples_per_file, template.shape[1]))
        corrs3 = np.sum(corrs2, axis=1) / channel_number
        return corrs3
    except Exception as e:
        raise RuntimeError(f"Error computing correlations: {e}")

def process_files_dos(file_list, template_list, chan_min, chan_max, channel_number, samples_per_file, b, a, full_path):
    """
    This fuction calculate the correlation between the template with data filt
    the correlation is the size of the template.
    
    The loop is opening 1 file from file_list and correlating them with all templates saved on template list
    Next it is doing the same but the next file in file_list
    
    
    """
    
    #start_time = time.perf_counter()
    
    for i, file in tqdm(enumerate(file_list)):
        for j, tem in tqdm(enumerate(template_list)):
            try:
                # Load data
                raw_template, raw_data, timestamps = loading_data(file, tem, chan_min, chan_max)
                
                # Filter
                data_filt = filter_data(raw_data, b, a)
                template  = filter_data(raw_template,b,a)
                
                # Compute correlations
                corrs3 = compute_correlations(template, data_filt, samples_per_file, channel_number)
                
                # output folder for correlations, based on the name of the template
                folder_name_parts = os.path.splitext(os.path.basename(tem))[0].split('_')[0:2]
                folder_name = '_'.join(folder_name_parts)
                folder_output = os.path.join(full_path, folder_name)
                
                # create it if does not exist
                if not os.path.exists(folder_output):
                    os.mkdir(folder_output)
                
                # Saved correlations values
                outfile_name = os.path.join(folder_output, f'corrs_{i}_.npy')
                np.save(outfile_name, corrs3)
                #print(f"Saved: {outfile_name}")
            except Exception as e:
                print(f"Error processing file {file} with template {tem}: {e}. Skipping this file and moving to the next template.")
                continue

    #end_time = time.perf_counter()
    #execution_time = end_time - start_time
    #print(f"The code took {execution_time} seconds.")
    
    
    
# ANALYSIS OF DATA

# Making .h5 with the date from file_list

    



def create_timestamps_h5(file_list, output_dir):
    """
    Creates an HDF5 file with timestamps extracted from the given list of files.

    Args:
        file_list (list): List of file paths containing the timestamps.
        output_dir (str): Directory where the HDF5 file will be saved.

    Returns:
        str: File path of the created HDF5 file.
    """
    # Extract date from the first and last file names
    first_file_name = os.path.basename(file_list[0])
    file_name_parts = first_file_name.split('_')
    date_time_part = '_'.join(file_name_parts[1:2])  # '2023-08-27_11.00.00_UTC'


    last_file_name = os.path.basename(file_list[-1])
    last_file_name_parts = last_file_name.split('_')
    last_date_time_part = '_'.join(last_file_name_parts[1:3])  # '2023-08-27_11.00.00_UTC'
    
    # Create the output filename with the desired format
    output_file_name = f"timestamps_{date_time_part}_{last_date_time_part}.h5"
    output_file_h5 = os.path.join(output_dir, output_file_name)

    ## Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the HDF5 file and save the timestamps

    with h5py.File(output_file_h5, 'w') as f:
    # Create a dataset to store the timestamps
        timestamps_dataset = f.create_dataset('timestamps', (0,), dtype='i8', maxshape=(None,))

    # Iterate over the files and add the timestamps to the dataset
        for file_path in tqdm(file_list, desc="Processing files"):
            with h5py.File(file_path, 'r') as f_read:
            # Get timestamps from the read file
                timestamps = np.array(f_read['Acquisition/Raw[0]/RawDataTime'])
                num_timestamps = len(timestamps)

            # Extend the dataset to add the new timestamps
                timestamps_dataset.resize((timestamps_dataset.shape[0] + num_timestamps,))
                timestamps_dataset[-num_timestamps:] = timestamps

# Print completion message
    print(f"Timestamps have been saved in the file {output_file_h5}.")
    return output_file_h5

from datetime import datetime
def convert_timestamps_to_utc(input_file_h5, pt_timezone_str='America/Los_Angeles', utc_timezone_str='UTC'):
    with h5py.File(input_file_h5, 'r') as f:
        timestamps_pt = np.array(f['timestamps'])
    pt_timezone = pytz.timezone(pt_timezone_str)
    timestamps_seconds = timestamps_pt / 1e6
    datetime_objects_pt = [datetime.fromtimestamp(ts, pt_timezone) for ts in timestamps_seconds]
    utc_timezone = pytz.timezone(utc_timezone_str)
    time_utc = [dt_pt.astimezone(utc_timezone) for dt_pt in datetime_objects_pt]
    
    return time_utc


from scipy.stats import norm

def mad_func_li(arr):
    """ Median Absolute Deviation: Using the formulation in Li and Zhan 2018
    Pushing the limit of earthquake detection with DAS and template matching
    """
    # arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    mean = np.mean(arr)
    med = np.median(arr)
    return np.median(np.abs(mean - med))
def mad_func_shelly(arr):
    """ Median Absolute Deviation: Using the formulation in Li and Zhan 2018
    Pushing the limit of earthquake detection with DAS and template matching
    """
    # arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    # mean = np.mean(arr)
    med = np.median(arr)
    return np.median(np.abs(arr - med))
def detections(cc_arr, mode='shelly'):
    """ using detection significance from Li and Zhan 2018
    det = (peak - med) / mad
    """
    med = np.median(cc_arr)
    if mode == 'shelly':
        mad = mad_func_shelly(cc_arr)
    elif mode == 'li':
        mad = mad_func_li(cc_arr)
    det = (cc_arr - med) / mad
    return det,mad

def mad_func_shelly(data):
    median = np.median(data)
    deviations = np.abs(data - median)
    mad = np.median(deviations)
    return mad

'''
process folder it is use to take all folders and the corrs and find the event that overpass a threshold.
'''
def process_folders(full_path, time_utc, matched_files_with_locations, template_size, output_directory):
    try:
        # Get a list of all folders in the base directory
        folders = [folder for folder in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, folder))]
    except Exception as e:
        print(f"Error listing folders in {full_path}: {e}")
        return

    # Initialize a list to store the concatenated data from each folder
    concatenated_data_per_folder = []

    # Iterate over the folders and load the .npy files
    for folder in folders:
        folder_path = os.path.join(full_path, folder)
        try:
            npy_files = [np.load(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if file.endswith('.npy')]
            concatenated_data_per_folder.append(np.concatenate(npy_files, axis=0))
        except Exception as e:
            print(f"Error processing folder {folder_path}: {e}")
            continue

    # Calculate MAD for each folder and define the thresholds
    mads_per_folder = {}
    thresholds_per_folder = {}
    for folder, folder_data in zip(folders, concatenated_data_per_folder):
        try:
            mad = mad_func_shelly(folder_data)
            threshold =  18*mad
            mads_per_folder[folder] = np.round(mad, decimals=3)
            thresholds_per_folder[folder] = np.round(threshold, decimals=3)
            #print(f"MAD for folder {folder}: {mads_per_folder[folder]}")
            print(f"Threshold for folder {folder}: {thresholds_per_folder[folder]}")
        except Exception as e:
            print(f"Error calculating MAD for folder {folder}: {e}")
  

    detection_times = []

    for folder, folder_data in zip(folders, concatenated_data_per_folder):
        print(f"Processing folder {folder}")
        threshold = thresholds_per_folder[folder]
        indices_above_threshold = np.where(np.abs(folder_data) > threshold)[0]
        diff_indices = np.diff(indices_above_threshold)
        group_changes = np.where(diff_indices > 20)[0]
        detection_groups = np.split(indices_above_threshold, group_changes + 1)

        detection_times_folder = []

        # Match with the folder
        matching_item = next((item for item in matched_files_with_locations if item['original_date'].replace(":", "-").replace(" ", "_") in folder), None)
        if matching_item:
            for group in detection_groups:
                if len(group) > 0:
                    first_detection_time_utc = time_utc[group[0]].strftime('%Y-%m-%d %H:%M:%S')
                    detection_times_folder.append({
                        'Detection Time (UTC)': first_detection_time_utc,
                        'Longitude': matching_item['longitude'],
                        'Latitude': matching_item['latitude'],
                        'Folder': folder
                    })
            print(f"Matching item found for folder {folder}: {matching_item}")
        else:
            print(f"No matching item found for folder {folder}")

        print(f"Detections for folder {folder}: {detection_times_folder}")
        detection_times.extend(detection_times_folder)

        # Save to CSV
        df = pd.DataFrame(detection_times_folder)
        df.to_csv(os.path.join(output_directory, f'detections_{folder}.csv'), index=False)

    # Remove duplicates from all CSV files
    if detection_times:  # Ensure there are detection times to process
        all_detections = pd.DataFrame(detection_times)
        all_detections['Detection Time (UTC)'] = pd.to_datetime(all_detections['Detection Time (UTC)'])
        all_detections = all_detections.sort_values(by='Detection Time (UTC)')

        # Remove detections that are within 10 seconds of each other
        threshold_seconds = 8
        unique_detections = []
        duplicates_dict = {}
        previous_time = None

        for index, row in all_detections.iterrows():
            detection_time = row['Detection Time (UTC)']
            folder = row['Folder']
            if previous_time is None or (detection_time - previous_time).total_seconds() > threshold_seconds:
                unique_detections.append(detection_time)
                duplicates_dict[detection_time] = [folder]
                previous_time = detection_time
            else:
                duplicates_dict[previous_time].append(folder)

        # Create DataFrame for unique detections and their duplicates
        unique_detections_data = []
        for unique_time, folders in duplicates_dict.items():
            unique_detections_data.append({
                'Unique Detection Time (UTC)': unique_time,
                'Duplicate Detections (UTC)': ', '.join(folders)
            })

        unique_detections_df = pd.DataFrame(unique_detections_data)

        # Print the first column of unique_detections_df
        print("Unique Detection Times (UTC):")
        print(unique_detections_df['Duplicate Detections (UTC)'])

        unique_detections_df.to_csv(os.path.join(output_directory, 'unique_detections.csv'), index=False)

        print(f"Detections saved in {output_directory}")
    else:
        print("No detection times to process.")

'''
This fuction is plotting the corrs values and the threshold for each folder
'''

def process_folders_and_plot(full_path, fs, time_utc, output_plot_directory, found_files):
    try:
        folders = [folder for folder in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, folder))]
    except Exception as e:
        print(f"Error listing folders in {full_path}: {e}")
        return

    concatenated_data_per_folder = []
    for folder in folders:
        folder_path = os.path.join(full_path, folder)
        try:
            npy_files = [np.load(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if file.endswith('.npy')]
            concatenated_data_per_folder.append(np.concatenate(npy_files, axis=0))
        except Exception as e:
            print(f"Error processing folder {folder_path}: {e}")
            continue

    mads_per_folder = {}
    thresholds_per_folder = {}
    for folder, folder_data in zip(folders, concatenated_data_per_folder):
        try:
            mad = mad_func_shelly(folder_data)
            threshold =  18*mad
            mads_per_folder[folder] = np.round(mad, decimals=3)
            thresholds_per_folder[folder] = np.round(threshold, decimals=3)
            #print(f"MAD for folder {folder}: {mads_per_folder[folder]}")
            print(f"Threshold for folder {folder}: {thresholds_per_folder[folder]}")
        except Exception as e:
            print(f"Error calculating MAD for folder {folder}: {e}")


    for i, (folder, folder_data) in enumerate(zip(folders, concatenated_data_per_folder)):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(time_utc[:len(folder_data)], folder_data, label=f'Folder {folder}', color='blue', linestyle='-')
        ax.axhline(y=thresholds_per_folder[folder], color='red', linestyle='--', label='Threshold')
        ax.set_title(f'Template {folder}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Correlation Value')
        ax.legend(loc='upper right')
        ax.grid(True)

        plot_filename = os.path.join(output_plot_directory, f'plot_{folder}.png')
        plt.savefig(plot_filename)
        plt.show()
        plt.close(fig)

    print(f"Plots saved in {output_plot_directory}")

'''
Fuction to know if find the same events that PNSN or not

'''

def compare_detection_times(unique_detections_df, matched_files_with_locations, output_directory):
    

    original_dates = [pd.to_datetime(item['original_date']) for item in matched_files_with_locations]

    # Inicializar una columna para los resultados de las coincidencias
    unique_detections_df['Match-PNSN'] = 'No'  # Todos inicializados a 'No'
    
    # Iterar sobre cada tiempo de detección única
    for index, row in unique_detections_df.iterrows():
        detection_time = pd.to_datetime(row['Unique Detection Time (UTC)'])
        has_match = False  # Indicador para detectar si se encontró alguna coincidencia

        # Comparar cada tiempo de detección con todas las fechas originales
        for original_date in original_dates:
            if abs((detection_time - original_date).total_seconds()) <= 10:
                has_match = True
                break  # Salir del bucle si se encuentra una coincidencia

        # Actualizar el DataFrame con el resultado de la coincidencia
        unique_detections_df.at[index, 'Match-PNSN'] = 'Yes' if has_match else 'No'

    # Guardar el DataFrame actualizado en el directorio existente
    output_csv_path = os.path.join(output_directory, 'resulting_detections.csv')
    unique_detections_df.to_csv(output_csv_path, index=False)
    
    print(f"DataFrame saved in {output_csv_path}")
    return unique_detections_df






