import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
from datetime import datetime, timedelta
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
from scipy.stats import norm
from templatematching import *
from getanalisisfiles import *
from template_maker2 import *


'''
Loading data to run, write here the path for your data.

'''

#base path: it's where the chunk of data is storaging the template files. 

base_path = "/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds" 
#base_path = '/1-fnp/petasaur/p-wd15/rainier-10-14-2023-drive1/'

#output_plot_directory : it's where the plots of corrs correlations are saved as png.
base_csv_plot = "/data/data4/veronica-scratch-rainier/swarm_august2023/results_CC_TMA/"

'''

choose your data window to run the algorithm

'''

start_date_process = "2023-08-25_00.00" 
end_date_process = "2023-08-31_00.00"



## using the fuction get_file_list from getanalisisfiles.py

file_list = get_file_list(base_path,start_date_process,end_date_process)

#  verify # of elements of file_list

if file_list:
    print(f"File list contains {len(file_list)} files.")
else:
    print("File list is empty.")

## choosing templates

events = search(starttime = datetime(2023, 8, 25, 0, 0), 
                endtime = datetime(2023,8,31,0,0),
                #endtime   = datetime.datetime.now(),
                latitude=46.879967,
                longitude=-121.726906,
                maxradius= 35/111.32) 
                #maxradius= 20) 
event_df = get_summary_data_frame(events)
print("Returned %s events" % len(events))

'''
Calculate the templates
'''

# Sort the DataFrame and extract the 'time' column
event_df = event_df.sort_values(by=['time'], ascending=True)
#df_time = event_df['time']

##  Find the files corresponding to the dates with their location in the DataFrame

#use find_files_with_dates from getanalisisfiles.py

matched_files_with_locations = find_files_with_dates(event_df, base_path)

#extracting found files and original dates from matched_files_with_locations

found_files = [item['matched_file'] for item in matched_files_with_locations]
original_dates = [item['original_date'] for item in matched_files_with_locations]


# Parameters of DAS data 

chan_min = 0
chan_max = 2500
channel_number = (chan_max -chan_min)

#define how long you want your templater in our case, more than 3 seconds is recommened to have p and s wave on the template

template_size = 3  # In seconds 

# Sampling frequency, this should be directly from atts of the files in drive1_ds

fs = 20  #hz

samples_per_file = 60*fs # should be integer number
#print(samples_per_file)

'''
Paths

'''
output_plot_directory = os.path.join(base_csv_plot, f"plot-CC-{template_size}sec-{start_date_process}-{end_date_process}")

if not os.path.exists(output_plot_directory):
    os.makedirs(output_plot_directory)
    print(f"Created directory: {output_plot_directory}")
else:
    print(f"Directory already exists: {output_plot_directory}")
    
#files_folder_path: it's where to find the raw data, to look for more events

files_folder_path = '/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds' # where to find the raw data
#files_folder_path = '/1-fnp/petasaur/p-wd15/rainier-10-14-2023-drive1/' #original data with no downsampling                                                   
#output_folder_path = '/data/data4/veronica-scratch-rainier/swarm_august2023/templates-files/template-two-seconds/' 

#output_folder_path:where to save the templates

output_folder_path = '/data/data4/veronica-scratch-rainier/swarm_august2023/templates-files/template-3second-test/' # 
                                   
# Fuction to generate raw templates, template_make2.py

process_files_to_cut(found_files, original_dates, base_path , output_folder_path, chan_min, chan_max, template_size) #this duction just need to be run once!

#filters frequencies

low_cut = 2 #min frequency
high_cut = 9.8 # max frequency


template_list = glob.glob(output_folder_path+'/*')

# Base directory to save files CC
base_directory_cc = '/data/data4/veronica-scratch-rainier/swarm_august2023/results_CC_TMA/'

# Variable folder name for 
folder_name = f'CC_{template_size}sec-tem_{start_date_process}-{end_date_process}'

# Full path
full_path = os.path.join(base_directory_cc, folder_name)

# Create folder if it doesn't exist
if not os.path.exists(full_path):
    os.makedirs(full_path)
print(full_path)

   
#  filter

b, a = butter(2, (low_cut, high_cut), 'bp', fs=fs)


## Buiding outputfiles and correlations for each template on the list

process_files_dos(file_list, template_list, chan_min, chan_max, channel_number, samples_per_file, b, a, full_path)

'''
Analisis of data
'''

#creating timestamps

output_dir = '/data/data4/veronica-scratch-rainier/swarm_august2023/results_CC_TMA/h5_files_timestamps'
output_file_h5 = create_timestamps_h5(file_list, output_dir)

##  Timestamps_to utc

time_utc = convert_timestamps_to_utc(output_file_h5)



base_csv = "/home/velgueta/notebooks/project_Mt-Rainier_DAS/txtfiles"

# Construir el path completo para el directorio de salida

output_directory = os.path.join(base_csv, f"template_{template_size}sec_csv_results-{start_date_process}-{end_date_process}")

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created directory: {output_directory}")
else:
    print(f"Directory already exists: {output_directory}")


# Process folders
process_folders(full_path, time_utc, matched_files_with_locations,template_size,output_directory)

process_folders_and_plot(full_path, fs, time_utc, output_plot_directory, found_files)


# Assuming that 'unique_detections_df' se ha cargado o preparado previamente:
unique_detections_df = pd.read_csv(os.path.join(output_directory, 'unique_detections.csv'))

# Llamar a la función con los datos necesarios
result_df = compare_detection_times(unique_detections_df, matched_files_with_locations, output_directory)


