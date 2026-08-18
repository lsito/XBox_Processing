# Function for data extraction and save to h5 file

# %% Importing necessary libraries
from tokenize import group
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import time
import math
import os

from collections import defaultdict
from nptdms import TdmsFile
from datetime import datetime

# %% Getting Trend Data

def get_trend_data(data_dir, startdate, enddate, keys):
    """
    Extract and consolidate trend data from TDMS files within a date range.
    
    Scans a directory for TDMS files containing "Trend" in their filename,
    filters by date range, and extracts specified channels into a unified
    pandas DataFrame with properly formatted timestamps.
    
    Parameters
    ----------
    data_dir : str
        Directory path containing the TDMS files to process.
    startdate : int
        Start date in YYYYMMDD format (inclusive). Files with dates >= this
        value will be included.
    enddate : int
        End date in YYYYMMDD format (exclusive). Files with dates < this
        value will be included.
    keys : list of str
        Channel names to extract from the TDMS files. Only channels matching
        these names will be included in the output DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the consolidated trend data with columns:
        - 'Timestamp': datetime64 objects converted from TDMS format
        - Additional columns for each channel specified in `keys`
        All data from matching files and groups are concatenated.
    
    Notes
    -----
    - Filenames must end with '.tdms' and contain 'Trend'
    - Date is extracted from the last underscore-separated segment of the
      filename (before the extension)
    - Files with invalid date formats are silently skipped
    - Timestamps are converted from TDMS format (seconds since 1904-01-01)
      to pandas datetime objects
    - All groups within each TDMS file are processed
    - Files are processed in sorted alphabetical order
    """
    columns = defaultdict(list)

    # Selecting tdms Trend files in given data range
    for filename in sorted(os.listdir(data_dir)):
            
            # Check for .tdms files with "Trend" in the filename
            if (filename.endswith(".tdms") and "Trend" in filename):
                
                # robust-ish date extraction: last '_' chunk of the stem
                stem = os.path.splitext(filename)[0]
                try:
                    file_date = int(stem.split("_")[-1])
                except ValueError:
                    continue
                
                # Check if file date is within range
                if (startdate <= file_date < enddate):
                    
                    file_path = os.path.join(data_dir, filename)
                    # Open the correct tdms file and process groups

                    with TdmsFile.open(file_path) as tdms_file:
                        for group in tdms_file.groups():
                            print('Processing group:', group.name)
    
                            ts = group["Timestamp"][:]
                            n = len(ts)

                            data = {"Timestamp": ts}

                            # (each becomes a raw column with same name)

                            for ch in group.channels():

                                name = ch.name
                                if name == "Timestamp":
                                    continue
                                if name in keys:
                                    data[name] = ch[:]

                            # Append this group's arrays into the global accumulator
                            for k, arr in data.items():
                                columns[k].append(arr)

    # Build final dataframe once
    final_data = {k: np.concatenate(v) for k, v in columns.items()}
    trend_data = pd.DataFrame(final_data)
    trend_data['Timestamp']=pd.to_datetime(trend_data['Timestamp'],unit='s',origin=datetime(1904,1,1))

    return trend_data
# %%
keys = [
    "Timestamp",
    "Line 1 Pressure",
    "Line 2 Pressure",
    "DUT 1 Room Temp",
    "DUT 2 Room Temp",
    # Line 2 Stand
    "Pulse Count 2",
    "Load 2 Pressure",
    "DUT 2 Pressure",
    "StructureBSurf_In Temp",
    "StructureBSurf_Out Temp",
    "StructureBWater_In Temp",
    "Load2 Temp",
    "DUT 2 Flow",
    # Line 1 Stand
    "Pulse Count 1",
    "Load 1 Pressure",
    "DUT 1 Pressure",
    "DUT 1 Surf In Temp",
    "DUT 1 Surf Out Temp",
    "DUT 1 Water In Temp",
    "Load 1 RF In Temp",
    "DUT 1 Flow",
]

structure_name="Xbox3_TD26CIEMAT_L1"

dir1 = "//cernbox-drive/winspaces/x/xboxes/"+structure_name+"/"
dir2 = "/data/"

startdate = 20260108
enddate = 20260109

Xbox = 3
structure_stand = 2
# %%
trend_df = get_trend_data(
    data_dir=dir1,
    startdate=20260107,
    enddate=20260108,
    keys=keys
)
# %%
display(trend_df)
# %% Getting Event Data
def process_event_data(data_dir, startdate, enddate, structure_stand):
    """
    Process all EventData TDMS files within [startdate, enddate) and write each file's
    result to HDF5 using the tdms stem as the key.
    """

    # Scan files deterministically
    #columns = defaultdict(list)
    columns = {}

    # Decide event prefix once
    if structure_stand == 2:
        event_token = "EventDataB"
    else:
        event_token = "EventDataA"

    # Selecting tdms Trend files in given data range
    for filename in sorted(os.listdir(data_dir)):
        # skip known non-data entries
        if filename in ("aux.tdms", "old"):
            continue

        # only tdms with event token
        if (filename.endswith(".tdms") and event_token in filename):
            
            # robust-ish date extraction: last '_' chunk of the stem
            stem = os.path.splitext(filename)[0]
            try:
                file_date = int(stem.split("_")[-1])
            except ValueError:
                continue
                
            # Check if file date is within range
            if (startdate <= file_date < enddate):
                    
                file_path = os.path.join(data_dir, filename)
                # Open the correct tdms file and process groups
                frames = []

                with TdmsFile.open(file_path) as tdms_file:

                    # For BDs I also need previous group
                    prev_group = None
                    for group in tdms_file.groups():
                        
                        if prev_group is None:
                            prev_group = group
                            continue

                        data = group_data_Xbox3(group,prev_group,structure_stand=structure_stand)
        
                        # Update for next iteration
                        prev_group = group

                        # Append this group's arrays into the global accumulator
                        for k, arr in data.items():
                            if k not in columns:
                                columns[k] = []
                            columns[k].append(arr)

                       
    # Build final dataframe once
    # final_data = {k: np.concatenate(v) for k, v in columns.items()}
    event_data = pd.DataFrame(columns)

    return event_data


def get_scaled_channel_data(channel):
    """
    Convert raw ADC channel data to scaled physical values.
    
    Applies scaling transformations based on the channel's scale type using
    coefficients stored in the channel properties.
    
    Parameters
    ----------
    channel : object
        Channel object containing raw data array (accessible via slicing) and
        a 'properties' dict with scaling parameters.
    
    Returns
    -------
    numpy.ndarray
        Scaled channel data as floating-point values, or np.nan if the scale
        type is undefined.
    
    Notes
    -----
    Supported scale types:
    - "Polynomial": Applies polynomial scaling using coefficients c0, c1, and
      optionally c2. For quadratic: c2*x^2 + c1*x + c0. For linear: c1*x + c0.
    - "Logarithmic": Applies exponential scaling: A * exp(b * x) + C
    - "None": Returns raw data without transformation
   
    """
    scale_type = channel.properties.get('Scale_Type')
    raw_data = channel[:]
    
    match scale_type:
        case "Polynomial":
            # Build coefficient list based on available terms
            if 'Scale_Coeff_c2' in channel.properties:
                # Quadratic: c2*x^2 + c1*x + c0
                coeffs = [
                    float(channel.properties['Scale_Coeff_c2']),
                    float(channel.properties['Scale_Coeff_c1']),
                    float(channel.properties['Scale_Coeff_c0']),
                ]
            else:
                # Linear: c1*x + c0
                coeffs = [
                    float(channel.properties['Scale_Coeff_c1']),
                    float(channel.properties['Scale_Coeff_c0']),
                ]
            return np.polyval(coeffs, raw_data.astype(float))
        
        case "Logarithmic":
            # Apply exponential scaling: A * exp(b * x) + C
            A = float(channel.properties['Log_Coeff_A'])
            b = float(channel.properties['Log_Coeff_b'])
            C = float(channel.properties['Log_Coeff_C'])
            
            return A * np.exp(b * raw_data) + C
        
        case "None":
            return raw_data
        
        case _:
            print(f"Warning: Undefined scale type '{scale_type}'. Returning NaN.")
            return np.full_like(raw_data, np.nan, dtype=float)


def get_PKI_info(channel):

    # 1. Scale channel data
    # 2. Return information about the chosen PKI channel:
        #* 'PKI_total_power' is the integral of the complete PKI signal and therefore the complete energy
        #* 'PKI_length' is the legth of the PKI signal (in s)
        #* 'PKI_peak' is the peak of the PKI signal
        #* 'PKI_mean' is the mean of the PKI signal
        #* 'PKI_start' is the start of the PKI signal (in s)

    try:
        channel_data = get_scaled_channel_data(channel)
    except:
        channel_data = [0,0]

    # Get x-axis step size
    delta_x = channel.properties['wf_increment']

    PKI_total_power = np.trapz(channel_data,dx=delta_x)
    PKI_peak = np.amax(channel_data)

    # By definition here, PKI start is where the signal exceeds 50% of peak
    PKI_start = np.argmax(np.greater(channel_data,PKI_peak*0.5)) * delta_x

    # By definition here, PKI length is the width of the region above 50% of peak
    flat_area = channel_data[np.greater(channel_data,PKI_peak*0.5)]
    PKI_length = len(flat_area)

    # Calculate mean of middle 50% of flat_area (25th to 75th percentile positions)
    start_idx = int(np.ceil(PKI_length / 4)) # Rounded up
    end_idx = int(np.floor(PKI_length * 3 / 4)) # Rounded down
    middle_section = flat_area[start_idx:end_idx]

    if middle_section.size == 0:
        # No valid data in middle section
        PKI_mean = np.nan
        PKI_length = 0
    else:
        PKI_mean = np.mean(middle_section)
    
    return [PKI_total_power,
            PKI_length * delta_x,
            PKI_peak,
            PKI_mean,
            PKI_start]

def get_PLR_info(channel):

    # 1. Scale channel data
    # 2. Return information about the chosen PKR channel:

    try:
        channel_data = get_scaled_channel_data(channel)
    except:
        channel_data = [0,0]

    # Get x-axis step size
    delta_x = channel.properties['wf_increment']
    PLR_total_power = np.trapz(channel_data, dx=delta_x)
    
    return [PLR_total_power]

def get_signal_start(channel_data,PKI_start,mid_pulse,threshold):
    # Used to find the start of a signal. 
    # For this the 'PKI_start' is used as a reference point
    start_area = channel_data[np.amax([PKI_start-50,0],0):(PKI_start+mid_pulse)+1]
    start_peak_value = np.amax(start_area)
    return np.argmax(np.greater(channel_data,start_peak_value*threshold))

def get_BD_pos(prev_channel_data, channel_data, threshold,signal_start=0,prev_signal_start=0):
    #used to find the BD position of a signal (PEI or PSR):
    #* For this the current and previous signal are alignes by their 'signal_start', subtracted and the absolute value is taken. Whenever now a rising edge is recognised, the signal is at the BD position. 
    #* Depending on the chosen thresholds, the edge will be recognised earlear or later.
    #* For visualisation, the option of plotting the BD position with the signal can be enabled later
    #alignement of signals
    if signal_start!=prev_signal_start:
        if signal_start>prev_signal_start:
            realignment_factor=signal_start-prev_signal_start
            prev_channel_data=prev_channel_data[:len(prev_channel_data)-(realignment_factor)]
            channel_data=channel_data[(realignment_factor):]
        else:
            realignment_factor=prev_signal_start-signal_start
            channel_data=channel_data[:len(prev_channel_data)-(realignment_factor)]
            prev_channel_data=prev_channel_data[(realignment_factor):]
    else:
        realignment_factor=0
        
    #subtraction and absolute value
    subtracted_data=np.abs(np.subtract(prev_channel_data,channel_data))
    
    peak_BD=np.amax(subtracted_data)
    pos=np.argmax(np.greater(subtracted_data,peak_BD*(threshold)))
    return pos+realignment_factor

def get_signal_I_info(channel,
                      prev_channel,
                      log_type,
                      PKI_start,
                      compression,
                      mid_pulse, 
                      threshold_BD = 0.5,
                      threshold_signal_start = 0.8):
    
    # Getting scaled channel data
    try:
        channel_data = get_scaled_channel_data(channel)
    except:
        channel_data = [0,0]

    delta_x = channel.properties['wf_increment']
    I_total_power = np.trapz(channel_data, dx=delta_x)
    
    # Convert PKI start and mid_pulse in points according to the incremental rate
    PKIA_start = np.round(np.divide(PKI_start,delta_x)).astype('int')
    mid_pulse = np.round(np.divide(mid_pulse,delta_x)).astype('int')
 
    # Area of peak depends on compression
    if compression == True:
        start = PKIA_start + mid_pulse
        end = PKIA_start + mid_pulse + mid_pulse + mid_pulse
        peak_area = channel_data[start:end]
    else:
        peak_area = channel_data[np.max([PKIA_start-50,0]):PKIA_start+mid_pulse+mid_pulse+50]

    if peak_area.size == 0:
        peak_value = 0 
    else:
        peak_value = np.amax(peak_area)
    
    flat_area = peak_area[np.greater(peak_area,peak_value*threshold_signal_start)]
    peak_length = len(flat_area)
    
    if (flat_area[np.ceil(peak_length/4).astype('int'):np.floor((peak_length/4)*3).astype('int')]).size==0:
        flat_mean = np.nan
    else:
        flat_mean = np.mean(flat_area[np.ceil(peak_length/4).astype('int'):np.floor((peak_length/4)*3).astype('int')])

    # Find Signal Start
    signal_start = get_signal_start(channel_data,PKIA_start,mid_pulse,threshold_signal_start)

    # Breakdown case
    if log_type == 2: 
        prev_channel_data = get_scaled_channel_data(prev_channel)
        time_BD = get_BD_pos(prev_channel_data, channel_data, threshold_BD,signal_start,get_signal_start(prev_channel_data,PKIA_start,mid_pulse,threshold_signal_start))
    else:
        time_BD=0

    return [I_total_power,peak_value,peak_length*delta_x,flat_mean,signal_start*delta_x,time_BD*delta_x]
        

def get_signal_R_info(channel,
                      prev_channel,
                      log_type,
                      PKI_start,
                      mid_pulse,
                      threshold_BD = 0.5):
    
    # Getting scaled channel data
    try:
        channel_data = get_scaled_channel_data(channel)
    except:
        channel_data = [0,0]

    delta_x = channel.properties['wf_increment']
    R_total_power = np.trapz(channel_data, dx=delta_x)
    
    # Convert PKI start and mid_pulse in points according to the incremental rate
    PKIA_start = np.round(np.divide(PKI_start,delta_x)).astype('int')
    mid_pulse = np.round(np.divide(mid_pulse,delta_x)).astype('int')

    if (channel_data[:np.max([PKIA_start-20,0])]).size == 0:
        PSR_offset = np.nan
    else:
        PSR_offset = np.mean(channel_data[:np.max([PKIA_start-20,0])])

    R_total_power_including_offset=np.trapz(np.subtract(channel_data,PSR_offset),dx=delta_x)

    if log_type == 2: #breakdown case
        time_BD = get_BD_pos(get_scaled_channel_data(prev_channel),channel_data,threshold_BD)
    else:
        time_BD = 0
    return [R_total_power,R_total_power_including_offset,time_BD*delta_x]
        
def get_DC_data(group):
    # Getting scaled channel data
    try:
        channel_data_DC_UP = get_scaled_channel_data(group['DC_UP'])
        channel_data_DC_DOWN = get_scaled_channel_data(group['DC_DOWN'])
    except:
        channel_data_DC_UP = [0,0]
        channel_data_DC_DOWN = [0,0]

    delta_x_DC_UP = group['DC_UP'].properties['wf_increment']
    delta_x_DC_DOWN = group['DC_DOWN'].properties['wf_increment']

    DC_total_UP = np.trapz(channel_data_DC_UP, dx=delta_x_DC_UP) * (-1)
    DC_peak_UP = np.amin(channel_data_DC_UP) * (-1)
    DC_total_DOWN = np.trapz(channel_data_DC_DOWN, dx=delta_x_DC_DOWN) * (-1)
    DC_peak_DOWN = np.amin(channel_data_DC_DOWN) * (-1)

    return [DC_total_UP, DC_peak_UP, DC_total_DOWN, DC_peak_DOWN]


def group_data_Xbox3(group,prev_group,structure_stand):
    
    # Get properties of the whole group
    data = {"pulse_count": group.properties['Pulse Count'],
            "log_type": group.properties['Log Type'],
            "timestamp": group.properties['Timestamp']
            }
    # Get synthetic info of each channel in the group
    
    channels_PK = ["PKIA_amp", "PKIB_amp", "PKRA", "PKRB"]
    dict_values_PK = ["total_power", "length", "peak", "mean", "start"]

    channels_PLR = ["PLRA", "PLRB"]
    dict_values_PLR = ["total_power"]

    keys = channels_PK + channels_PLR
    
    for name in keys:
        if name in channels_PK:
            pki_info = get_PKI_info(group[name])
            for idx, el in enumerate(dict_values_PK):
                data.update({f"{name}_{el}": pki_info[idx]})
      
        if name in channels_PLR:
            plr_info = get_PLR_info(group[name])
            for idx, el in enumerate(dict_values_PLR):
                data.update({f"{name}_{el}": plr_info[idx]})


    # Check if the pulse was compressed (needed for signal processing)
    compression = ((data['PKIA_amp_length']>1e-6) | 
                    (data['PKIB_amp_length']>1e-6))
    mid_pulse = np.divide(np.maximum(data['PKIA_amp_length'],
                                        data['PKIB_amp_length']),2)
    log_type = data['log_type']
    PKI_start = np.maximum(data['PKIA_amp_start'],data['PKIB_amp_start'])


    dict_values_PSI = ["total_power", "peak", "peak_length", "mean_flat", "start", "t"]
    dict_values_PEI = ["total_power", "peak", "peak_length", "mean_flat", "start", "t"]
    dict_values_PSR = ["total_power", "total_power_including_offset", "t"]

    dict_values_DC = ["DC_up_total", "DC_up_peak", "DC_down_total", "DC_down_peak"]

    if structure_stand == 1:
        
        psi_info = get_signal_I_info(group['PSI_amp'],
                                            prev_group['PSI_amp'],
                                            log_type,
                                            PKI_start,
                                            compression,
                                            mid_pulse
                                            )
        for idx, el in enumerate(dict_values_PSI):
            data.update({f"{name}A_{el}": psi_info[idx]})

     
        # PSR data
        psr_info = get_signal_R_info(group['PSR_amp'],
                                     prev_group['PSR_amp'],
                                     log_type,
                                     PKI_start, 
                                     mid_pulse
                                     )
        for idx, el in enumerate(dict_values_PSR):
            data.update({f"{name}A_{el}": psr_info[idx]})
                                            
        # PEI data
        pei_info = get_signal_I_info(group['PEI_amp'],
                                     prev_group['PEI_amp'],
                                     log_type,
                                     PKI_start,
                                     compression,
                                     mid_pulse
                                     )
        for idx, el in enumerate(dict_values_PEI):
            data.update({f"{name}A_{el}": pei_info[idx]})
        
        # DC up and DC down data
        dc_info = get_DC_data(group)
        for idx, el in enumerate(dict_values_DC):
            data.update({f"{el}_A": dc_info[idx]})
        
    elif structure_stand == 2:
        
        psi_info = get_signal_I_info(group['PSI_amp'],
                                            prev_group['PSI_amp'],
                                            log_type,
                                            PKI_start,
                                            compression,
                                            mid_pulse
                                            )
        for idx, el in enumerate(dict_values_PSI):
            data.update({f"{name}B_{el}": psi_info[idx]})

     
        # PSR data
        psr_info = get_signal_R_info(group['PSR_amp'],
                                     prev_group['PSR_amp'],
                                     log_type,
                                     PKI_start, 
                                     mid_pulse
                                     )
        for idx, el in enumerate(dict_values_PSR):
            data.update({f"{name}B_{el}": psr_info[idx]})
                                            
        # PEI data
        pei_info = get_signal_I_info(group['PEI_amp'],
                                     prev_group['PEI_amp'],
                                     log_type,
                                     PKI_start,
                                     compression,
                                     mid_pulse
                                     )
        for idx, el in enumerate(dict_values_PEI):
            data.update({f"{name}B_{el}": pei_info[idx]})
        
        # DC up and DC down data
        dc_info = get_DC_data(group)
        for idx, el in enumerate(dict_values_DC):
            data.update({f"{el}_B": dc_info[idx]})

    # Breakdown structure
    BD_loc = 0
    data.update({"BD_struct": 0})

    if log_type == 2 and "BD_PKRA" in group.properties: # BD
        data.update({"BD_struct": structure_stand})
        if group.properties['BD_PERA'] == True or group.properties['BD_PERB'] == True: 
            BD_loc = np.bitwise_or(BD_loc,1) #change bit at pos 0
        if (group.properties['BD_DC_DOWN'] == True or group.properties['BD_DC_UP'] == True):
            BD_loc = np.bitwise_or(BD_loc,2) #change bit at pos 1
        if (group.properties['BD_PSR_amp'] == True or group.properties['BD_PSR_ph'] == True):
            BD_loc = np.bitwise_or(BD_loc,4) #change bit at pos 2
        if group.properties['BD_PLRA'] == True or group.properties['BD_PLRB'] == True:
            BD_loc = np.bitwise_or(BD_loc,8) #change bit at pos 3
        if group.properties['BD_PKRA'] == True or group.properties['BD_PKRB'] == True:
            BD_loc = np.bitwise_or(BD_loc,8) #change bit at pos 3

    data.update({"BD_loc": BD_loc})

    # Plotting capability (for the future)

    """
    if plotting==True and log_type==2:
        if structure_stand==1:
            plot_BD_pulse_XBox3(group,this_data['PEIA_t'],this_data['PSRA_t'],0)
        elif structure_stand==2:
            plot_BD_pulse_XBox3(group,this_data['PEIB_t'],this_data['PSRB_t'],0)
    """
    return data
# %%
event_df = process_event_data(
    data_dir=dir1,
    startdate=20260107,
    enddate=20260108,
    structure_stand=1
)

#%% Getting h5 files
save_dir = "../data/event_data.h5"

event_df.to_hdf(save_dir,key="event_data", mode='a')
# %%


import pandas as pd

df_Leo = pd.read_hdf('..\data\event_data.h5')
df_Paz = pd.read_hdf('..\data\Xbox3_TD26CIEMAT_L1\_pandas_new_test.h5')
# %%
