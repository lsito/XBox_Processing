import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
import math
import os
from nptdms import TdmsFile
from datetime import datetime


def get_channel_data(channel):
    #converts the raw ADC data to the needed values with the corresponding coefficients 
    # and returnd the data.
    match channel.properties['Scale_Type']:
        case "Polynomial":
            if 'Scale_Coeff_c2' in channel.properties:
                coeff = [
                    float(channel.properties['Scale_Coeff_c2']),
                    float(channel.properties['Scale_Coeff_c1']),
                    float(channel.properties['Scale_Coeff_c0']),
                ]
            else:
                coeff = [
                    float(channel.properties['Scale_Coeff_c1']),
                    float(channel.properties['Scale_Coeff_c0']),
                ]
            return np.polyval(coeff, (channel[:]).astype('float'))
            
        case "Logarithmic":
            coeff= [
                float(channel.properties['Log_Coeff_A']),
                float(channel.properties['Log_Coeff_b']),
                float(channel.properties['Log_Coeff_C'])
            ]
            return np.add(np.multiply(coeff[0],np.exp(np.multiply(coeff[1],channel[:]))),coeff[2])
            
        case "None":
            return channel[:]
        case _:
            print("Dot defined Scale type.")
            return np.nan

def get_time_length(channel):
    #returns the total length of the signal. This value should be always the same.
    samples = channel.properties['wf_samples']
    increment=channel.properties['wf_increment']
    return samples*increment

def get_PKI_info(channel):
    #returns information about the chosen PKI channel:
    #* 'PKI_total_power' is the integral of the complete PKI signal and therefore the complete energy
    #* 'PKI_length' is the legth of the PKI signal (in s)
    #* 'PKI_peak' is the peak of the PKI signal
    #* 'PKI_mean' is the mean of the PKI signal
    #* 'PKI_start' is the start of the PKI signal (in s)
    try:
        channel_data=get_channel_data(channel)
    except:
        channel_data=[0,0]
    incremental_rate=channel.properties['wf_increment']
    PKI_total_power=np.trapz(channel_data,dx=incremental_rate)
    PKI_peak=np.amax(channel_data)
    PKI_start=np.argmax(np.greater(channel_data,PKI_peak*0.5))*incremental_rate
    flat_area=channel_data[np.greater(channel_data,PKI_peak*0.5)]
    PKI_length=len(flat_area)
    if (flat_area[np.ceil(PKI_length/4).astype('int'):np.floor((PKI_length/4)*3).astype('int')]).size==0:
        #print('no PKI mean area found')
        PKI_mean=np.nan
        PKI_length=0
    else:
        PKI_mean = np.mean(flat_area[np.ceil(PKI_length/4).astype('int'):np.floor((PKI_length/4)*3).astype('int')])
    
    return [PKI_total_power,PKI_length*incremental_rate,PKI_peak,PKI_mean,PKI_start]
 
def get_signal_start(channel_data,PKI_start,mid_pulse,threshold):
    #used to find the start of a signal. For this the 'PKI_start' is used as a reference point
    start_area=channel_data[np.amax([PKI_start-50,0],0):(PKI_start+mid_pulse)+1]
    start_peak_value=np.amax(start_area)
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

def signal_data_start(channel,PKIA_start,mid_pulse):
    channel_data=get_channel_data(channel)
    incremental_rate=channel.properties['wf_increment']
    
    #convert PKI start and mid_pulse in points according to the incremental rate
    PKIA_start=np.round(np.divide(PKIA_start,incremental_rate)).astype('int')
    mid_pulse=np.round(np.divide(mid_pulse,incremental_rate)).astype('int')
    threshold_signal_start=0.8

    #find start
    signal_start=get_signal_start(channel_data,PKIA_start,mid_pulse,threshold_signal_start)
    return signal_start*incremental_rate

def signal_data(channel,log_type,PKIA_start,compression,mid_pulse,prev_channel,signal, threshold_BD=0.5):
    #returns parameters of one signal:
    #* 'total_power' is the total energy of this signal (integration)
    #* 'peak_value' is the peak of the signal
    #* 'peak_length' is the length of the signal, depending on compression. If the pulse is not compressed, this is the full pulse, if the pule is compressed only the compressed part lenth is taken into account
    #* 'flat_mean' the mean of the signal, with the same interval definition as the 'peak_lentgh'
    #* 'signal_start' uses the 'get_signal_start' function
    #* 'time_BD' uses the 'get_BD_pos' function to find the BD position, if the signal was clarified as a BD (i.e. log==2)
    #* 'threshold_BD' can be used to adjust the threshold for finding a BD, default is 0.5
    #* 'signal' selects type of signal, options are 'I'or 'R'
    channel_data=get_channel_data(channel)
    incremental_rate=channel.properties['wf_increment']
    total_power=np.trapz(channel_data,dx=incremental_rate)
    
    #convert PKI start and mid_pulse in points according to the incremental rate
    PKIA_start=np.round(np.divide(PKIA_start,incremental_rate)).astype('int')
    mid_pulse=np.round(np.divide(mid_pulse,incremental_rate)).astype('int')
    threshold_signal_start=0.8

    match signal:
            case "I":
                #depending if compression was done or not, the area to search for the peak changes
                if compression==True:
                    start=PKIA_start+mid_pulse
                    end=PKIA_start+mid_pulse+mid_pulse+mid_pulse
                    peak_area=channel_data[start:end]
                else:
                    peak_area=channel_data[np.max([PKIA_start-50,0]):PKIA_start+mid_pulse+mid_pulse+50]

                if peak_area.size == 0:
                    peak_value = 0 
                else:
                    peak_value = np.amax(peak_area)
                
                flat_area=peak_area[np.greater(peak_area,peak_value*threshold_signal_start)]
                peak_length=len(flat_area)
                
                if (flat_area[np.ceil(peak_length/4).astype('int'):np.floor((peak_length/4)*3).astype('int')]).size==0:
                    #print('No flat area found')
                    flat_mean=np.nan
                else:
                    flat_mean=np.mean(flat_area[np.ceil(peak_length/4).astype('int'):np.floor((peak_length/4)*3).astype('int')])

                #find start
                signal_start=get_signal_start(channel_data,PKIA_start,mid_pulse,threshold_signal_start)

                if log_type==2: #breakdown case
                    prev_channel_data=get_channel_data(prev_channel)
                    time_BD = get_BD_pos(prev_channel_data, channel_data, threshold_BD,signal_start,get_signal_start(prev_channel_data,PKIA_start,mid_pulse,threshold_signal_start))
                else:
                    time_BD=0

                return [total_power,peak_value,peak_length*incremental_rate,flat_mean,signal_start*incremental_rate,time_BD*incremental_rate]
            
            case "R":
                if (channel_data[:np.max([PKIA_start-20,0])]).size==0:
                    #print('No PSR offset found')
                    PSR_offset=np.nan
                else:
                    PSR_offset=np.mean(channel_data[:np.max([PKIA_start-20,0])])
                total_power_including_offset=np.trapz(np.subtract(channel_data,PSR_offset),dx=incremental_rate)

                if log_type==2: #breakdown case
                    time_BD=get_BD_pos(get_channel_data(prev_channel),channel_data,threshold_BD)
                else:
                    time_BD=0
                return [total_power,total_power_including_offset,time_BD*incremental_rate]
            
            case _:
                print("Not defined signal type. Please, select I or R")


def group_data_event(group,prev_group,Xbox,structure_stand=0,plotting=False):
    if Xbox==2:
        return group_data_Xbox2(group,prev_group,plotting)
    elif Xbox==3:
        return group_data_Xbox3(group,prev_group,plotting,structure_stand)

def group_data_Xbox2(group,prev_group,plotting):
    this_data=pd.DataFrame()
    this_data['pulse_count']=[group.properties['Pulse Count']]
    this_data['log_type']=[group.properties['Log Type']]
    this_data['timestamp']=[group.properties['Timestamp']]

    #total_length=get_time_length(group['PKIA_amp'])
    #get PKI data, PKR data and compression knowledge
    this_data[['PKI_total_power','PKI_length','PKI_peak','PKI_mean','PKI_start']]=get_PKI_info(group['PKI_amp'])
    this_data[['PKR_total_power','PKR_length','PKR_peak','PKR_mean','PKR_start']]=get_PKI_info(group['PKR_log'])
    #check if the pulse was compressed
    compression=this_data['PKI_length'][0]>1e-6
    mid_pulse=np.divide(this_data['PKI_length'][0],2)
    
    log_type=this_data['log_type'][0]
    PKI_start=this_data['PKI_start'][0]
    #PSI data
    this_data[['PSIA_total_power','PSIA_peak','PSIA_peak_length','PSIA_mean_flat','PSIA_start','PSIA_t']]=signal_data(group['PSIA_amp'],log_type ,PKI_start,compression,mid_pulse,prev_group['PSIA_amp'],'I')
    this_data[['PSIB_total_power','PSIB_peak','PSIB_peak_length','PSIB_mean_flat','PSIB_start','PSIB_t']]=signal_data(group['PSIB_amp'],log_type ,PKI_start,compression,mid_pulse,prev_group['PSIB_amp'],'I')
    # PSR data
    this_data[['PSRA_total_power','PSRA_total_power_including_offset','PSRA_t']]=signal_data(group['PSRA_amp'],log_type,PKI_start,compression, mid_pulse,prev_group['PSRA_amp'],'R')
    this_data[['PSRB_total_power','PSRB_total_power_including_offset','PSRB_t']]=signal_data(group['PSRB_amp'],log_type,PKI_start,compression, mid_pulse,prev_group['PSRB_amp'],'R')
    #PEI data
    this_data[['PEIA_total_power','PEIA_peak','PEIA_peak_length','PEIA_mean_flat','PEIA_start','PEIA_t']]=signal_data(group['PEIA_amp'],log_type,PKI_start,compression,mid_pulse,prev_group['PEIA_amp'],'I')
    this_data[['PEIB_total_power','PEIB_peak','PEIB_peak_length','PEIB_mean_flat','PEIB_start','PEIB_t']]=signal_data(group['PEIB_amp'],log_type,PKI_start,compression,mid_pulse,prev_group['PEIB_amp'],'I')
    #DC up and DC down data
    #inverted to the negative values
    this_data['DC_up_total_A'] = [np.trapz(get_channel_data(group['DC_UP_A']),dx=group['DC_UP_A'].properties['wf_increment'])*-1]
    this_data['DC_up_total_B'] = [np.trapz(get_channel_data(group['DC_UP_B']),dx=group['DC_UP_B'].properties['wf_increment'])*-1]
    this_data['DC_up_peak_A']=[np.amin(get_channel_data(group['DC_UP_A']))*-1]
    this_data['DC_up_peak_B']=[np.amin(get_channel_data(group['DC_UP_B']))*-1]
    this_data['DC_down_total_A'] = [np.trapz(get_channel_data(group['DC_DOWN_A']),dx=group['DC_UP_A'].properties['wf_increment'])*-1]
    this_data['DC_down_total_B'] = [np.trapz(get_channel_data(group['DC_DOWN_B']),dx=group['DC_UP_B'].properties['wf_increment'])*-1]
    this_data['DC_down_peak_A']=[np.amin(get_channel_data(group['DC_DOWN_A']))*-1]
    this_data['DC_down_peak_B']=[np.amin(get_channel_data(group['DC_DOWN_B']))*-1]
    
    BD_loc=0
    this_data['BD_struct']=[0]
    if log_type==2 and "BD_Flag_PERA_log" in group.properties: #BD
        if group.properties['BD_Flag_PERA_log']==1: 
            this_data['BD_struct']=1
            BD_loc=np.bitwise_or(BD_loc,1) #change bit at pos 0
        if group.properties['BD_Flag_PERB_log']==1: 
            this_data['BD_struct']=2
            BD_loc=np.bitwise_or(BD_loc,1) #change bit at pos 0
        if (group.properties['BD_Flag_DC_DOWN_A']==1 or group.properties['BD_Flag_DC_UP_A']==1):
            this_data['BD_struct']=1
            BD_loc=np.bitwise_or(BD_loc,2) #change bit at pos 1
        if (group.properties['BD_Flag_DC_DOWN_B']==1 or group.properties['BD_Flag_DC_UP_B']==1):
            this_data['BD_struct']=2
            BD_loc=np.bitwise_or(BD_loc,2) #change bit at pos 1
        if group.properties['BD_Flag_PSRA_log']==1:
            this_data['BD_struct']=1
            BD_loc=np.bitwise_or(BD_loc,4) #change bit at pos 2
        if group.properties['BD_Flag_PSRB_log']==1:
            this_data['BD_struct']=2
            BD_loc=np.bitwise_or(BD_loc,4) #change bit at pos 2
        if group.properties['BD_Flag_PKR_log']==1:
            BD_loc=np.bitwise_or(BD_loc,8) #change bit at pos 3
    
    this_data['BD_loc']=[BD_loc]
    if plotting==True and log_type==2:
        if this_data['BD_struct'][0]==1:
            plot_BD_pulse_Xbox2(group,this_data['PEIA_t'],this_data['PSRA_t'],this_data['BD_struct'][0])
        elif this_data['BD_struct'][0]==2:
            plot_BD_pulse_Xbox2(group,this_data['PEIB_t'],this_data['PSRB_t'],this_data['BD_struct'][0])
    return this_data

def group_data_Xbox3(group,prev_group,plotting,structure_stand):
    this_data=pd.DataFrame()
    this_data['pulse_count']=[group.properties['Pulse Count']]
    this_data['log_type']=[group.properties['Log Type']]
    this_data['timestamp']=[group.properties['Timestamp']]

    #total_length=get_time_length(group['PKIA_amp'])
    #get PKI data, PKR data and compression knowledge

    this_data[['PKIA_total_power','PKIA_length','PKIA_peak','PKIA_mean','PKIA_start']]=get_PKI_info(group['PKIA_amp'])
    this_data[['PKIB_total_power','PKIB_length','PKIB_peak','PKIB_mean','PKIB_start']]=get_PKI_info(group['PKIB_amp'])
    this_data[['PKRA_total_power','PKRA_length','PKRA_peak','PKRA_mean','PKRA_start']]=get_PKI_info(group['PKRA'])
    this_data[['PKRB_total_power','PKRB_length','PKRB_peak','PKRB_mean','PKRB_start']]=get_PKI_info(group['PKRB'])
    #check if the pulse was compressed
    compression=((this_data['PKIA_length'][0]>1e-6)|(this_data['PKIB_length'][0]>1e-6))
    mid_pulse=np.divide(np.maximum(this_data['PKIA_length'][0],this_data['PKIB_length'][0]),2)


    #PLRA and PLRB
    this_data['PLRA_total_power']=[np.trapz(get_channel_data(group['PLRA']),dx=group['PLRA'].properties['wf_increment'])]
    this_data['PLRB_total_power']=[np.trapz(get_channel_data(group['PLRB']),dx=group['PLRB'].properties['wf_increment'])]
    
    log_type=this_data['log_type'][0]
    PKI_start=np.maximum(this_data['PKIA_start'][0],this_data['PKIB_start'][0])
    if structure_stand==1:
        #PSI data
        this_data[['PSIA_total_power','PSIA_peak','PSIA_peak_length','PSIA_mean_flat','PSIA_start','PSIA_t']]=signal_data(group['PSI_amp'],log_type ,PKI_start,compression,mid_pulse,prev_group['PSI_amp'],'I')
        # PSR data
        this_data[['PSRA_total_power','PSRA_total_power_including_offset','PSRA_t']]=signal_data(group['PSR_amp'],log_type,PKI_start,compression, mid_pulse,prev_group['PSR_amp'],'R')
        #PEI data
        this_data[['PEIA_total_power','PEIA_peak','PEIA_peak_length','PEIA_mean_flat','PEIA_start','PEIA_t']]=signal_data(group['PEI_amp'],log_type ,PKI_start,compression,mid_pulse,prev_group['PEI_amp'],'I')
        #DC up and DC down data
        #inverted to the negative values
        this_data['DC_up_total_A'] = [np.trapz(get_channel_data(group['DC_UP']),dx=group['DC_UP'].properties['wf_increment'])*-1]
        this_data['DC_up_peak_A']=[np.amin(get_channel_data(group['DC_UP']))*-1]
        this_data['DC_down_total_A'] = [np.trapz(get_channel_data(group['DC_DOWN']),dx=group['DC_DOWN'].properties['wf_increment'])*-1]
        this_data['DC_down_peak_A']=[np.amin(get_channel_data(group['DC_DOWN']))*-1]
    elif structure_stand==2:
        #PSI data
        this_data[['PSIB_total_power','PSIB_peak','PSIB_peak_length','PSIB_mean_flat','PSIB_start','PSIB_t']]=signal_data(group['PSI_amp'],log_type,PKI_start,compression,mid_pulse,prev_group['PSI_amp'],'I')
        # PSR data
        this_data[['PSRB_total_power','PSRB_total_power_including_offset','PSRB_t']]=signal_data(group['PSR_amp'],log_type,PKI_start,compression, mid_pulse,prev_group['PSR_amp'],'R')
        #PEI data
        this_data[['PEIB_total_power','PEIB_peak','PEIB_peak_length','PEIB_mean_flat','PEIB_start','PEIB_t']]=signal_data(group['PEI_amp'],log_type ,PKI_start,compression,mid_pulse,prev_group['PEI_amp'],'I')
        #DC up and DC down data
        #inverted to the negative values
        this_data['DC_up_total_B'] = [np.trapz(get_channel_data(group['DC_UP']),dx=group['DC_UP'].properties['wf_increment'])*-1]
        this_data['DC_up_peak_B']=[np.amin(get_channel_data(group['DC_UP']))*-1]
        this_data['DC_down_total_B'] = [np.trapz(get_channel_data(group['DC_DOWN']),dx=group['DC_DOWN'].properties['wf_increment'])*-1]
        this_data['DC_down_peak_B']=[np.amin(get_channel_data(group['DC_DOWN']))*-1]

    BD_loc=0
    this_data['BD_struct']=[0]

    if log_type==2 and "BD_PKRA" in group.properties: #BD
        this_data['BD_struct']=structure_stand
        if group.properties['BD_PERA']==True or group.properties['BD_PERB']==True: 
            BD_loc=np.bitwise_or(BD_loc,1) #change bit at pos 0
        if (group.properties['BD_DC_DOWN']==True or group.properties['BD_DC_UP']==True):
            BD_loc=np.bitwise_or(BD_loc,2) #change bit at pos 1
        if (group.properties['BD_PSR_amp']==True or group.properties['BD_PSR_ph']==True):
            BD_loc=np.bitwise_or(BD_loc,4) #change bit at pos 2
        if group.properties['BD_PLRA']==True or group.properties['BD_PLRB']==True:
            BD_loc=np.bitwise_or(BD_loc,8) #change bit at pos 3
        if group.properties['BD_PKRA']==True or group.properties['BD_PKRB']==True:
            BD_loc=np.bitwise_or(BD_loc,8) #change bit at pos 3

    this_data['BD_loc']=[BD_loc]
    if plotting==True and log_type==2:
        if structure_stand==1:
            plot_BD_pulse_XBox3(group,this_data['PEIA_t'],this_data['PSRA_t'],0)
        elif structure_stand==2:
            plot_BD_pulse_XBox3(group,this_data['PEIB_t'],this_data['PSRB_t'],0)
    return this_data

def process_event_data(load_directory,save_directory,startdate,enddate,Xbox,structure_stand,plotting):
    #process all event data within chosen timeframe
    for filename in os.listdir(load_directory):
        ending=os.path.splitext(filename)[1]
        if filename=='aux.tdms':
            continue
        if filename=='old':
            continue
        
        if structure_stand==2:
            event="EventDataB"
        elif structure_stand==1:
            event="EventDataA"
        else:
            event="EventDataA"

        if event not in filename:    
            continue
        current_date=(filename.split("_")[1]).split('.')[0]
        print('File: ',filename)
        
        if ending==".tdms" and event in filename and int(current_date)>=startdate and int(current_date)<enddate:
            tic = time.time()
            path=os.path.join(load_directory,filename)
            save_data=os.path.splitext(filename)[0]
            #loop over all groups
            with TdmsFile.open(path) as tdms_file:
                all_groups = tdms_file.groups()
                all_data=pd.DataFrame()
                
                # loops over groups
                for i in range(len(all_groups)):
                    all_data=pd.concat([all_data,group_data_event(all_groups[i],all_groups[i-1],Xbox,structure_stand,plotting)],ignore_index=True)

            all_data.to_hdf(save_directory,key=save_data,mode='a')
                
            toc = time.time()
            print('Calculations done. Time taken:', math.floor(toc - tic),"seconds")
        else: 
            print("File not used")

def plot_BD_pulse_Xbox2(current_group,PEI_t,PSR_t,structure):
    #used to plot the BD positions while running the script. This way the correct location can be verified.
    if structure==1:
        PSI_x=np.arange(0,current_group['PSIA_amp'].properties['wf_samples'])*current_group['PSIA_amp'].properties['wf_increment']
        PSI_y=get_channel_data(current_group['PSIA_amp'])
        PSR_x=np.arange(0,current_group['PSRA_amp'].properties['wf_samples'])*current_group['PSRA_amp'].properties['wf_increment']
        PSR_y=get_channel_data(current_group['PSRA_amp'])
        PEI_x=np.arange(0,current_group['PEIA_amp'].properties['wf_samples'])*current_group['PEIA_amp'].properties['wf_increment']
        PEI_y=get_channel_data(current_group['PEIA_amp'])
    elif structure==2:
        PSI_x=np.arange(0,current_group['PSIB_amp'].properties['wf_samples'])*current_group['PSIB_amp'].properties['wf_increment']
        PSI_y=get_channel_data(current_group['PSIB_amp'])
        PSR_x=np.arange(0,current_group['PSRB_amp'].properties['wf_samples'])*current_group['PSRB_amp'].properties['wf_increment']
        PSR_y=get_channel_data(current_group['PSRB_amp'])
        PEI_x=np.arange(0,current_group['PEIB_amp'].properties['wf_samples'])*current_group['PEIB_amp'].properties['wf_increment']
        PEI_y=get_channel_data(current_group['PEIB_amp'])
            
    max_part=np.max(PSI_y)
    plt.plot(PSI_x,PSI_y)
    plt.plot(PEI_x,PEI_y)
    plt.plot(PSR_x,PSR_y)
    plt.plot((PEI_t,PEI_t),(0,max_part),'k-')
    plt.plot((PSR_t,PSR_t),(0,max_part),'y-')
    plt.xlim([0,3e-6])
    plt.legend(['PSI','PEI','PSR','PEI_t','PSR_t'])
    plt.title('XBox-2 Structure '+str(structure))
    plt.xlabel("Time [s]")
    plt.ylabel("Power [W]")
    plt.show()

def plot_BD_pulse_XBox3(current_group,PEI_t,PSR_t):
    #used to plot the BD positions while running the script. This way the correct location can be verified.
    PSI_x=np.arange(0,current_group['PSI_amp'].properties['wf_samples'])*current_group['PSI_amp'].properties['wf_increment']
    PSI_y=get_channel_data(current_group['PSI_amp'])
    PSR_x=np.arange(0,current_group['PSR_amp'].properties['wf_samples'])*current_group['PSR_amp'].properties['wf_increment']
    PSR_y=get_channel_data(current_group['PSR_amp'])
    PEI_x=np.arange(0,current_group['PEI_amp'].properties['wf_samples'])*current_group['PEI_amp'].properties['wf_increment']
    PEI_y=get_channel_data(current_group['PEI_amp'])
    
    max_part=np.max(PSI_y)
    plt.plot(PSI_x,PSI_y)
    plt.plot(PEI_x,PEI_y)
    plt.plot(PSR_x,PSR_y)
    plt.plot((PEI_t,PEI_t),(0,max_part),'k-')
    plt.plot((PSR_t,PSR_t),(0,max_part),'k-')
    plt.legend(['PSI','PEI','PSR','PEI_t','PSR_t'])
    plt.show()

def load_event_data(load_directory):
    #used to load all the event data
    event_data=pd.DataFrame()
    with pd.HDFStore(load_directory) as store:
        event_data = pd.concat([store.get(filename) for filename in store.keys()])

    #reset index
    event_data.reset_index(drop=True, inplace=True)
    return event_data

def group_data_trend_xbox2(group):
    #used to extract all the data of the file
    #add or change channel names here
    this_data=pd.DataFrame()
    this_data['pulse_count']=group['Pulse Count'][:]
    this_data['timestamp_trend']=group['Timestamp'][:]
    this_data['Pressure_line']=group['Line Pressure'][:]
    this_data['Pressure_struct_A']=group['Structure A Pressure'][:]
    this_data['Pressure_struct_B']=group['Structure B Pressure'][:]
    this_data['Pressure_splitter_B4']=group['B4 Splitter Pressure'][:]
    this_data['Pressure_load_1']=group['Load 1 Pressure'][:]
    this_data['Pressure_load_2']=group['Load 2 Pressure'][:]
    this_data['Temp_power_splitter']=group['Power Splitter Temp'][:]
    this_data['Temp_struct_A']=group['Structure A Temp'][:]
    this_data['Temp_struct_B']=group['Structure B Temp'][:]
    this_data['Temp_load_1']=group['Load 1 Temp'][:]
    this_data['Temp_load_2']=group['Load 2 Temp'][:]
    try:
        this_data["Temp_WaterIn_A"]=group['Water In A Temp'][:]
    except:
        this_data["Temp_WaterIn_A"]=np.nan
    try:
        this_data["Temp_WaterIn_B"]=group['Water In B Temp'][:]
    except:
        this_data["Temp_WaterIn_B"]=np.nan
    try:
        this_data["Temp_WaterOutSP_A"]=group['Water Out SP A Temp'][:]
    except:
        this_data["Temp_WaterOutSP_A"]=np.nan
    try:
        this_data["Temp_WaterOutSP_B"]=group['Water Out SP B Temp'][:]
    except:
        this_data["Temp_WaterOutSP_B"]=np.nan
    try:
        this_data["Temp_WaterOutLP_A"]=group['Water Out LP A Temp'][:]
    except:
        this_data["Temp_WaterOutLP_A"]=np.nan
    try:
        this_data["Temp_WaterOutLP_B"]=group['Water Out LP B Temp'][:]
    except:
        this_data["Temp_WaterOutLP_B"]=np.nan

    return this_data

def group_data_trend_xbox3(group,structure_stand):
    #used to extract all the data of the file
    this_data=pd.DataFrame()
    this_data['timestamp_trend']=group['Timestamp'][:]
    this_data['Pressure_line_1']=group['Line 1 Pressure'][:]
    this_data['Pressure_line_2']=group['Line 2 Pressure'][:]
    try:
        this_data['Room_temp_A']=group['DUT 1 Room Temp']
    except:
        this_data['Room_temp_A']=np.empty(np.shape(this_data['timestamp_trend'])[0])
    try:
        this_data['Room_temp_B']=group['DUT 2 Room Temp']
    except:
        this_data['Room_temp_B']=np.empty(np.shape(this_data['timestamp_trend'])[0])
    if structure_stand==1:
        this_data['pulse_count']=group['Pulse Count 1'][:]
        this_data['Pressure_load_A']=group['Load 1 Pressure'][:]
        this_data['Pressure_struct_A']=group['DUT 1 Pressure'][:]
        this_data['Temp_struct_in_A']=group['DUT 1 Surf In Temp'][:]
        this_data['Temp_struct_out_A']=group['DUT 1 Surf Out Temp'][:]
        this_data['Temp_water_in_struct_A']=group['DUT 1 Water In Temp'][:]
        this_data['Temp_load_A']=group['Load 1 RF In Temp'][:]
        try:
            this_data['Flow_struct_A']=group['DUT 1 Flow']
        except:
            this_data['Flow_struct_A']=np.empty(np.shape(this_data['timestamp_trend'])[0])
    elif structure_stand==2:
        this_data['pulse_count']=group['Pulse Count 2'][:]
        this_data['Pressure_load_B']=group['Load 2 Pressure'][:]
        this_data['Pressure_struct_B']=group['DUT2 Pressure'][:]
        this_data['Temp_struct_in_B']=group['DUT 2 Surf In Temp'][:]
        this_data['Temp_struct_out_B']=group['DUT 2 Surf Out Temp'][:]
        this_data['Temp_water_in_struct_B']=group['DUT 2 Water In Temp'][:]
        this_data['Temp_load_B']=group['Load 2 RF In Temp'][:]
        try:
            this_data['Flow_struct_B']=group['DUT 2 Flow']
        except:
            this_data['Flow_struct_B']=np.empty(np.shape(this_data['timestamp_trend'])[0])
    return this_data

def load_trend_data(directory_trend,save_directory,startdate,enddate,Xbox,structure_stand):
    #used to load all trend data
    trend_data=pd.DataFrame()
    for filename in os.listdir(directory_trend):
        ending=os.path.splitext(filename)[1]
        current_date=(filename.split("_")[1]).split('.')[0]
        #print(current_date)
        if ending==".tdms" and "Trend" in filename and int(current_date)>=startdate and int(current_date)<enddate:
            tic = time.time()
            data_name=os.path.splitext(filename)[0]
            print('File: ',data_name)
            path = os.path.join(directory_trend,filename)
            with TdmsFile.open(path) as tdms_file:
                all_groups = tdms_file.groups()
                # loops over groups
                for i in range(len(all_groups)):
                    if not np.isnan(all_groups[i]['Timestamp'][:]).any():
                        if Xbox==2:
                            trend_data=pd.concat([trend_data,group_data_trend_xbox2(all_groups[i])],ignore_index=True) 
                        elif Xbox==3:
                            trend_data=pd.concat([trend_data,group_data_trend_xbox3(all_groups[i],structure_stand)],ignore_index=True)
                    
            toc = time.time()
            print('Calculations done. Time taken:', math.floor(toc - tic),"seconds")   
        else: 
            pass
    #adjust timestamp to fit the event data timestamp
    trend_data['timestamp_trend']=pd.to_datetime(trend_data['timestamp_trend'],unit='s',origin=datetime(1904,1,1))
    trend_data.to_hdf(save_directory,key=data_name,mode='a')
    return trend_data

def add_gradient(input_data,P_ref,G_ref,Xbox,structure_stand):
    if Xbox==2:
        input_data['gradient_A']=np.sqrt(input_data["PSIA_peak"] / P_ref) * G_ref
        input_data['gradient_B']=np.sqrt(input_data["PSIB_peak"] / P_ref) * G_ref
    elif Xbox==3:
        if structure_stand==1:
            input_data['gradient_A']=np.sqrt(input_data["PSIA_peak"] / P_ref) * G_ref
        elif structure_stand==2:
            input_data['gradient_B']=np.sqrt(input_data["PSIB_peak"] / P_ref) * G_ref
    return input_data

def merge_data(save_directory,save_directory_trend,directory_trend,Xbox,structure_name,structure_stand):
    #used to merge and edit all data
    #* Adjust the values fpr P-ref and G_ref depending on the structure here!
    event_data=load_event_data(save_directory)
    trend_data=load_trend_data(directory_trend,save_directory_trend,Xbox,structure_stand)
    event_data['pulse_count']=pd.to_numeric(event_data['pulse_count'],downcast='integer')
    trend_data['pulse_count']=pd.to_numeric(trend_data['pulse_count'],downcast='integer')
    event_data=event_data.sort_values(by=['pulse_count'])
    trend_data=trend_data.sort_values(by=['pulse_count'])
    all_data=pd.merge_asof(event_data,trend_data,on='pulse_count',direction='backward')

    if structure_name=='Xbox2_TD31N3N4' or structure_name=='Xbox3_TD31N1_L1' or structure_name=='Xbox3_TD31N2_L2':
        #calcualtion of gradient
        P_ref = 36.1*1e6  # [MW]
        G_ref = 72*1e6  # MV/m
    else:
        print("New values for P_ref and G_ref for the new structure are needed!")

    all_data=add_gradient(all_data,P_ref,G_ref,Xbox,structure_stand)

    all_data.reset_index(drop=True, inplace=True)
    return all_data

def calculate_BDR(input_data,Xbox,structure_stand):
    if Xbox==2:
        return calculate_BDR_Xbox2(input_data)
    elif Xbox==3:
        return calculate_BDR_Xbox3(input_data,structure_stand)

def calculate_BDR_Xbox2(input_data):
    #used to calculate the BDR
    #* DC condition for threshold of DC up and DC down can be adjusted here!
    #calculate BDR and cummul. BD and append
    #Threshold for DC up and DC down to be considered:
    dc_up_threshold = 140
    dc_down_threshold = 100
    bd_loc = input_data["BD_loc"].values
    bd_struct = input_data["BD_struct"].values
    dc_up_peak_A = input_data["DC_up_peak_A"].values
    dc_down_peak_A = input_data["DC_down_peak_A"].values
    dc_up_peak_B = input_data["DC_up_peak_B"].values
    dc_down_peak_B = input_data["DC_down_peak_B"].values
    
    ##PERA/PERB #Load
    bd_loc = np.where((bd_loc&1)==1, 1, bd_loc)
    # BD with enough DC current
    dc_condition_A = np.logical_and(bd_struct == 1, np.logical_or(dc_up_peak_A >= dc_up_threshold, dc_down_peak_A >= dc_down_threshold))
    dc_condition_B = np.logical_and(bd_struct == 2, np.logical_or(dc_up_peak_B >= dc_up_threshold, dc_down_peak_B >= dc_down_threshold))
    dc_condition = np.logical_or(dc_condition_A, dc_condition_B)
    bd_loc = np.where(((bd_loc&2)==2) & (dc_condition), 2, bd_loc)
    # BD without enough current and no other BD
    dc_without_current_condition = (bd_loc==2) & (~dc_condition)
    bd_loc = np.where(dc_without_current_condition, 16, bd_loc)   # BDs that 
    # PSR -> without BDC
    bd_loc = np.where((bd_loc&4)==4, 4, bd_loc)
    # HyPC
    bd_loc = np.where((bd_loc&8)==8, 8, bd_loc)
    # Update the "BD_loc" column in the DataFrame
    input_data["BD_loc"] = bd_loc
    input_data["cum_BD_Load_A"] = np.cumsum((bd_loc == 1) & (bd_struct == 1))
    input_data["cum_BD_Load_B"] = np.cumsum((bd_loc == 1)& (bd_struct == 2))
    input_data["cum_BD_DUT_withDC_A"] = np.cumsum((bd_loc == 2)& (bd_struct == 1))
    input_data["cum_BD_DUT_withDC_B"] = np.cumsum((bd_loc == 2)& (bd_struct == 2))
    input_data["cum_BD_DUT_withoutDC_A"] = np.cumsum((bd_loc == 4)& (bd_struct == 1))
    input_data["cum_BD_DUT_withoutDC_B"] = np.cumsum((bd_loc == 4)& (bd_struct == 2))
    input_data["cum_BD_HyPC"] = np.cumsum((bd_loc == 8) )
    input_data["BDR_DUT_A"] = 0
    input_data["BDR_DUT_B"] = 0
    input_data["BDR_Load_A"] = 0
    input_data["BDR_Load_B"] = 0
    input_data["BDR_HyPC"] = 0
    amount_BDR_sum = 1e6
    first_pulse_count = input_data.loc[0, "pulse_count"]
    last_pulse_j = 0
    pulse_counts = input_data["pulse_count"].values
    for i in range(1, input_data.shape[0]):
        current_pulse_count = input_data.loc[i, "pulse_count"]
        if current_pulse_count - amount_BDR_sum >= first_pulse_count:
            last_pulse_j = np.argmax(pulse_counts >= (current_pulse_count - amount_BDR_sum))
        input_data.loc[i, "BDR_DUT_A"] = ((input_data.loc[i, "cum_BD_DUT_withDC_A"] + input_data.loc[i, "cum_BD_DUT_withoutDC_A"]) - (input_data.loc[last_pulse_j, "cum_BD_DUT_withDC_A"] + input_data.loc[last_pulse_j, "cum_BD_DUT_withoutDC_A"])) / amount_BDR_sum
        input_data.loc[i, "BDR_DUT_B"] = ((input_data.loc[i, "cum_BD_DUT_withDC_B"] + input_data.loc[i, "cum_BD_DUT_withoutDC_B"]) - (input_data.loc[last_pulse_j, "cum_BD_DUT_withDC_B"] + input_data.loc[last_pulse_j, "cum_BD_DUT_withoutDC_B"])) / amount_BDR_sum
        input_data.loc[i, "BDR_Load_A"] = (input_data.loc[i, "cum_BD_Load_A"] - input_data.loc[last_pulse_j, "cum_BD_Load_A"]) / amount_BDR_sum
        input_data.loc[i, "BDR_Load_B"] = (input_data.loc[i, "cum_BD_Load_B"] - input_data.loc[last_pulse_j, "cum_BD_Load_B"]) / amount_BDR_sum
        input_data.loc[i, "BDR_HyPC"] = (input_data.loc[i, "cum_BD_HyPC"] - input_data.loc[last_pulse_j, "cum_BD_HyPC"]) / amount_BDR_sum
    return input_data

def calculate_BDR_Xbox3(input_data,structure_stand):
    #used to calculate the BDR
    #* DC condition for threshold of DC up and DC down can be adjusted here!
    #calculate BDR and cummul. BD and append
    #Threshold for DC up and DC down to be considered:
    dc_up_threshold = 140
    dc_down_threshold = 100
    bd_loc = input_data["BD_loc"].values
    
    if structure_stand==1:
        dc_up_peak = input_data["DC_up_peak_A"].values
        dc_down_peak = input_data["DC_down_peak_A"].values
    elif structure_stand==2:
        dc_up_peak = input_data["DC_up_peak_B"].values
        dc_down_peak = input_data["DC_down_peak_B"].values
    ##PERA/PERB #Load
    bd_loc = np.where((bd_loc&1)==1, 1, bd_loc)
    # DC with enough DC current

    dc_condition = np.logical_or(dc_up_peak >= dc_up_threshold, dc_down_peak >= dc_down_threshold)
    
    bd_loc = np.where(((bd_loc&2)==2) & (dc_condition), 2, bd_loc)
    # DC without enough current and no other BD
    dc_without_current_condition = (bd_loc==2) & (~dc_condition)
    bd_loc = np.where(dc_without_current_condition, 16, bd_loc)
    # PSR -> without BDC
    bd_loc = np.where((bd_loc&4)==4, 4, bd_loc)
    # HyPC
    bd_loc = np.where((bd_loc&8)==8, 8, bd_loc)
    # Update the "BD_loc" column in the DataFrame
    input_data["BD_loc"] = bd_loc

    if structure_stand==1:
        input_data["cum_BD_Load_A"] = np.cumsum((bd_loc == 1) )
        input_data["cum_BD_DUT_withDC_A"] = np.cumsum((bd_loc == 2))
        input_data["cum_BD_DUT_withoutDC_A"] = np.cumsum((bd_loc == 4))
        input_data["cum_BD_HyPC_A"] = np.cumsum((bd_loc == 8) )
    elif structure_stand==2:
        input_data["cum_BD_Load_B"] = np.cumsum((bd_loc == 1))
        input_data["cum_BD_DUT_withDC_B"] = np.cumsum((bd_loc == 2))
        input_data["cum_BD_DUT_withoutDC_B"] = np.cumsum((bd_loc == 4))
        input_data["cum_BD_HyPC_B"] = np.cumsum((bd_loc == 8))
    if structure_stand==1:
        input_data["BDR_DUT_A"] = 0
        input_data["BDR_Load_A"] = 0
        input_data["BDR_HyPC_A"] = 0
    elif structure_stand==2:
        input_data["BDR_DUT_B"] = 0
        input_data["BDR_Load_B"] = 0
        input_data["BDR_HyPC_B"] = 0
    amount_BDR_sum = 1e6
    first_pulse_count = input_data.loc[0, "pulse_count"]
    last_pulse_j = 0
    pulse_counts = input_data["pulse_count"].values
    for i in range(1, input_data.shape[0]):
        current_pulse_count = input_data.loc[i, "pulse_count"]
        if current_pulse_count - amount_BDR_sum >= first_pulse_count:
            last_pulse_j = np.argmax(pulse_counts >= (current_pulse_count - amount_BDR_sum))
        if structure_stand==1:
            input_data.loc[i, "BDR_DUT_A"] = ((input_data.loc[i, "cum_BD_DUT_withDC_A"] + input_data.loc[i, "cum_BD_DUT_withoutDC_A"]) - (input_data.loc[last_pulse_j, "cum_BD_DUT_withDC_A"] + input_data.loc[last_pulse_j, "cum_BD_DUT_withoutDC_A"])) / amount_BDR_sum
            input_data.loc[i, "BDR_Load_A"] = (input_data.loc[i, "cum_BD_Load_A"] - input_data.loc[last_pulse_j, "cum_BD_Load_A"]) / amount_BDR_sum
            input_data.loc[i, "BDR_HyPC_A"] = (input_data.loc[i, "cum_BD_HyPC_A"] - input_data.loc[last_pulse_j, "cum_BD_HyPC_A"]) / amount_BDR_sum
        elif structure_stand==2:
            input_data.loc[i, "BDR_DUT_B"] = ((input_data.loc[i, "cum_BD_DUT_withDC_B"] + input_data.loc[i, "cum_BD_DUT_withoutDC_B"]) - (input_data.loc[last_pulse_j, "cum_BD_DUT_withDC_B"] + input_data.loc[last_pulse_j, "cum_BD_DUT_withoutDC_B"])) / amount_BDR_sum
            input_data.loc[i, "BDR_Load_B"] = (input_data.loc[i, "cum_BD_Load_B"] - input_data.loc[last_pulse_j, "cum_BD_Load_B"]) / amount_BDR_sum
            input_data.loc[i, "BDR_HyPC_B"] = (input_data.loc[i, "cum_BD_HyPC_B"] - input_data.loc[last_pulse_j, "cum_BD_HyPC_B"]) / amount_BDR_sum
    return input_data

def calculate_BD_pos(input_data,Xbox,structure_stand, t_fill):
    #calculated the BD position based on PSR-PEI
    if Xbox==2:
        input_data["BD_time"]=np.where(
                input_data["BD_struct"]==1,
                (input_data["PSRA_t"]-input_data["PEIA_t"]+t_fill)/2,
                (input_data["PSRB_t"]-input_data["PEIB_t"]+t_fill)/2)
    elif Xbox==3:
        if structure_stand==1:
            input_data["BD_time"]=(input_data["PSRA_t"]-input_data["PEIA_t"]+t_fill)/2
        elif structure_stand==2:
            input_data["BD_time"]=(input_data["PSRB_t"]-input_data["PEIB_t"]+t_fill)/2
    return input_data
    
def group_data_xbox2(group):
    #to extract all data out of one group for plotting
    #Used in GUI
    log_type=group.properties['Log Type']
    timestamp=group.properties['Timestamp']
    pulse_count=group.properties['Pulse Count']
    #time_array=get_time_array(group['PSI_amp'])

    #PKIA and PKIB
    PKI_x=np.arange(0,group['PKI_amp'].properties['wf_samples'])*group['PKI_amp'].properties['wf_increment']
    PKI_y=get_channel_data(group['PKI_amp'])
    PKR_x=np.arange(0,group['PKR_log'].properties['wf_samples'])*group['PKR_log'].properties['wf_increment']
    PKR_y=get_channel_data(group['PKR_log'])
    
    #PSI, PRS, PEI
    PSIA_x=np.arange(0,group['PSIA_amp'].properties['wf_samples'])*group['PSIA_amp'].properties['wf_increment']
    PSIA_y=get_channel_data(group['PSIA_amp'])
    PSRA_x=np.arange(0,group['PSRA_amp'].properties['wf_samples'])*group['PSRA_amp'].properties['wf_increment']
    PSRA_y=get_channel_data(group['PSRA_amp'])
    PEIA_x=np.arange(0,group['PEIA_amp'].properties['wf_samples'])*group['PEIA_amp'].properties['wf_increment']
    PEIA_y=get_channel_data(group['PEIA_amp'])
    PSIB_x=np.arange(0,group['PSIB_amp'].properties['wf_samples'])*group['PSIB_amp'].properties['wf_increment']
    PSIB_y=get_channel_data(group['PSIB_amp'])
    PSRB_x=np.arange(0,group['PSRB_amp'].properties['wf_samples'])*group['PSRB_amp'].properties['wf_increment']
    PSRB_y=get_channel_data(group['PSRB_amp'])
    PSRB_total=np.trapz(get_channel_data(group['PSRB_amp']),dx=group['PSRB_amp'].properties['wf_increment'])
    PEIB_x=np.arange(0,group['PEIB_amp'].properties['wf_samples'])*group['PEIB_amp'].properties['wf_increment']
    PEIB_y=get_channel_data(group['PEIB_amp'])

    #log files
    PERA_log_x=np.arange(0,group['PERA_log'].properties['wf_samples'])*group['PERA_log'].properties['wf_increment']
    PERA_log_y=get_channel_data(group['PERA_log'])
    PERB_log_x=np.arange(0,group['PERB_log'].properties['wf_samples'])*group['PERB_log'].properties['wf_increment']
    PERB_log_y=get_channel_data(group['PERB_log'])
    PSRA_log_x=np.arange(0,group['PSRA_log'].properties['wf_samples'])*group['PSRA_log'].properties['wf_increment']
    PSRA_log_y=get_channel_data(group['PSRA_log'])
    PSRB_log_x=np.arange(0,group['PSRB_log'].properties['wf_samples'])*group['PSRB_log'].properties['wf_increment']
    PSRB_log_y=get_channel_data(group['PSRB_log'])
    
    #DC_up, DC_down
    DC_UP_A_x=np.arange(0,group['DC_UP_A'].properties['wf_samples'])*group['DC_UP_A'].properties['wf_increment']
    DC_UP_A_y=get_channel_data(group['DC_UP_A'])
    DC_DOWN_A_x=np.arange(0,group['DC_DOWN_A'].properties['wf_samples'])*group['DC_DOWN_A'].properties['wf_increment']
    DC_DOWN_A_y=get_channel_data(group['DC_DOWN_A'])
    DC_UP_B_x=np.arange(0,group['DC_UP_B'].properties['wf_samples'])*group['DC_UP_B'].properties['wf_increment']
    DC_UP_B_y=get_channel_data(group['DC_UP_B'])
    DC_DOWN_B_x=np.arange(0,group['DC_DOWN_B'].properties['wf_samples'])*group['DC_DOWN_B'].properties['wf_increment']
    DC_DOWN_B_y=get_channel_data(group['DC_DOWN_B'])
    return [log_type,timestamp,pulse_count,PKI_x,PKI_y,PKR_x,PKR_y,
            PSIA_x,PSIA_y,PSRA_x,PSRA_y,PEIA_x,PEIA_y,
            PSIB_x,PSIB_y,PSRB_x,PSRB_y,PSRB_total,PEIB_x,PEIB_y,
            PERA_log_x,PERA_log_y,PERB_log_x,PERB_log_y,PSRA_log_x,PSRA_log_y,PSRB_log_x,PSRB_log_y,
            DC_UP_A_x,DC_UP_A_y,DC_DOWN_A_x,DC_DOWN_A_y,DC_UP_B_x,DC_UP_B_y,DC_DOWN_B_x,DC_DOWN_B_y]

def group_data_xbox3(group):
    #to extract all data out of one group for plotting
    #Used in GUI
    log_type=group.properties['Log Type']
    timestamp=group.properties['Timestamp']
    pulse_count=group.properties['Pulse Count']

    #PKIA and PKIB
    PKIA_x=np.arange(0,group['PKIA_amp'].properties['wf_samples'])*group['PKIA_amp'].properties['wf_increment']
    PKIA_y=get_channel_data(group['PKIA_amp'])
    PKIB_x=np.arange(0,group['PKIB_amp'].properties['wf_samples'])*group['PKIB_amp'].properties['wf_increment']
    PKIB_y=get_channel_data(group['PKIB_amp'])
    PKRA_x=np.arange(0,group['PKRA'].properties['wf_samples'])*group['PKRA'].properties['wf_increment']
    PKRA_y=get_channel_data(group['PKRA'])
    PKRB_x=np.arange(0,group['PKRB'].properties['wf_samples'])*group['PKRB'].properties['wf_increment']
    PKRB_y=get_channel_data(group['PKRB'])
    
    #PSI, PRS, PEI
    PSI_x=np.arange(0,group['PSI_amp'].properties['wf_samples'])*group['PSI_amp'].properties['wf_increment']
    PSI_y=get_channel_data(group['PSI_amp'])
    PSR_x=np.arange(0,group['PSR_amp'].properties['wf_samples'])*group['PSR_amp'].properties['wf_increment']
    PSR_y=get_channel_data(group['PSR_amp'])
    PEI_x=np.arange(0,group['PEI_amp'].properties['wf_samples'])*group['PEI_amp'].properties['wf_increment']
    PEI_y=get_channel_data(group['PEI_amp'])
    
    #DC_up, DC_down
    DC_UP_x=np.arange(0,group['DC_UP'].properties['wf_samples'])*group['DC_UP'].properties['wf_increment']
    DC_UP_y=get_channel_data(group['DC_UP'])
    DC_DOWN_x=np.arange(0,group['DC_DOWN'].properties['wf_samples'])*group['DC_DOWN'].properties['wf_increment']
    DC_DOWN_y=get_channel_data(group['DC_DOWN'])
    
    return [log_type,timestamp,pulse_count,
            PKIA_x,PKIA_y,PKIB_x,PKIB_y,PKRA_x,PKRA_y,PKRB_x,PKRB_y,
            PSI_x,PSI_y,PSR_x,PSR_y,PEI_x,PEI_y,
            DC_UP_x,DC_UP_y,DC_DOWN_x,DC_DOWN_y]