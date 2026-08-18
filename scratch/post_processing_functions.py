import pandas as pd
import numpy as np
import scipy.stats as kde
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors



def load_event_data_with_BD(load_directory):
    #used to load all the event data with the BD information
    event_data=pd.DataFrame()
    with pd.HDFStore(load_directory) as store:
        event_data = pd.concat([store.get(filename) for filename in store.keys()])

    #reset index
    event_data.reset_index(drop=True, inplace=True)
    return event_data


def adjust_variables(input_data,Xbox,structure_stand):
    if Xbox==2:
        input_data["PSIA_peak"]=input_data["PSIA_peak"]*1e-6 #[MW]
        input_data["PSIA_peak_length"]=input_data["PSIA_peak_length"]*1e9 #[ns]
        input_data["PSIA_start"]=input_data["PSIA_start"]*1.25e-9
        input_data["PEIA_start"]=input_data["PEIA_start"]*1.25e-9
        
        input_data["PSIB_peak"]=input_data["PSIB_peak"]*1e-6 #[MW]
        input_data["PSIB_peak_length"]=input_data["PSIB_peak_length"]*1e9 #[ns]
        input_data["PSIB_start"]=input_data["PSIB_start"]*1.25e-9
        input_data["PEIB_start"]=input_data["PEIB_start"]*1.25e-9
    elif Xbox==3:
        if structure_stand==1:
            input_data["PSIA_peak"]=input_data["PSIA_peak"]*1e-6 #[MW]
            input_data["PSIA_peak_length"]=input_data["PSIA_peak_length"]*1e9 #[ns]
            input_data["PSIA_start"]=input_data["PSIA_start"]*6.25e-10
            input_data["PEIA_start"]=input_data["PEIA_start"]*6.25e-10 
        elif structure_stand==2:
            input_data["PSIB_peak"]=input_data["PSIB_peak"]*1e-6 #[MW]
            input_data["PSIB_peak_length"]=input_data["PSIB_peak_length"]*1e9 #[ns]
            input_data["PSIB_start"]=input_data["PSIB_start"]*6.25e-10
            input_data["PEIB_start"]=input_data["PEIB_start"]*6.25e-10 

    return input_data

def add_lost_power(input_data,Xbox,structure_stand):
    if Xbox==2:
        input_data['lost_power_A']=input_data["PSIA_total_power"]-input_data["PSRA_total_power"]-input_data["PEIA_total_power"]
        input_data['lost_power_B']=input_data["PSIB_total_power"]-input_data["PSRB_total_power"]-input_data["PEIB_total_power"]
    elif Xbox==3:
        if structure_stand==1:
            input_data['lost_power_A']=input_data["PSIA_total_power"]-input_data["PSRA_total_power"]-input_data["PEIA_total_power"]
        elif structure_stand==2:
            input_data['lost_power_B']=input_data["PSIB_total_power"]-input_data["PSRB_total_power"]-input_data["PEIB_total_power"]
    return input_data


def filter_PSI(input_data,structure_stand,min_value=0,max_value=100):
    print("a")
    #print(input_data["PSIB_peak"])
    if structure_stand==1:
        output_data=input_data[(input_data["PSIA_peak"]>=min_value)&(input_data["PSIA_peak"]<=max_value)] 
        #output_data["pulse_count"]=np.arange(len(output_data["pulse_count"]))+output_data["pulse_count"][0]
    elif structure_stand==2:
        #print(input_data["PSIB_peak"])
        output_data=input_data[(input_data["PSIB_peak"]>=min_value)&(input_data["PSIB_peak"]<=max_value)]
        #output_data["pulse_count"]=np.arange(len(output_data["pulse_count"]))+output_data["pulse_count"][0]
    return output_data
    
def filter_PKI(input_data,Xbox,structure_stand,min_value=0,max_value=100):
    if Xbox==2:
        output_data= input_data[(input_data["PKI_length"]>=min_value)&(input_data["PKI_length"]<=max_value)]
        #output_data["pulse_count"]=np.arange(len(output_data["pulse_count"]))+output_data["pulse_count"][0]
    elif Xbox==3:
        output_data= input_data[
            ((input_data["PKIA_length"]>=min_value)&(input_data["PKIA_length"]<=max_value))|
            ((input_data["PKIB_length"]>=min_value)&(input_data["PKIB_length"]<=max_value))]
        #output_data["pulse_count"]=np.arange(len(output_data["pulse_count"]))+output_data["pulse_count"][0]
    return output_data



def gaussian_kde_calculation(kernel_factor,input_data,xi,positions):
    values=np.vstack([input_data["pulse_count"],input_data["BD_time"]*1e9 ])#ns
    kernel = kde.gaussian_kde(values, bw_method='silverman') #kernel density estimate, gaussian kernel
    kernel = kde.gaussian_kde(values, bw_method=kernel.factor/kernel_factor)
    zi=np.reshape(kernel(positions).T,xi.shape)
    return zi

def gaussian_kde_position(input_data):
    x_min=np.amin(input_data["pulse_count"])
    x_max=np.amax(input_data["pulse_count"])
    y_min=0 #-5e-8*1e9 #ns
    y_max=6e-8*1e9 #ns
    xbin=3000
    ybin=500
    xi,yi=np.mgrid[x_min:x_max:xbin*1j,y_min:y_max:ybin*1j]
    positions=np.vstack([xi.ravel(), yi.ravel()])
    return [xi,yi,positions]
    


def plot_BDR(input_data,structure_stand,Xbox,left_part,right_part=0,last_data=0,including_last_data=False):
    if structure_stand==1:
        left_part.plot(input_data["pulse_count"],input_data["BDR_DUT_A"],'brown',markersize=0.4)
        left_part.plot(input_data["pulse_count"],input_data["BDR_Load_A"],'green',markersize=0.4)
        if including_last_data:
            right_part.plot(last_data["pulse_count"],last_data["BDR_DUT_A"],'brown',markersize=0.4)
            right_part.plot(last_data["pulse_count"],last_data["BDR_Load_A"],'green',markersize=0.4)
    elif structure_stand==2:
        left_part.plot(input_data["pulse_count"],input_data["BDR_DUT_B"],'brown',markersize=0.4)
        left_part.plot(input_data["pulse_count"],input_data["BDR_Load_B"],'green',markersize=0.4)
        if including_last_data:
            right_part.plot(last_data["pulse_count"],last_data["BDR_DUT_B"],'brown',markersize=0.4)
            right_part.plot(last_data["pulse_count"],last_data["BDR_Load_B"],'green',markersize=0.4)
    if Xbox==2:
        left_part.plot(input_data["pulse_count"],input_data["BDR_HyPC"],'violet',markersize=0.4)
        if including_last_data:
            right_part.plot(last_data["pulse_count"],last_data["BDR_HyPC"],'violet',markersize=0.4)
    elif Xbox==3:
        if structure_stand==1:
            left_part.plot(input_data["pulse_count"],input_data["BDR_HyPC_A"],'violet',markersize=0.4)
            if including_last_data:
                right_part.plot(last_data["pulse_count"],last_data["BDR_HyPC_A"],'violet',markersize=0.4)
        elif structure_stand==2:
            left_part.plot(input_data["pulse_count"],input_data["BDR_HyPC_B"],'violet',markersize=0.4)
            if including_last_data:
                right_part.plot(last_data["pulse_count"],last_data["BDR_HyPC_B"],'violet',markersize=0.4)
    left_part.ticklabel_format(style='sci',scilimits=(0,0), axis='both')
    left_part.set_yscale('log')
    left_part.set_ybound(1e-6,1e-4)
    left_part.yaxis.set_ticks_position('right')
    if including_last_data:
        right_part.legend(['DUT','Load','HyPC'])
        right_part.set_ylabel('BDR',color='brown')
        right_part.yaxis.set_label_position('right')

def plot_peak(input_data,structure_stand,left_part,right_part=0,last_data=0,including_last_data=False):
    if structure_stand==1:
        left_part.plot(input_data["pulse_count"],input_data["PSIA_peak"],'b,')
        if including_last_data:
            right_part.plot(last_data["pulse_count"],last_data["PSIA_peak"],'b,')
    elif structure_stand==2:
        left_part.plot(input_data["pulse_count"],input_data["PSIB_peak"],'b,')
        if including_last_data:
            right_part.plot(last_data["pulse_count"],last_data["PSIB_peak"],'b,')
    if including_last_data:
        left_part.plot((last_data["pulse_count"].iloc[0],last_data["pulse_count"].iloc[0]),(0,60),'k-')
        right_part.set_ylabel('Peak Power [MW]',color='b')
        right_part.set_ybound(0,60)
    left_part.set_ylabel('Peak Power [MW]',color='b')
    left_part.set_ybound(0,60)


def plot_gradient(input_data,structure_stand,left_part,right_part=0,last_data=0,including_last_data=False):
    if structure_stand==1:
        left_part.plot(input_data["pulse_count"],input_data["gradient_A"],'b,')
        if including_last_data:
            right_part.plot(last_data["pulse_count"],last_data["gradient_A"],'b,')
    elif structure_stand==2:
        left_part.plot(input_data["pulse_count"],input_data["gradient_B"],'b,')
        if including_last_data:
            right_part.plot(last_data["pulse_count"],last_data["gradient_B"],'b,')
    if including_last_data:
        left_part.plot((last_data["pulse_count"].iloc[0],last_data["pulse_count"].iloc[0]),(0,120),'k-')
        right_part.set_ylabel('Gradient [MV/m]',color='b')
        right_part.set_ybound(0,120)
    left_part.set_ylabel('Gradient [MV/m]',color='b')
    left_part.set_ybound(0,120)

def plot_peak_length(input_data,structure_stand,left_part,right_part=0,last_data=0,including_last_data=False):
    if structure_stand==1:
        left_part.plot(input_data["pulse_count"],input_data["PSIA_peak_length"],'.',markersize=0.3,color='purple')
        if including_last_data:
            right_part.plot(last_data["pulse_count"],last_data["PSIA_peak_length"],'.',markersize=0.3,color='purple')
    elif structure_stand==2:
        left_part.plot(input_data["pulse_count"],input_data["PSIB_peak_length"],'.',markersize=0.3,color='purple')
        if including_last_data:
            right_part.plot(last_data["pulse_count"],last_data["PSIB_peak_length"],'.',markersize=0.3,color='purple')
    left_part.set_ylabel('Pulse length [ns]',color='purple')
    left_part.set_ybound(0,250)
    if including_last_data:
        right_part.set_ybound(0,250)

def plot_cum_BD(input_data,structure_stand,Xbox,left_part,right_part=0,last_data=0,including_last_data=False):
    if structure_stand==1:
        left_part.plot(input_data["pulse_count"],input_data["cum_BD_DUT_withDC_A"],'.', color='goldenrod',markersize=1)
        left_part.plot(input_data["pulse_count"],input_data["cum_BD_DUT_withoutDC_A"],'.', color='brown',markersize=1)
        left_part.plot(input_data["pulse_count"],input_data["cum_BD_Load_A"],'.',color='green',markersize=1)
        if including_last_data:
            right_part.plot(last_data["pulse_count"],last_data["cum_BD_DUT_withDC_A"],'.', color='goldenrod',markersize=1)
            right_part.plot(last_data["pulse_count"],last_data["cum_BD_DUT_withoutDC_A"],'.', color='brown',markersize=1)
            right_part.plot(last_data["pulse_count"],last_data["cum_BD_Load_A"],'.',color='green',markersize=1)        

    elif structure_stand==2:
        left_part.plot(input_data["pulse_count"],input_data["cum_BD_DUT_withDC_B"],'.', color='goldenrod',markersize=1)
        left_part.plot(input_data["pulse_count"],input_data["cum_BD_DUT_withoutDC_B"],'.', color='brown',markersize=1)
        left_part.plot(input_data["pulse_count"],input_data["cum_BD_Load_B"],'.',color='green',markersize=1)
        if including_last_data:
            right_part.plot(last_data["pulse_count"],last_data["cum_BD_DUT_withDC_B"],'.', color='goldenrod',markersize=1)
            right_part.plot(last_data["pulse_count"],last_data["cum_BD_DUT_withoutDC_B"],'.', color='brown',markersize=1)
            right_part.plot(last_data["pulse_count"],last_data["cum_BD_Load_B"],'.',color='green',markersize=1)
    if Xbox==2:
        left_part.plot(input_data["pulse_count"],input_data["cum_BD_HyPC"],'.',color='violet',markersize=1)
        if including_last_data:
            right_part.plot(last_data["pulse_count"],last_data["cum_BD_HyPC"],'.',color='violet',markersize=1)
    elif Xbox==3:
        if structure_stand==1:
            left_part.plot(input_data["pulse_count"],input_data["cum_BD_HyPC_A"],'.',color='violet',markersize=1)
            if including_last_data:
                right_part.plot(last_data["pulse_count"],last_data["cum_BD_HyPC_A"],'.',color='violet',markersize=1)
        elif structure_stand==2:
            left_part.plot(input_data["pulse_count"],input_data["cum_BD_HyPC_B"],'.',color='violet',markersize=1)
            if including_last_data:
                right_part.plot(last_data["pulse_count"],last_data["cum_BD_HyPC_B"],'.',color='violet',markersize=1)
                right_part.set_ybound(0,None)
    left_part.legend(['DUT BDs with BDC','DUT BDs without BDC','Load BDs','HyPc BDs'],markerscale=5)
    left_part.set_ybound(0,None)
    

def plot_BD_pos(xi,yi,zi,left_part,right_part=0,xi_last=0,yi_last=0,zi_last=0,including_last_data=False):
    left_part.pcolormesh(xi,yi,zi,cmap="Reds",norm=mcolors.PowerNorm(0.5),alpha=1,shading='auto')
    left_part.contour(xi,yi,zi,cmap='gray',alpha=0.2,linewidths=0.5)
    left_part.set_ylabel("BDpos [ns]")
    if including_last_data:
        right_part.pcolormesh(xi_last,yi_last,zi_last,cmap="Reds",norm=mcolors.PowerNorm(0.5),alpha=1,shading='auto')
        right_part.contour(xi_last,yi_last,zi_last,cmap='gray',alpha=0.2,linewidths=0.5)
        right_part.set_ylabel("BDpos [ns]")
        right_part.yaxis.set_label_position('right')
        right_part.yaxis.set_ticks_position('right')

def plot_3_in_1_peak_plot(input_data,last_data,xi,yi,zi,xi_last,yi_last,zi_last,structure_stand,Xbox,save_directory):
    #plotting all in one
    fig = plt.figure(figsize=(15,7.5))
    gs=fig.add_gridspec(3,2,hspace=0,height_ratios=(4,1,1),width_ratios=(2,1),wspace=0.08)
    part1=fig.add_subplot(gs[0,0]) #left top
    part2=fig.add_subplot(gs[1,0], sharex=part1) #left middle
    part3=fig.add_subplot(gs[2,0],sharex=part1) #left bottom
    part4=fig.add_subplot(gs[0,1],sharey=part1) #rigth top
    part5=fig.add_subplot(gs[1,1],sharey=part2, sharex=part4) #right middle
    part6=fig.add_subplot(gs[2,1],sharey=part3,sharex=part4) #rigth bottom

    part4.yaxis.set_label_position('right')
    part4.yaxis.set_ticks_position('right')
    plot_peak(input_data,structure_stand,part1,part4,last_data,including_last_data=True)
    
    twin1_part1=part1.twinx()
    twin1_part1.spines['left'].set_position(('axes',-0.1))
    twin1_part1.yaxis.set_label_position('left')
    twin1_part1.yaxis.set_ticks_position('left')
    twin1_part4=part4.twinx()

    plot_peak_length(input_data,structure_stand,twin1_part1,twin1_part4,last_data,including_last_data=True)
    
    twin2_part1=part1.twinx()
    twin2_part4=part4.twinx()
    twin2_part4.spines['right'].set_position(('axes',1.2))
    twin2_part4.yaxis.set_label_position('right')
    twin2_part4.yaxis.set_ticks_position('right')
    twin2_part4.set_ylabel("Accumulated BDs", color='goldenrod')
    
    plot_cum_BD(input_data,structure_stand,Xbox,twin2_part1,twin2_part4,last_data,including_last_data=True)
    
    plot_BDR(input_data,structure_stand,Xbox,part2,part5,last_data,including_last_data=True)

    plot_BD_pos(xi,yi,zi,part3,part6,xi_last,yi_last,zi_last,including_last_data=True)

    part3.set_xbound(input_data["pulse_count"].iloc[0],input_data["pulse_count"].iat[-1])
    print(input_data["pulse_count"].iat[-1])
    part6.set_xbound(last_data["pulse_count"].iloc[0],last_data["pulse_count"].iat[-1])
    part3.set_xlabel('Pulses')
    part6.set_xlabel('Pulses')

    plt.setp(twin2_part1.get_yticklabels(),visible=False)
    plt.setp(twin2_part1.get_yticklines(),visible=False)
    plt.setp(twin1_part4.get_yticklabels(),visible=False)
    plt.setp(twin1_part4.get_yticklines(),visible=False)
    plt.setp(part4.get_yticklabels(),visible=True)
    plt.setp(part4.get_yticklines(),visible=False)
    plt.setp(part5.get_yticklabels(),visible=False)
    plt.setp(part5.get_yticklines(),visible=False)
    part4.yaxis.set_ticks_position('right')
    plt.setp(twin1_part4.get_xticklabels(),visible=False)
    plt.setp(twin2_part4.get_xticklabels(),visible=False)
    plt.setp(part4.get_xticklabels(),visible=False)
    plt.setp(part5.get_xticklabels(),visible=False)

    plt.setp(twin1_part1.get_xticklabels(),visible=False)
    plt.setp(twin2_part1.get_xticklabels(),visible=False)
    plt.setp(part1.get_xticklabels(),visible=False)
    plt.setp(part2.get_xticklabels(),visible=False)
    if structure_stand==1:
        plt.title("Structure A")
    elif structure_stand==2:
        plt.title("Structure B")
    plt.savefig(save_directory,bbox_inches='tight')
    plt.show()


def plot_3_in_1_peak_plot_nozoom(input_data,last_data,xi,yi,zi,xi_last,yi_last,zi_last,structure_stand,Xbox,save_directory):
    #plotting all in one
    fig = plt.figure(figsize=(15,7.5))
    gs=fig.add_gridspec(3,1,hspace=0,height_ratios=(4,1,1))
    part1=fig.add_subplot(gs[0,0]) #left top
    part2=fig.add_subplot(gs[1,0], sharex=part1) #left middle
    part3=fig.add_subplot(gs[2,0],sharex=part1) #left bottom

    plot_peak(input_data,structure_stand,part1)
    
    twin1_part1=part1.twinx()
    twin1_part1.spines['left'].set_position(('axes',-0.1))
    twin1_part1.yaxis.set_label_position('left')
    twin1_part1.yaxis.set_ticks_position('left')

    plot_peak_length(input_data,structure_stand,twin1_part1)
    
    twin2_part1=part1.twinx()
    twin2_part1.spines['right'].set_position(('axes',1.05))
    twin2_part1.yaxis.set_label_position('right')
    twin2_part1.yaxis.set_ticks_position('right')
    twin2_part1.set_ylabel("Accumulated BDs", color='goldenrod')
    
    plot_cum_BD(input_data,structure_stand,Xbox,twin2_part1)
    
    plot_BDR(input_data,structure_stand,Xbox,part2)

    plot_BD_pos(xi,yi,zi,part3)

    part3.set_xbound(input_data["pulse_count"].iloc[0],input_data["pulse_count"].iat[-1])
    print(input_data["pulse_count"].iat[-1])
    part3.set_xlabel('Pulses')
  

    plt.setp(twin2_part1.get_yticklabels(),visible=True)
    plt.setp(twin2_part1.get_yticklines(),visible=True)
    plt.setp(twin1_part1.get_xticklabels(),visible=False)
    plt.setp(twin2_part1.get_xticklabels(),visible=False)
    plt.setp(part1.get_xticklabels(),visible=False)
    plt.setp(part2.get_xticklabels(),visible=False)
    if structure_stand==1:
        plt.title("Structure A")
    elif structure_stand==2:
        plt.title("Structure B")
    plt.savefig(save_directory,bbox_inches='tight')
    plt.show()


def plot_3_in_1_gradient_plot(input_data,last_data,xi,yi,zi,xi_last,yi_last,zi_last,structure_stand,Xbox,save_directory):
    #plotting all in one gradient
    fig = plt.figure(figsize=(15,7.5))

    gs=fig.add_gridspec(3,2,hspace=0,height_ratios=(4,1,1),width_ratios=(2,1),wspace=0.08)
    part1=fig.add_subplot(gs[0,0]) #left top
    part2=fig.add_subplot(gs[1,0], sharex=part1) #left middle
    part3=fig.add_subplot(gs[2,0],sharex=part1) #left bottotm
    part4=fig.add_subplot(gs[0,1],sharey=part1) #rigth top
    part5=fig.add_subplot(gs[1,1],sharey=part2, sharex=part4) #right middle
    part6=fig.add_subplot(gs[2,1],sharey=part3,sharex=part4) #rigth bottom

    part4.yaxis.set_label_position('right')
    part4.yaxis.set_ticks_position('right')
    plot_gradient(input_data,structure_stand,part1,part4,last_data,including_last_data=True)

    twin1_part1=part1.twinx()
    twin1_part1.spines['left'].set_position(('axes',-0.1))
    twin1_part1.yaxis.set_label_position('left')
    twin1_part1.yaxis.set_ticks_position('left')
    twin1_part4=part4.twinx()
    plot_peak_length(input_data,structure_stand,twin1_part1,twin1_part4,last_data,including_last_data=True)
    
    twin2_part1=part1.twinx()
    twin2_part4=part4.twinx()
    twin2_part4.spines['right'].set_position(('axes',1.2))
    twin2_part4.yaxis.set_label_position('right')
    twin2_part4.yaxis.set_ticks_position('right')
    twin2_part4.set_ylabel("Accumulated BDs", color='goldenrod')
    plot_cum_BD(input_data,structure_stand,Xbox,twin2_part1,twin2_part4,last_data,including_last_data=True)

    plot_BDR(input_data,structure_stand,Xbox,part2,part5,last_data,including_last_data=True)

    plot_BD_pos(xi,yi,zi,part3,part6,xi_last,yi_last,zi_last,including_last_data=True)

    part3.set_xbound(input_data["pulse_count"].iloc[0],input_data["pulse_count"].iat[-1])
    print(input_data["pulse_count"].iat[-1])
    part6.set_xbound(last_data["pulse_count"].iloc[0],last_data["pulse_count"].iat[-1])
    part3.set_xlabel('Pulses')
    part6.set_xlabel('Pulses')
    

    #plt.setp(twin3_part4.get_yticklabels(),visible=True)
    plt.setp(twin1_part4.get_yticklabels(),visible=False)
    plt.setp(twin1_part4.get_yticklines(),visible=False)
    plt.setp(part4.get_yticklabels(),visible=False)
    plt.setp(part4.get_yticklines(),visible=False)
    plt.setp(part5.get_yticklabels(),visible=False)
    plt.setp(part5.get_yticklines(),visible=False)
    plt.setp(twin2_part1.get_yticklabels(),visible=False)
    plt.setp(twin1_part4.get_xticklabels(),visible=False)
    plt.setp(twin2_part4.get_xticklabels(),visible=False)
    plt.setp(part4.get_xticklabels(),visible=True)
    plt.setp(part5.get_xticklabels(),visible=False)
    part4.yaxis.set_ticks_position('right')
    plt.setp(twin1_part1.get_xticklabels(),visible=False)
    plt.setp(twin2_part1.get_xticklabels(),visible=False)
    plt.setp(part1.get_xticklabels(),visible=False)
    plt.setp(part2.get_xticklabels(),visible=False)

    plt.savefig(save_directory,bbox_inches='tight')
    plt.show()

def plot_5_in_1_peak_plot(input_data,last_data,xi,yi,zi,zi_withDC,zi_withoutDC,xi_last,yi_last,zi_last,zi_withDC_last,zi_withoutDC_last,structure_stand,Xbox,save_directory):
    #extra plot with division on location
    #plotting all in one
    fig = plt.figure(figsize=(15,12))

    gs=fig.add_gridspec(5,2,hspace=0,height_ratios=(4,1,1,1,1),width_ratios=(2,1),wspace=0.08)
    part1=fig.add_subplot(gs[0,0]) #left 1st
    part2=fig.add_subplot(gs[1,0], sharex=part1) #left 2nd
    part3=fig.add_subplot(gs[2,0],sharex=part1) #left 3rd
    part4=fig.add_subplot(gs[0,1],sharey=part1) #rigth 1st
    part5=fig.add_subplot(gs[1,1],sharey=part2, sharex=part4) #rigth 2nd
    part6=fig.add_subplot(gs[2,1],sharey=part3,sharex=part4) #rigth 3rd

    part7=fig.add_subplot(gs[3,0],sharex=part1) #left 4th 
    part8=fig.add_subplot(gs[4,0],sharex=part1) #left 5th 
    part9=fig.add_subplot(gs[3,1],sharey=part7,sharex=part4) #rigth 4th
    part10=fig.add_subplot(gs[4,1],sharey=part8,sharex=part4) #rigth 5th

    plot_peak(input_data,structure_stand,part1,part4,last_data,including_last_data=True)

    twin1_part1=part1.twinx()
    twin1_part4=part4.twinx()
    twin1_part1.spines['left'].set_position(('axes',-0.1))
    twin1_part1.yaxis.set_label_position('left')
    twin1_part1.yaxis.set_ticks_position('left')
    plot_peak_length(input_data,structure_stand,twin1_part1,twin1_part4,last_data,including_last_data=True)

    twin2_part1=part1.twinx()
    twin2_part4=part4.twinx()
    twin2_part4.set_ylabel('Accumulated BDs',color='goldenrod')
    plot_cum_BD(input_data,structure_stand,Xbox,twin2_part1,twin2_part4,last_data,including_last_data=True)

    plot_BDR(input_data,structure_stand,Xbox,part2,part5,last_data,including_last_data=True)

    plot_BD_pos(xi,yi,zi,part3,part6,xi_last,yi_last,zi_last,including_last_data=True)

    #plot BD pos with DC
    plot_BD_pos(xi,yi,zi_withDC,part7,part9,xi_last,yi_last,zi_withDC_last,including_last_data=True)

    #plot BD pos without DC
    plot_BD_pos(xi,yi,zi_withoutDC,part8,part10,xi_last,yi_last,zi_withoutDC_last,including_last_data=True)

    
    part8.set_xbound(input_data["pulse_count"].iloc[0],input_data["pulse_count"].iat[-1])
    part10.set_xbound(last_data["pulse_count"].iloc[0],last_data["pulse_count"].iat[-1])
    part3.set_xlabel('Pulses')
    part6.set_xlabel('Pulses')

    plt.setp(twin2_part1.get_yticklabels(),visible=False)
    plt.setp(twin2_part1.get_yticklines(),visible=False)
    plt.setp(twin1_part4.get_yticklabels(),visible=False)
    plt.setp(twin1_part4.get_yticklines(),visible=False)
    plt.setp(part4.get_yticklabels(),visible=False)
    plt.setp(part4.get_yticklines(),visible=False)
    plt.setp(part5.get_yticklabels(),visible=False)
    plt.setp(part5.get_yticklines(),visible=False)

    plt.setp(twin1_part4.get_xticklabels(),visible=False)
    plt.setp(twin2_part4.get_xticklabels(),visible=False)
    plt.setp(part4.get_xticklabels(),visible=False)
    plt.setp(part5.get_xticklabels(),visible=False)

    plt.setp(twin1_part1.get_xticklabels(),visible=False)
    plt.setp(twin2_part1.get_xticklabels(),visible=False)
    plt.setp(part1.get_xticklabels(),visible=False)
    plt.setp(part2.get_xticklabels(),visible=False)

    plt.savefig(save_directory,bbox_inches='tight')
    plt.show()