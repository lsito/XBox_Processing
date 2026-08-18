import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import tkinter as tk  
import numpy as np
import os
from nptdms import TdmsFile
from datetime import datetime

def get_channel_data(channel):
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

def getTotalData(group,Group_type):
    return np.trapz(get_channel_data(group[Group_type]),dx=group[Group_type].properties['wf_increment'])

def get_PKI_info(channel):
    channel_data=get_channel_data(channel)
    PKI_peak=np.amax(channel_data)
    PKI_start=np.argmax(np.greater(channel_data,PKI_peak*0.5))*channel.properties['wf_increment']
    flat_area=channel_data[np.greater(channel_data,PKI_peak*0.5)]
    PKI_length=len(flat_area)
    
    return [PKI_length*channel.properties['wf_increment'],PKI_start]

def get_BD_pos(prev_channel_data, channel_data, channel, threshold,signal_start=0,prev_signal_start=0):
    #alignement of signals
    if signal_start!=prev_signal_start:
        if signal_start>prev_signal_start:
            prev_channel_data=prev_channel_data[:len(prev_channel_data)-(signal_start-prev_signal_start)]
            channel_data=channel_data[(signal_start-prev_signal_start):]
        else:
            channel_data=channel_data[:len(prev_channel_data)-(prev_signal_start-signal_start)]
            prev_channel_data=prev_channel_data[(prev_signal_start-signal_start):]
    #subtraction
    subtracted_data=np.abs(np.subtract(prev_channel_data,channel_data))
    
    peak_BD=np.amax(subtracted_data)
    return np.argmax(np.greater(subtracted_data,peak_BD*(1-threshold)))


def get_signal_start(channel_data,PKIA_start,mid_pulse,threshold):
    start_area=channel_data[np.amax([PKIA_start-50,0]):(PKIA_start+mid_pulse)]
    start_peak_value=np.amax(start_area)
    return np.argmax(np.greater(channel_data,start_peak_value*threshold))

def PSR_signal_data(channel,prev_channel):
    channel_data=get_channel_data(channel)
    p=0.5
    time_BD=get_BD_pos(get_channel_data(prev_channel),channel_data,channel,p)
        
    return time_BD*channel.properties['wf_increment']

def signal_data(channel,PKIA_start,mid_pulse,prev_channel):
    channel_data=get_channel_data(channel)
    incremental_rate=channel.properties['wf_increment']
    PKIA_start=np.round(np.divide(PKIA_start,incremental_rate)).astype('int')
    mid_pulse=np.round(np.divide(mid_pulse,incremental_rate)).astype('int')
    #find start
    signal_start=get_signal_start(channel_data,PKIA_start,mid_pulse,0.8)
    time_BD = get_BD_pos(get_channel_data(prev_channel), channel_data, channel, 0.8,signal_start,get_signal_start(get_channel_data(prev_channel),PKIA_start,mid_pulse,0.8))
    return time_BD*channel.properties['wf_increment']

def group_data_xbox2(group):
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
            DC_UP_A_x,DC_UP_A_y,DC_DOWN_A_x,DC_DOWN_A_y,DC_UP_B_x,DC_UP_B_y,DC_DOWN_B_x,DC_DOWN_B_y]

def group_data_xbox3(group):
    log_type=group.properties['Log Type']
    timestamp=group.properties['Timestamp']
    pulse_count=group.properties['Pulse Count']
    #time_array=get_time_array(group['PSI_amp'])

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
    PER_x=np.arange(0,group['PERA'].properties['wf_samples'])*group['PERA'].properties['wf_increment']
    PER_y=get_channel_data(group['PERA'])
    
    #DC_up, DC_down
    DC_UP_x=np.arange(0,group['DC_UP'].properties['wf_samples'])*group['DC_UP'].properties['wf_increment']
    DC_UP_y=get_channel_data(group['DC_UP'])
    DC_DOWN_x=np.arange(0,group['DC_DOWN'].properties['wf_samples'])*group['DC_DOWN'].properties['wf_increment']
    DC_DOWN_y=get_channel_data(group['DC_DOWN'])
    
    return [log_type,timestamp,pulse_count,
            PKIA_x,PKIA_y,PKIB_x,PKIB_y,PKRA_x,PKRA_y,PKRB_x,PKRB_y,
            PSI_x,PSI_y,PSR_x,PSR_y,PEI_x,PEI_y,PER_x,PER_y,
            DC_UP_x,DC_UP_y,DC_DOWN_x,DC_DOWN_y]

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self,master)
        self.var_day = StringVar(value="2023-06-26")
        self.var_time = StringVar(value="05:15:20")
        self.chosenStruct=IntVar(value=1)
        self.chosenXBox=IntVar(value=2)
        self.checkbox_PSI_total_var=IntVar(value=0)
        self.checkbox_PSR_total_var=IntVar(value=0)
        self.checkbox_PEI_total_var=IntVar(value=0)
        self.checkbox_PKI_total_var=IntVar(value=0)
        self.checkbox_PKR_total_var=IntVar(value=0)
        self.createFigure()
        self.createWidgets()
        
    def apply_figure_settings(self):
        self.part7.yaxis.set_label_position('right')
        self.part7.yaxis.set_ticks_position('right')
        self.part8.yaxis.set_label_position('right')
        self.part8.yaxis.set_ticks_position('right')
        self.part9.yaxis.set_label_position('right')
        self.part9.yaxis.set_ticks_position('right')

        if self.chosenXBox.get()==2:
            self.part1.set_ylim([0,1e6])
            self.part2.set_ylim([0,1e6])
            self.part4.set_ylim([0,1e6])
            self.part5.set_ylim([0,1e6])
            self.part7.set_ylim([0,1e6])
            self.part8.set_ylim([0,1e6])
        else:
            self.part1.set_ylim([0,6e6])
            self.part2.set_ylim([0,6e6])
            self.part4.set_ylim([0,6e6])
            self.part5.set_ylim([0,6e6])
            self.part7.set_ylim([0,6e6])
            self.part8.set_ylim([0,6e6])


        plt.setp(self.part1.get_xticklines(),visible=False)
        plt.setp(self.part1.get_xticklabels(),visible=False)
        plt.setp(self.part2.get_xticklines(),visible=False)
        plt.setp(self.part2.get_xticklabels(),visible=False)
        plt.setp(self.part4.get_xticklines(),visible=False)
        plt.setp(self.part4.get_xticklabels(),visible=False)
        plt.setp(self.part5.get_xticklines(),visible=False)
        plt.setp(self.part5.get_xticklabels(),visible=False)
        plt.setp(self.part7.get_xticklines(),visible=False)
        plt.setp(self.part7.get_xticklabels(),visible=False)
        plt.setp(self.part8.get_xticklines(),visible=False)
        plt.setp(self.part8.get_xticklabels(),visible=False)

        plt.setp(self.part4.get_yticklines(),visible=False)
        plt.setp(self.part4.get_yticklabels(),visible=False)
        plt.setp(self.part5.get_yticklines(),visible=False)
        plt.setp(self.part5.get_yticklabels(),visible=False)  
        plt.setp(self.part6.get_yticklines(),visible=False)
        plt.setp(self.part6.get_yticklabels(),visible=False)

    def createFigure(self):
        fig=plt.figure(figsize=(12,8))
        gs=fig.add_gridspec(3,3,hspace=0,height_ratios=(1,2,1),width_ratios=(1,1,1),wspace=0.05)
        self.part1=fig.add_subplot(gs[0,0]) #left top
        self.part2=fig.add_subplot(gs[1,0], sharex=self.part1) #left middle
        self.part3=fig.add_subplot(gs[2,0],sharex=self.part1) #left bottom

        self.part4=fig.add_subplot(gs[0,1],sharey=self.part1) #middle top
        self.part5=fig.add_subplot(gs[1,1],sharey=self.part2,sharex=self.part4) #middle middle
        self.part6=fig.add_subplot(gs[2,1],sharey=self.part3,sharex=self.part4) #middle bottom

        self.part7=fig.add_subplot(gs[0,2],sharey=self.part1) #right top
        self.part8=fig.add_subplot(gs[1,2],sharey=self.part2,sharex=self.part7) #right middle
        self.part9=fig.add_subplot(gs[2,2],sharey=self.part3,sharex=self.part7) #right bottom

        self.apply_figure_settings()
        self.canvas=FigureCanvasTkAgg(fig,master=root)
        self.canvas.get_tk_widget().grid(row=1,column=1, rowspan=12,columnspan=3)
        self.canvas.draw()

    def createWidgets(self):
        self.titleField = tk.Text(master=root, height=2)
        self.titleField.grid(row=0,column=0, columnspan=4)
        self.var_day_entry = tk.Entry(master=root, width=15, textvariable=self.var_day)
        #self.var_day_entry.insert(0,"2023-06-26")
        self.var_day_entry.grid(row=1,column=0)
        self.XBoxButton2=tk.Radiobutton(master=root,text="XBox2", variable=self.chosenXBox, value=2, command=lambda: self.set_buttons_invisible())
        self.XBoxButton2.grid(row=2,column=0,sticky=SW)
        self.XBoxButton3=tk.Radiobutton(master=root,text="XBox3", variable=self.chosenXBox, value=3, command=lambda: self.set_buttons_invisible())
        self.XBoxButton3.grid(row=3,column=0,sticky=NW)
        self.showButton=tk.Button(master=root, text="Load Day", command=lambda: self.load_day())
        self.showButton.grid(row=4,column=0)
        self.var_time_entry = tk.Entry(master=root, width=15, textvariable=self.var_time)
        

        self.StructureButtonA=tk.Radiobutton(master=root,text="Structure A", variable=self.chosenStruct, value=1,command=lambda: self.show_current_pulse())
        self.StructureButtonB=tk.Radiobutton(master=root,text="Structure B", variable=self.chosenStruct, value=2,command=lambda: self.show_current_pulse())
        self.StructureButtonA.grid(row=6, column=0,sticky=SW)
        self.StructureButtonB.grid(row=7, column=0,sticky=NW)

        self.showButton=tk.Button(master=root, text="Show", command=lambda: self.show_current_pulse())
        self.nextButton=tk.Button(master=root, text="Show next pulse", command=lambda: self.show_next_pulse())
        self.nextBDButton=tk.Button(master=root, text="Show next BD pulse", command=lambda: self.show_next_BDpulse())
        self.prevButton=tk.Button(master=root, text="Show prev pulse", command=lambda: self.get_prev_pulse())
        self.prevBDButton=tk.Button(master=root, text="Show prev BD pulse", command=lambda: self.show_prev_BDpulse())
        self.showBDButton=tk.Button(master=root, text='Show BD position', command=lambda: self.show_BDposition())
        

        self.resultField=tk.Text(master=root,height=5)
        self.resultField.grid(row=13,column=1,columnspan=1,rowspan=3,sticky=NW)

        #create checkboxes
        self.checkbox_PSI_total=tk.Checkbutton(master=root,text='PSI',variable=self.checkbox_PSI_total_var,command=lambda: self.updatePrint())
        self.checkbox_PSR_total=tk.Checkbutton(master=root,text='PSR',variable=self.checkbox_PSR_total_var,command=lambda: self.updatePrint())
        self.checkbox_PEI_total=tk.Checkbutton(master=root,text='PEI',variable=self.checkbox_PEI_total_var,command=lambda: self.updatePrint())
        self.checkbox_PKI_total=tk.Checkbutton(master=root,text='PKI',variable=self.checkbox_PKI_total_var,command=lambda: self.updatePrint())
        self.checkbox_PKR_total=tk.Checkbutton(master=root,text='PKR',variable=self.checkbox_PKR_total_var,command=lambda: self.updatePrint())

        self.set_buttons_invisible()
        #trace the variable
        #self.checkbox_PSI_total_var.trace_add('write',self.updatePrint())
        #self.checkbox_PSI_total_var.trace('w',self.updatePrint)
        #self.checkbox_PSR_total_var.trace('w',self.updatePrint)
        #self.checkbox_PEI_total_var.trace('w',self.updatePrint)
        #self.checkbox_PKI_total_var.trace('w',self.updatePrint)
        #self.checkbox_PKR_total_var.trace('w',self.updatePrint)
        #self.chosenStruct.trace('w',self.show_current_pulse)

    def set_buttons_invisible(self):
        self.var_time_entry.grid_forget()
        self.showButton.grid_forget()
        self.nextButton.grid_forget()
        self.nextBDButton.grid_forget()
        self.prevButton.grid_forget()
        self.prevBDButton.grid_forget()
        self.showBDButton.grid_forget()
        self.checkbox_PSI_total.grid_forget()
        self.checkbox_PSR_total.grid_forget()
        self.checkbox_PEI_total.grid_forget()
        self.checkbox_PKI_total.grid_forget()
        self.checkbox_PKR_total.grid_forget()
        
    def updatePrint(self):
        #print results
        self.resultField.delete("1.0",END)
        if self.checkbox_PSI_total_var.get()==1:
            if self.chosenXBox.get()==2:
                if self.chosenStruct.get()==1:
                    total_data=getTotalData(self.all_groups[self.i],'PSIA_amp')
                else:
                    total_data=getTotalData(self.all_groups[self.i],'PSIB_amp')
            else:
                total_data=getTotalData(self.all_groups[self.i],'PSI_amp')
            self.resultField.insert(END,'Total Power PSI: '+str(total_data)+'\n')
        if self.checkbox_PSR_total_var.get()==1:
            if self.chosenXBox.get()==2:
                if self.chosenStruct.get()==1:
                    total_data=getTotalData(self.all_groups[self.i],'PSRA_amp')
                else:
                    total_data=getTotalData(self.all_groups[self.i],'PSRB_amp')
            else:
                total_data=getTotalData(self.all_groups[self.i],'PSR_amp')
            self.resultField.insert(END,'Total Power PSR: '+str(total_data)+'\n')
        if self.checkbox_PEI_total_var.get()==1:
            if self.chosenXBox.get()==2:
                if self.chosenStruct.get()==1:
                    total_data=getTotalData(self.all_groups[self.i],'PEIA_amp')
                else:
                    total_data=getTotalData(self.all_groups[self.i],'PEIB_amp')
            else:
                total_data=getTotalData(self.all_groups[self.i],'PEI_amp')
            self.resultField.insert(END,'Total Power PEI: '+str(total_data)+'\n')
        if self.checkbox_PKI_total_var.get()==1:
            if self.chosenXBox.get()==2:
                total_data=getTotalData(self.all_groups[self.i],'PKI_amp')
                self.resultField.insert(END,'Total Power PKI: '+str(total_data)+'\n')
            else:
                total_dataA=getTotalData(self.all_groups[self.i],'PKIA_amp')
                total_dataB=getTotalData(self.all_groups[self.i],'PKIB_amp')
                self.resultField.insert(END,'Total Power PKIA: '+str(total_dataA)+'\n')
                self.resultField.insert(END,'Total Power PKIB: '+str(total_dataB)+'\n')
        if self.checkbox_PKR_total_var.get()==1:
            if self.chosenXBox.get()==2:
                total_data=getTotalData(self.all_groups[self.i],'PKR_log')
                self.resultField.insert(END,'Total Power PKR: '+str(total_data)+'\n')
            else:
                total_dataA=getTotalData(self.all_groups[self.i],'PKRA')
                total_dataB=getTotalData(self.all_groups[self.i],'PKRB')
                self.resultField.insert(END,'Total Power PKRA: '+str(total_dataA)+'\n')
                self.resultField.insert(END,'Total Power PKRB: '+str(total_dataB)+'\n')
        
    def load_day(self):
        self.var_day=self.var_day_entry.get()
        adjusted_date=self.var_day.replace('-','')
        if self.chosenXBox.get()==2:
            directory=r"//cernbox-drive/winspaces/x/xboxes/Xbox2_Polarix/"
            #filename=r"EventDataA_"+adjusted_date+".tdms"
            filename=r"EventData_"+adjusted_date+".tdms"
        else:
            if self.chosenStruct.get()==1:
                directory=r"//cernbox-drive/winspaces/x/xboxes/Xbox3_TD26CIEMAT_L1/"
                filename=r"EventDataA_"+adjusted_date+".tdms"
            else:
                directory=r"//cern.ch/dfs/Workspaces/x/Xbox3_TD31N2_L2/"
                filename=r"EventDataB_"+adjusted_date+".tdms"
        path=os.path.join(directory,filename)
        try:
            self.tdms_file=TdmsFile.open(path)
            self.all_groups = self.tdms_file.groups()
            self.titleField.delete("1.0",END)
            self.titleField.insert(END,"Data loaded")
            #show button and entry
            self.var_time_entry.grid(row=5, column=0)
            self.showButton.grid(row=8,column=0)
            #print("Date loaded")
        except:
            self.titleField.delete("1.0",END)
            self.titleField.insert(END,"Wrong date")
            self.set_buttons_invisible()
            #print("Wrong date")
          
    def load_and_update(self):
        if self.var_day!=self.var_day_entry.get():
            self.load_day()
        self.var_time=self.var_time_entry.get()

    def get_pulse_data(self):
        #print(self.var_time.split(':'))
        for i in range(len(self.all_groups)):
            timestamp = self.all_groups[i].properties['Timestamp']
            search_date=self.var_day+'T'+self.var_time
            if timestamp>np.datetime64(search_date):
                self.i=i
                break
            else:
                self.i=i

    def get_next_BD_data(self):
        i=self.i+1
        while i<len(self.all_groups):
        #for i in range(len(self.all_groups)):
            timestamp = self.all_groups[i].properties['Timestamp']
            search_date=self.var_day+'T'+self.var_time
            if timestamp>np.datetime64(search_date) and self.all_groups[i].properties['Log Type']==2:
                self.i=i
                break
            else:
                i=i+1
        if i==len(self.all_groups):
            print('No next BD found')

    def get_prev_BD_data(self):
        i=self.i-1
        while i>=0:
        #for i in range(len(self.all_groups)):
            timestamp = self.all_groups[i].properties['Timestamp']
            search_date=self.var_day+'T'+self.var_time
            if timestamp>np.datetime64(search_date) and self.all_groups[i].properties['Log Type']==2:
                self.i=i
                break
            else:
                i=i-1
        if i<0:
            print('No prev BD found')

    def update_entry(self):
        self.var_time_entry.delete(0,END)
        timestamp=self.all_groups[self.i].properties['Timestamp']
        new_time=timestamp.astype(datetime).strftime('%H:%M:%S')
        self.var_time_entry.insert(0,new_time)
        #print('Hour: ',timestamp.astype(datetime).strftime('%H:%M:%S'))

    def show_current_pulse(self):
        self.load_and_update()
        self.get_pulse_data()
        self.plot()
        self.update_entry()
        self.updatePrint()
        #show invisible buttons
        self.nextButton.grid(row=9,column=0)
        self.nextBDButton.grid(row=10,column=0)
        self.prevButton.grid(row=11,column=0)
        self.prevBDButton.grid(row=12,column=0)
        self.checkbox_PSI_total.grid(row=13,column=2,sticky=W)
        self.checkbox_PSR_total.grid(row=14,column=2,sticky=W)
        self.checkbox_PEI_total.grid(row=15,column=2,sticky=W)
        self.checkbox_PKI_total.grid(row=13,column=3,sticky=W)
        self.checkbox_PKR_total.grid(row=15,column=3,sticky=W)
    
    def show_next_BDpulse(self):
        if self.i+1<len(self.all_groups):
            self.get_next_BD_data()
            self.plot()
            self.update_entry()
            self.updatePrint()
        else:
            print('Last possible data for the day')

    def show_next_pulse(self):
        if self.i+1<len(self.all_groups):
            self.i=self.i+1
            self.plot()
            self.update_entry()
            self.updatePrint()
        else:
            print('Last possible data for the day')

    def show_prev_BDpulse(self):
        if self.i-1>=0:
            self.get_prev_BD_data()
            self.plot()
            self.update_entry()
            self.updatePrint()
        else:
            print('First possible data for the day')

    def get_prev_pulse(self):
        if self.i-1>=0:
            self.i=self.i-1
            self.plot()
            self.update_entry()
            self.updatePrint()
        else:
            print('First possible data for the day')

    def plot_xBox2(self):
        #get second last pulse data
        if self.i-2>=0:
            [log_type,timestamp,pulse_count,PKI_x,PKI_y,PKR_x,PKR_y,
            PSIA_x,PSIA_y,PSRA_x,PSRA_y,PEIA_x,PEIA_y,
            PSIB_x,PSIB_y,PSRB_x,PSRB_y,PSRB_total,PEIB_x,PEIB_y,
            DC_UP_A_x,DC_UP_A_y,DC_DOWN_A_x,DC_DOWN_A_y,DC_UP_B_x,DC_UP_B_y,DC_DOWN_B_x,DC_DOWN_B_y]=group_data_xbox2(self.all_groups[self.i-2])

            self.part1.plot(PKI_x,PKI_y)
            self.part1.plot(PKR_x,PKR_y)
            self.part1.legend(['PKI','PKR'])
            self.part1.set_title('Pulse: '+str(pulse_count)+'\n'+'Log Type: '+str(log_type))

            if self.chosenStruct.get()==1:
                self.part2.plot(PSIA_x,PSIA_y)
                self.part2.plot(PSRA_x,PSRA_y)
                self.part2.plot(PEIA_x,PEIA_y)
                self.part3.plot(DC_UP_A_x,DC_UP_A_y)
                self.part3.plot(DC_DOWN_A_x,DC_DOWN_A_y)
            else:
                self.part2.plot(PSIB_x,PSIB_y)
                self.part2.plot(PSRB_x,PSRB_y)
                self.part2.plot(PEIB_x,PEIB_y)
                self.part3.plot(DC_UP_B_x,DC_UP_B_y)
                self.part3.plot(DC_DOWN_B_x,DC_DOWN_B_y)

            self.part2.legend(["PSIA", "PSRA","PEIA"])            
            self.part3.legend(["DC_Up","DC_down"])

        #get last pulse data
        if self.i-1>=0:
            [log_type,timestamp,pulse_count,PKI_x,PKI_y,PKR_x,PKR_y,
            PSIA_x,PSIA_y,PSRA_x,PSRA_y,PEIA_x,PEIA_y,
            PSIB_x,PSIB_y,PSRB_x,PSRB_y,PSRB_total,PEIB_x,PEIB_y,
            DC_UP_A_x,DC_UP_A_y,DC_DOWN_A_x,DC_DOWN_A_y,DC_UP_B_x,DC_UP_B_y,DC_DOWN_B_x,DC_DOWN_B_y]=group_data_xbox2(self.all_groups[self.i-1])

            self.part4.plot(PKI_x,PKI_y)
            self.part4.plot(PKR_x,PKR_y)
            self.part4.legend(['PKI','PKR'])
            self.part4.set_title('Pulse: '+str(pulse_count)+'\n'+'Log Type: '+str(log_type))

            if self.chosenStruct.get()==1:
                self.part5.plot(PSIA_x,PSIA_y)
                self.part5.plot(PSRA_x,PSRA_y)
                self.part5.plot(PEIA_x,PEIA_y)
                self.part6.plot(DC_UP_A_x,DC_UP_A_y)
                self.part6.plot(DC_DOWN_A_x,DC_DOWN_A_y)
            else:
                self.part5.plot(PSIB_x,PSIB_y)
                self.part5.plot(PSRB_x,PSRB_y)
                self.part5.plot(PEIB_x,PEIB_y)
                self.part6.plot(DC_UP_B_x,DC_UP_B_y)
                self.part6.plot(DC_DOWN_B_x,DC_DOWN_B_y)

            self.part5.legend(["PSIA", "PSRA","PEIA"])            
            self.part6.legend(["DC_Up","DC_down"])

        #get actual pulse data
        [log_type,timestamp,pulse_count,PKI_x,PKI_y,PKR_x,PKR_y,
            PSIA_x,PSIA_y,PSRA_x,PSRA_y,PEIA_x,PEIA_y,
            PSIB_x,PSIB_y,PSRB_x,PSRB_y,PSRB_total,PEIB_x,PEIB_y,
            DC_UP_A_x,DC_UP_A_y,DC_DOWN_A_x,DC_DOWN_A_y,DC_UP_B_x,DC_UP_B_y,DC_DOWN_B_x,DC_DOWN_B_y]=group_data_xbox2(self.all_groups[self.i])
        
        self.part7.plot(PKI_x,PKI_y)
        self.part7.plot(PKR_x,PKR_y)
        self.part7.legend(['PKI','PKR'])
        self.part7.set_title('Pulse: '+str(pulse_count)+'\n'+'Log Type: '+str(log_type))

        if self.chosenStruct.get()==1:
            self.part8.plot(PSIA_x,PSIA_y)
            self.part8.plot(PSRA_x,PSRA_y)
            self.part8.plot(PEIA_x,PEIA_y)
            self.part9.plot(DC_UP_A_x,DC_UP_A_y)
            self.part9.plot(DC_DOWN_A_x,DC_DOWN_A_y)
        else:
            self.part8.plot(PSIB_x,PSIB_y)
            self.part8.plot(PSRB_x,PSRB_y)
            self.part8.plot(PEIB_x,PEIB_y)
            self.part9.plot(DC_UP_B_x,DC_UP_B_y)
            self.part9.plot(DC_DOWN_B_x,DC_DOWN_B_y)

        self.part8.legend(["PSIA", "PSRA","PEIA"])            
        self.part9.legend(["DC_Up","DC_down"])

        #print title
        self.titleField.delete("1.0",END)
        self.titleField.insert(END,timestamp)
        self.titleField.insert(END,'\n')
        self.titleField.insert(END,'XBox 2 ')
        if self.chosenStruct.get()==1:
            self.titleField.insert(END,'Structure A ')
        else:
            self.titleField.insert(END,'Structure B ')
        self.titleField.insert(END,' Log Type: '+str(log_type))

        if log_type==2:
            self.showBDButton.grid(row=13, column=0, sticky=S)
        else:
            self.showBDButton.grid_forget()

    def plot_xBox3(self):
        #get second last pulse data
        if self.i-2>=0:
            [log_type,timestamp,pulse_count,
            PKIA_x,PKIA_y,PKIB_x,PKIB_y,PKRA_x,PKRA_y,PKRB_x,PKRB_y,
            PSI_x,PSI_y,PSR_x,PSR_y,PEI_x,PEI_y,PER_x,PER_y,
            DC_UP_x,DC_UP_y,DC_DOWN_x,DC_DOWN_y]=group_data_xbox3(self.all_groups[self.i-2])

            self.part1.plot(PKIA_x,PKIA_y)
            self.part1.plot(PKIB_x,PKIB_y)
            self.part1.plot(PKRA_x,PKRA_y)
            self.part1.plot(PKRB_x,PKRB_y)
            self.part1.legend(['PKIA','PKIB','PKRA','PKRB'])
            self.part1.set_title('Pulse: '+str(pulse_count)+'\n'+'Log Type: '+str(log_type))

            self.part2.plot(PSI_x,PSI_y)
            self.part2.plot(PSR_x,PSR_y)
            self.part2.plot(PEI_x,PEI_y)
            self.part2.plot(PER_x,PER_y)
        
            self.part3.plot(DC_UP_x,DC_UP_y)
            self.part3.plot(DC_DOWN_x,DC_DOWN_y)

            self.part2.legend(["PSI", "PSR","PEI", "PER"])            
            self.part3.legend(["DC_Up","DC_down"])

        #get last pulse data
        if self.i-1>=0:
            [log_type,timestamp,pulse_count,
            PKIA_x,PKIA_y,PKIB_x,PKIB_y,PKRA_x,PKRA_y,PKRB_x,PKRB_y,
            PSI_x,PSI_y,PSR_x,PSR_y,PEI_x,PEI_y,PER_x,PER_y,
            DC_UP_x,DC_UP_y,DC_DOWN_x,DC_DOWN_y]=group_data_xbox3(self.all_groups[self.i-1])

            self.part4.plot(PKIA_x,PKIA_y)
            self.part4.plot(PKIB_x,PKIB_y)
            self.part4.plot(PKRA_x,PKRA_y)
            self.part4.plot(PKRB_x,PKRB_y)
            self.part4.legend(['PKIA','PKIB','PKRA','PKRB'])
            self.part4.set_title('Pulse: '+str(pulse_count)+'\n'+'Log Type: '+str(log_type))

            self.part5.plot(PSI_x,PSI_y)
            self.part5.plot(PSR_x,PSR_y)
            self.part5.plot(PEI_x,PEI_y)
            self.part5.plot(PER_x,PER_y)
            self.part6.plot(DC_UP_x,DC_UP_y)
            self.part6.plot(DC_DOWN_x,DC_DOWN_y)

            self.part5.legend(["PSI", "PSR","PEI", "PER"])            
            self.part6.legend(["DC_Up","DC_down"])

        #get actual pulse data
        [log_type,timestamp,pulse_count,
            PKIA_x,PKIA_y,PKIB_x,PKIB_y,PKRA_x,PKRA_y,PKRB_x,PKRB_y,
            PSI_x,PSI_y,PSR_x,PSR_y,PEI_x,PEI_y,PER_x,PER_y,
            DC_UP_x,DC_UP_y,DC_DOWN_x,DC_DOWN_y]=group_data_xbox3(self.all_groups[self.i])

        self.part7.plot(PKIA_x,PKIA_y)
        self.part7.plot(PKIB_x,PKIB_y)
        self.part7.plot(PKRA_x,PKRA_y)
        self.part7.plot(PKRB_x,PKRB_y)
        self.part7.legend(['PKIA','PKIB','PKRA','PKRB'])
        self.part7.set_title('Pulse: '+str(pulse_count)+'\n'+'Log Type: '+str(log_type))

        self.part8.plot(PSI_x,PSI_y)
        self.part8.plot(PSR_x,PSR_y)
        self.part8.plot(PEI_x,PEI_y)
        self.part8.plot(PER_x,PER_y)
        self.part9.plot(DC_UP_x,DC_UP_y)
        self.part9.plot(DC_DOWN_x,DC_DOWN_y)

        self.part8.legend(["PSI", "PSR","PEI","PER"])            
        self.part9.legend(["DC_Up","DC_down"])

        #print title
        self.titleField.delete("1.0",END)
        self.titleField.insert(END,timestamp)
        self.titleField.insert(END,'\n')
        self.titleField.insert(END,'XBox 3 ')
        if self.chosenStruct.get()==1:
            self.titleField.insert(END,'Structure A ')
        else:
            self.titleField.insert(END,'Structure B ')
        self.titleField.insert(END,' Log Type: '+str(log_type))

        if log_type==2:
            self.showBDButton.grid(row=13, column=0, sticky=S)
        else:
            self.showBDButton.grid_forget()

    def show_BDposition(self):
        if self.i>0:
            if self.chosenXBox.get()==2:
                [PKI_length,PKI_start]=get_PKI_info(self.all_groups[self.i]['PKI_amp'])
                mid_pulse=np.divide(PKI_length,2)
                if self.chosenStruct.get()==1:
                    PSR_t=PSR_signal_data(self.all_groups[self.i]["PSRA_amp"],self.all_groups[self.i-1]["PSRA_amp"])
                    PEI_t=signal_data(self.all_groups[self.i]["PEIA_amp"],PKI_start,mid_pulse,self.all_groups[self.i-1]["PEIA_amp"])
                else:
                    PSR_t=PSR_signal_data(self.all_groups[self.i]["PSRB_amp"],self.all_groups[self.i-1]["PSRB_amp"])
                    PEI_t=signal_data(self.all_groups[self.i]["PEIB_amp"],PKIA_start,mid_pulse,self.all_groups[self.i-1]["PEIB_amp"])
            else:
                [PKIA_length,PKIA_start]=get_PKI_info(self.all_groups[self.i]['PKIA_amp'])
                mid_pulse=np.divide(PKIA_length,2)
                PSR_t=PSR_signal_data(self.all_groups[self.i]["PSR_amp"],self.all_groups[self.i-1]["PSR_amp"])
                PEI_t=signal_data(self.all_groups[self.i]["PEI_amp"],PKIA_start,mid_pulse,self.all_groups[self.i-1]["PEI_amp"])
            self.part8.plot((PSR_t,PSR_t),(0,1e10),'k-')
            self.part8.plot((PEI_t,PEI_t),(0,1e10),'b-')
            self.part8.legend(['PSR','PEI'])
            self.canvas.draw()
            self.BD_time=(PSR_t-PEI_t+8.4e-8)/2
            self.resultField.insert(END,'BD time: '+str(self.BD_time)+'\n')
        else:
            self.titleField.delete("1.0",END)
            self.titleField.insert(END,'Not enough data')
    def plot(self):
        #clear all plots
        self.part1.clear()
        self.part2.clear()
        self.part3.clear()
        self.part4.clear()
        self.part5.clear()
        self.part6.clear()
        self.part7.clear()
        self.part8.clear()
        self.part9.clear()

        if self.chosenXBox.get()==2:
            self.plot_xBox2()
        else:
            self.plot_xBox3()
        self.apply_figure_settings()

        self.canvas.draw()

root=tk.Tk()
root.title('XBox')
#root.bind("<Return>", showPulse)
app=Application(master=root)
app.mainloop()
