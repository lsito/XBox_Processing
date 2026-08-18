import numpy as np
import pandas as pd

def unit_scaling(df, structure_stand):
    
    if structure_stand==1:
        df["PSIA_peak"]=df["PSIA_peak"]*1e-6 #[MW]
        df["PSIA_peak_length"]=df["PSIA_peak_length"]*1e9 #[ns]
        df["PSIA_start"]=df["PSIA_start"]*6.25e-10
        df["PEIA_start"]=df["PEIA_start"]*6.25e-10 
    elif structure_stand==2:
        df["PSIB_peak"]=df["PSIB_peak"]*1e-6 #[MW]
        df["PSIB_peak_length"]=df["PSIB_peak_length"]*1e9 #[ns]
        df["PSIB_start"]=df["PSIB_start"]*6.25e-10
        df["PEIB_start"]=df["PEIB_start"]*6.25e-10 

    return df

def add_lost_power(df, structure_stand):
    if structure_stand == 1:
        series = (
            df["PSIA_total_power"]
            - df["PSRA_total_power"]
            - df["PEIA_total_power"]
        )
        return {"lost_power_A": series}

    elif structure_stand == 2:
        series = (
            df["PSIB_total_power"]
            - df["PSRB_total_power"]
            - df["PEIB_total_power"]
        )

    return {"lost_power_B": series}

def add_gradient(df,P_ref,G_ref,structure_stand):
    if structure_stand == 1:
        series = (
            np.sqrt(df["PSIA_peak"] / P_ref) * G_ref
        )
        return {"gradient_A": series}

    elif structure_stand == 2:
        series = (
            np.sqrt(df["PSIB_peak"] / P_ref) * G_ref
        )
        return {"gradient_B": series}


def add_scaled_gradient(df, bdr_ref, pulse_length_ref, structure_stand):
    
    if structure_stand == 1:
        series = (df['gradient_A'] * np.power(df['PSIA_peak_length'] / pulse_length_ref, 1 / 6)
                / np.power(df['BDR_DUT_A']/bdr_ref, 1 / 30)
                )
        return {"scaled_gradient_A": series}

    elif structure_stand == 2:
        series = (df['gradient_B'] * np.power(df['PSIB_peak_length'] / pulse_length_ref, 1 / 6)
                  / np.power(df['BDR_DUT_B']/bdr_ref, 1 / 30)
                )
        return {"scaled_gradient_B": series}
    
def get_BDP(df, structure_stand, t_fill):
    if structure_stand == 1:
        series  = ((df["PSRA_t"] - df["PEIA_t"] + t_fill) / 2)
                    
    elif structure_stand == 2:
        series  = ((df["PSRB_t"] - df["PEIB_t"] + t_fill) / 2)
        
    return {"BD_time": series}

def get_BDR(df, structure_stand, bdr_cfg):

    dc_up_threshold = bdr_cfg.dc_up_threshold
    dc_down_threshold = bdr_cfg.dc_down_threshold
    amount_BDR_sum = bdr_cfg.amount_BDR_sum

    bd_loc = df["BD_loc"].values
    
    if structure_stand==1:
        dc_up_peak = df["DC_up_peak_A"].values
        dc_down_peak = df["DC_down_peak_A"].values
    elif structure_stand==2:
        dc_up_peak = df["DC_up_peak_B"].values
        dc_down_peak = df["DC_down_peak_B"].values

    # BD in Load (1): PERA/PERB
    # If the first bit is one the location is Load
    bd_loc = np.where((bd_loc & 1) == 1, 1, bd_loc)

    # BD in DUT with enough DC current
    dc_condition = np.logical_or(dc_up_peak >= dc_up_threshold, 
                                 dc_down_peak >= dc_down_threshold)
    
    bd_loc = np.where(((bd_loc & 2) == 2) & (dc_condition), 2, bd_loc)

    # BD in DUT with not enough DC current
    dc_without_current_condition = (bd_loc == 2) & (~dc_condition)
    bd_loc = np.where(dc_without_current_condition, 16, bd_loc)

    # BD in DUT without BDC: PSR
    bd_loc = np.where((bd_loc & 4) == 4, 4, bd_loc)

    # BD in HyPC
    bd_loc = np.where((bd_loc & 8) == 8, 8, bd_loc)
    
    out = {}

    if structure_stand == 1:
        out["cum_BD_Load_A"] = np.cumsum(bd_loc == 1)
        out["cum_BD_DUT_withDC_A"] = np.cumsum(bd_loc == 2)
        out["cum_BD_DUT_withoutDC_A"] = np.cumsum(bd_loc == 4)
        out["cum_BD_HyPC_A"] = np.cumsum(bd_loc == 8)

        out["BDR_DUT_A"] = 0
        out["BDR_Load_A"] = 0
        out["BDR_HyPC_A"] = 0

    elif structure_stand == 2:

        out["cum_BD_Load_B"] = np.cumsum(bd_loc == 1)
        out["cum_BD_DUT_withDC_B"] = np.cumsum(bd_loc == 2)
        out["cum_BD_DUT_withoutDC_B"] = np.cumsum(bd_loc == 4)
        out["cum_BD_HyPC_B"] = np.cumsum(bd_loc == 8)

        out["BDR_DUT_B"] = 0
        out["BDR_Load_B"] = 0
        out["BDR_HyPC_B"] = 0

     # Allocate output arrays
    n = df.shape[0]

    bdr_dut = np.zeros(n, dtype=float)
    bdr_load = np.zeros(n, dtype=float)
    bdr_hypc = np.zeros(n, dtype=float)

    # Pick the right cumulative arrays (NumPy arrays)
    if structure_stand == 1:
        cum_load = np.asarray(out["cum_BD_Load_A"])
        cum_dut_with = np.asarray(out["cum_BD_DUT_withDC_A"])
        cum_dut_without = np.asarray(out["cum_BD_DUT_withoutDC_A"])
        cum_hypc = np.asarray(out["cum_BD_HyPC_A"])
        suffix = "A"

    elif structure_stand == 2:
        cum_load = np.asarray(out["cum_BD_Load_B"])
        cum_dut_with = np.asarray(out["cum_BD_DUT_withDC_B"])
        cum_dut_without = np.asarray(out["cum_BD_DUT_withoutDC_B"])
        cum_hypc = np.asarray(out["cum_BD_HyPC_B"])
        suffix = "B"

    pulse_counts = df["pulse_count"].to_numpy()
    first_pulse_count = pulse_counts[0]
    
    last_pulse_j = 0
    
    for i in range(1, n):
        current_pulse = pulse_counts[i]

        if current_pulse - amount_BDR_sum >= first_pulse_count:
            last_pulse_j = int(np.argmax(pulse_counts >= (current_pulse - amount_BDR_sum)))

        dut_i = cum_dut_with[i] + cum_dut_without[i]
        dut_j = cum_dut_with[last_pulse_j] + cum_dut_without[last_pulse_j]

        bdr_dut[i]  = (dut_i - dut_j) / amount_BDR_sum
        bdr_load[i] = (cum_load[i] - cum_load[last_pulse_j]) / amount_BDR_sum
        bdr_hypc[i] = (cum_hypc[i] - cum_hypc[last_pulse_j]) / amount_BDR_sum
        
         # Return as out-dict entries
    if suffix == "A":
        out.update({
            "BDR_DUT_A": bdr_dut,
            "BDR_Load_A": bdr_load,
            "BDR_HyPC_A": bdr_hypc,
        })
    else:
        out.update({
            "BDR_DUT_B": bdr_dut,
            "BDR_Load_B": bdr_load,
            "BDR_HyPC_B": bdr_hypc,
        })
    return out
