import numpy as np
from .signals import (
    get_PKI_info,
    get_PLR_info,
    get_signal_I_info,
    get_signal_R_info,
    get_DC_data,
)

from .scaling import *

# Special signal configuration for Polarix Structure Conditioning
def group_data_xbox2_polarix(group, prev_group):
    try:
        data = {
            "pulse_count": group.properties["Pulse Count"],
            "log_type": group.properties["Log Type"],
            "timestamp": group.properties["Timestamp"],
        }
    except:
        print("Missing properties in group")
        data = {
            "pulse_count": np.nan,
            "log_type": np.nan, 
            "timestamp": np.nan,
        }

    channels_PK = ["PKI Amplitude"]
    channels_PK_nn = ["PKI_amp"]
    pk_fields = ["total_power", "length", "peak", "mean", "start"]

    for ch, nn in zip(channels_PK, channels_PK_nn):
        vals = get_PKI_info(group[ch])
        for i, name in enumerate(pk_fields):
            data[nn + "_" + name] = vals[i]

    # for ch in ["PLR"]:
        # data[ch + "_total_power"] = get_PLR_info(group[ch])[0]

    compression = data["PKI_amp_length"] > 1e-6
    mid_pulse = data["PKI_amp_length"] / 2
    PKI_start = data["PKI_amp_start"]
    log_type = data["log_type"]

    
    psi = get_signal_I_info(group["PSI Amplitude"], prev_group["PSI Amplitude"],
                                log_type, PKI_start, compression, mid_pulse)
    for i, k in enumerate(["total_power", "peak", "peak_length", "mean_flat", "start", "t"]):
        data["PSI" + "_" + k] = psi[i]

    psr = get_signal_R_info(group["PSR Amplitude"], prev_group["PSR Amplitude"],
                            log_type, PKI_start, mid_pulse)
    for i, k in enumerate(["total_power", "total_power_including_offset", "t"]):
        data["PSR" + "_" + k] = psr[i]

    pei1 = get_signal_I_info(group["PEI1 Amplitude"], prev_group["PEI1 Amplitude"],
                            log_type, PKI_start, compression, mid_pulse)
    for i, k in enumerate(["total_power", "peak", "peak_length", "mean_flat", "start", "t"]):
        data["PEI1" + "_" + k] = pei1[i]

    pei2 = get_signal_I_info(group["PEI2 Amplitude"], prev_group["PEI2 Amplitude"],
                            log_type, PKI_start, compression, mid_pulse)
    for i, k in enumerate(["total_power", "peak", "peak_length", "mean_flat", "start", "t"]):
        data["PEI2" + "_" + k] = pei2[i]


    def get_DC_data_polarix(group):
        try:
            up = get_scaled_channel_data(group["DC Up"])
            down = get_scaled_channel_data(group["DC Down"])
        except:
            up = np.array([0.0, 0.0])
            down = np.array([0.0, 0.0])
            
        try:
            dxu = group["DC Up"].properties["wf_increment"]
            dxd = group["DC Down"].properties["wf_increment"]
        except:
            return [
             np.nan, np.nan, np.nan, np.nan
        ]

        return [
            -np.trapz(up, dx=dxu),
            -np.min(up) if up.size else np.nan,
            -np.trapz(down, dx=dxd),
            -np.min(down) if down.size else np.nan,
        ]

    dc = get_DC_data_polarix(group)
    for i, k in enumerate(["DC_up_total", "DC_up_peak", "DC_down_total", "DC_down_peak"]):
        data[k] = dc[i]

    BD_loc = 0
    data["BD_struct"] = 0

    if log_type == 2:
        if group.properties.get("BD_PER log"):
            BD_loc |= 1
        if group.properties.get("BD_DC Up") or group.properties.get("BD_DC Down"):
            BD_loc |= 2
        if group.properties.get("BD_PSR"):
            BD_loc |= 4

    data["BD_loc"] = BD_loc

    return data

