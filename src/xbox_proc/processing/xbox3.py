import numpy as np
from .signals import (
    get_PKI_info,
    get_PLR_info,
    get_signal_I_info,
    get_signal_R_info,
    get_DC_data,
)


def group_data_xbox3(group, prev_group, structure_stand):
    data = {
        "pulse_count": group.properties["Pulse Count"],
        "log_type": group.properties["Log Type"],
        "timestamp": group.properties["Timestamp"],
    }

    channels_PK = ["PKIA_amp", "PKIB_amp", "PKRA", "PKRB"]
    pk_fields = ["total_power", "length", "peak", "mean", "start"]

    for ch in channels_PK:
        vals = get_PKI_info(group[ch])
        for i, name in enumerate(pk_fields):
            data[ch + "_" + name] = vals[i]

    for ch in ["PLRA", "PLRB"]:
        data[ch + "_total_power"] = get_PLR_info(group[ch])[0]

    compression = (data["PKIA_amp_length"] > 1e-6) or (data["PKIB_amp_length"] > 1e-6)
    mid_pulse = max(data["PKIA_amp_length"], data["PKIB_amp_length"]) / 2
    PKI_start = max(data["PKIA_amp_start"], data["PKIB_amp_start"])
    log_type = data["log_type"]

    suffix = "A" if structure_stand == 1 else "B"

    psi = get_signal_I_info(group["PSI_amp"], prev_group["PSI_amp"],
                            log_type, PKI_start, compression, mid_pulse)
    for i, k in enumerate(["total_power", "peak", "peak_length", "mean_flat", "start", "t"]):
        data["PSI" + suffix + "_" + k] = psi[i]

    psr = get_signal_R_info(group["PSR_amp"], prev_group["PSR_amp"],
                            log_type, PKI_start, mid_pulse)
    for i, k in enumerate(["total_power", "total_power_including_offset", "t"]):
        data["PSR" + suffix + "_" + k] = psr[i]

    pei = get_signal_I_info(group["PEI_amp"], prev_group["PEI_amp"],
                            log_type, PKI_start, compression, mid_pulse)
    for i, k in enumerate(["total_power", "peak", "peak_length", "mean_flat", "start", "t"]):
        data["PEI" + suffix + "_" + k] = pei[i]

    dc = get_DC_data(group)
    for i, k in enumerate(["DC_up_total", "DC_up_peak", "DC_down_total", "DC_down_peak"]):
        data[k + "_" + suffix] = dc[i]

    BD_loc = 0
    data["BD_struct"] = 0

    if log_type == 2:
        data["BD_struct"] = structure_stand
        if group.properties.get("BD_PERA") or group.properties.get("BD_PERB"):
            BD_loc |= 1
        if group.properties.get("BD_DC_UP") or group.properties.get("BD_DC_DOWN"):
            BD_loc |= 2
        if group.properties.get("BD_PSR_amp") or group.properties.get("BD_PSR_ph"):
            BD_loc |= 4
        if group.properties.get("BD_PLRA") or group.properties.get("BD_PLRB"):
            BD_loc |= 8
        if group.properties.get("BD_PKRA") or group.properties.get("BD_PKRB"):
            BD_loc |= 8

    data["BD_loc"] = BD_loc
    return data
