import numpy as np
from .scaling import get_scaled_channel_data


def get_PKI_info(channel):
    try:
        data = get_scaled_channel_data(channel)
    except:
        data = np.array([0.0, 0.0])

    try:
        dx = channel.properties["wf_increment"]
    except:
        return [np.nan, np.nan, np.nan, np.nan, np.nan]

    total = np.trapz(data, dx=dx)
    peak = np.max(data) if data.size else 0

    start = np.argmax(data > peak * 0.5) * dx if data.size else 0

    flat = data[data > peak * 0.5]
    L = len(flat)

    mid = flat[int(np.ceil(L / 4)) : int(np.floor(3 * L / 4))]
    mean = np.mean(mid) if mid.size else np.nan

    return [total, L * dx, peak, mean, start]


def get_PLR_info(channel):
    try:
        data = get_scaled_channel_data(channel)
    except:
        data = np.array([0.0, 0.0])

    dx = channel.properties["wf_increment"]
    return [np.trapz(data, dx=dx)]


def get_signal_start(data, PKI_start, mid_pulse, threshold):
    lo = max(PKI_start - 50, 0)
    hi = PKI_start + mid_pulse + 1
    area = data[lo:hi]
    if area.size == 0:
        return 0
    peak = np.max(area)
    return np.argmax(data > peak * threshold)


def get_BD_pos(prev, curr, threshold, s0=0, s1=0):
    if s0 != s1:
        if s0 > s1:
            shift = s0 - s1
            prev = prev[:-shift]
            curr = curr[shift:]
        else:
            shift = s1 - s0
            curr = curr[:-shift]
            prev = prev[shift:]
    else:
        shift = 0

    sub = np.abs(prev - curr)
    peak = np.max(sub)
    pos = np.argmax(sub > peak * threshold)
    return pos + shift


def get_signal_I_info(channel, prev, log_type, PKI_start, compression, mid_pulse,
                      threshold_BD=0.5, threshold_start=0.8):

    try:
        data = get_scaled_channel_data(channel)
    except:
        data = np.array([0.0, 0.0])

    try:    
        dx = channel.properties["wf_increment"]
    except:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    
    total = np.trapz(data, dx=dx)

    PKI_idx = int(PKI_start / dx)
    mid_idx = int(mid_pulse / dx)

    if compression:
        peak_area = data[PKI_idx + mid_idx : PKI_idx + 3 * mid_idx]
    else:
        peak_area = data[max(PKI_idx - 50, 0) : PKI_idx + 2 * mid_idx + 50]

    peak = np.max(peak_area) if peak_area.size else 0
    flat = peak_area[peak_area > peak * threshold_start]
    L = len(flat)

    mid = flat[int(np.ceil(L / 4)) : int(np.floor(3 * L / 4))]
    mean = np.mean(mid) if mid.size else np.nan

    start = get_signal_start(data, PKI_idx, mid_idx, threshold_start)

    if log_type == 2:
        prev_data = get_scaled_channel_data(prev)
        prev_start = get_signal_start(prev_data, PKI_idx, mid_idx, threshold_start)
        BD = get_BD_pos(prev_data, data, threshold_BD, start, prev_start)
    else:
        BD = 0

    return [total, peak, L * dx, mean, start * dx, BD * dx]


def get_signal_R_info(channel, prev, log_type, PKI_start, mid_pulse, threshold_BD=0.5):

    try:
        data = get_scaled_channel_data(channel)
    except:
        data = np.array([0.0, 0.0])

    try:
        dx = channel.properties["wf_increment"]
    except:
        return [np.nan, np.nan, np.nan]
    
    total = np.trapz(data, dx=dx)

    PKI_idx = int(PKI_start / dx)
    pre = data[:max(PKI_idx - 20, 0)]
    offset = np.mean(pre) if pre.size else np.nan

    total_offset = np.trapz(data - offset, dx=dx) if np.isfinite(offset) else np.nan

    if log_type == 2:
        BD = get_BD_pos(get_scaled_channel_data(prev), data, threshold_BD)
    else:
        BD = 0

    return [total, total_offset, BD * dx]


def get_DC_data(group):
    try:
        up = get_scaled_channel_data(group["DC_UP"])
        down = get_scaled_channel_data(group["DC_DOWN"])
    except:
        up = np.array([0.0, 0.0])
        down = np.array([0.0, 0.0])

    dxu = group["DC_UP"].properties["wf_increment"]
    dxd = group["DC_DOWN"].properties["wf_increment"]

    return [
        -np.trapz(up, dx=dxu),
        -np.min(up),
        -np.trapz(down, dx=dxd),
        -np.min(down),
    ]
