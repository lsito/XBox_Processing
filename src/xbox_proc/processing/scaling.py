import numpy as np


def get_scaled_channel_data(channel):
    scale_type = channel.properties.get("Scale_Type")
    raw_data = channel[:]

    if scale_type == "Polynomial":
        if "Scale_Coeff_c2" in channel.properties:
            coeffs = [
                float(channel.properties["Scale_Coeff_c2"]),
                float(channel.properties["Scale_Coeff_c1"]),
                float(channel.properties["Scale_Coeff_c0"]),
            ]
        else:
            coeffs = [
                float(channel.properties["Scale_Coeff_c1"]),
                float(channel.properties["Scale_Coeff_c0"]),
            ]
        return np.polyval(coeffs, raw_data.astype(float))

    elif scale_type == "Logarithmic":
        A = float(channel.properties["Log_Coeff_A"])
        b = float(channel.properties["Log_Coeff_b"])
        C = float(channel.properties["Log_Coeff_C"])
        return A * np.exp(b * raw_data) + C

    elif scale_type == "None" or scale_type is None:
        return raw_data

    else:
        return np.full_like(raw_data, np.nan, dtype=float)
