import pandas as pd
import numpy as np
import scipy.stats as kde
from pathlib import Path
from dataclasses import dataclass

from .io.tdms_trend import read_trend_df
from .io.tdms_event import read_event_df
from .storage.hdf import save_hdf, load_hdf

from xbox_proc.post_processing.post_processing import (
    unit_scaling,
    add_lost_power,
    add_gradient,
    add_scaled_gradient,
    get_BDP,
    get_BDR
)

class DateRange:
    def __init__(self, start, end):
        self.start = start   # YYYYMMDD
        self.end = end       # YYYYMMDD

# To configure the post processing BDR calculation
@dataclass
class BDRConfig:
    dc_up_threshold: float = 140
    dc_down_threshold: float = 100
    amount_BDR_sum: float = 1e6

class Structure:
    """
    Holds configuration and orchestrates:
    - TDMS reading
    - feature extraction
    - saving/loading
    - plotting
    """

    def __init__(self, name, data_dir, xbox, stand, 
                 P_ref, G_ref, t_fill, BDR_ref, PL_ref,
                 trend_keys=[], bdr_cfg=None):
        
        self.name = name
        self.data_dir = Path(data_dir)
        self.xbox = xbox
        self.stand = stand
        self.P_ref = P_ref
        self.G_ref = G_ref
        self.t_fill = t_fill

        self.BDR_ref = BDR_ref
        self.PL_ref = PL_ref

        self.trend_keys = trend_keys
        self.bdr_cfg = bdr_cfg or BDRConfig()

        self.df = None   # active dataframe for plotting

    # ---------------- Extraction ----------------

    def extract_trends(self, date_range, out_h5=None, key="trend_data", mode="a"):
        df = read_trend_df(
            self.data_dir,
            date_range.start,
            date_range.end,
            self.trend_keys
        )

        if out_h5 is not None:
            save_hdf(df, out_h5, key, mode)

        self.df = df
        return df

    def extract_events(self, date_range, out_h5=None, key="event_data", mode="a"):
        df = read_event_df(
            self.data_dir,
            date_range.start,
            date_range.end,
            self.stand,
            self.xbox
        )

        if out_h5 is not None:
            save_hdf(df, out_h5, key, mode)

        self.df = df
        return df

    # ---------------- Storage ----------------

    def load(self, path, key=None):
        self.df = load_hdf(path, key)
        return self.df

    # ---------------- Utilities ----------------

    def require_df(self):
        if self.df is None:
            raise RuntimeError("No dataframe loaded. Use extract_* or load() first.")
        return self.df

    # ---------------- Post Processing ----------------

    def post_process(self):

        df = self.require_df()

        # Changing the df in place
        df = unit_scaling(df, self.stand)

        # Adding info to the dataframe
        lp_cols = add_lost_power(df, self.stand)
        for col_name, series in lp_cols.items():
            df[col_name] = series

        bdp_cols = get_BDP(df, self.stand, self.t_fill)
        for col_name, series in bdp_cols.items():
            df[col_name] = series

        bdr_cols = get_BDR(df, self.stand, self.bdr_cfg)
        for col_name, series in bdr_cols.items():
            df[col_name] = series

        g_col = add_gradient(df, self.P_ref, self.G_ref, self.stand)
        for col_name, series in g_col.items():
            df[col_name] = series

        sg_col = add_scaled_gradient(df, self.BDR_ref, self.PL_ref, self.stand)
        for col_name, series in sg_col.items():
            df[col_name] = series

        return self.df

    # ---------------- BD PDF Processing ----------------

    def gaussian_kde_calc(self, 
                          kernel_factor=20., 
                          xbin=3000, 
                          ybin=500, 
                          y_min=0, 
                          y_max=6e-8*1e9):
            
        x_min = np.amin(self.df["pulse_count"])
        x_max = np.amax(self.df["pulse_count"])

        # y_min=0 #-5e-8*1e9 #ns
        # y_max=6e-8*1e9 #ns

        xi, yi = np.mgrid[x_min:x_max:xbin*1j,y_min:y_max:ybin*1j]
        positions = np.vstack([xi.ravel(), yi.ravel()])

        values = np.vstack([self.df["pulse_count"],self.df["BD_time"]*1e9 ])#ns
        kernel = kde.gaussian_kde(values, bw_method='silverman') #kernel density estimate, gaussian kernel
        kernel = kde.gaussian_kde(values, bw_method=kernel.factor/kernel_factor)
        zi = np.reshape(kernel(positions).T,xi.shape)

        return [xi, yi, zi, positions]

