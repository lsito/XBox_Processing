import numpy as np
import pandas as pd

from tqdm import tqdm

from datetime import datetime
from nptdms import TdmsFile

from .tdms_scan import list_tdms_files, filter_by_date

def read_trend_df(data_dir, startdate, enddate, keys):
    dfs = []

    files = list_tdms_files(data_dir, "Trend")
    files = filter_by_date(files, startdate, enddate)

    for filename, _ in files:
        path = data_dir / filename

        with TdmsFile.open(path) as tdms:

            for group in tqdm(
                tdms.groups(),
                desc=f"{filename}",
                leave=False
            ):
                # Skip groups without Timestamp
                if "Timestamp" not in [ch.name for ch in group.channels()]:
                    continue

                ts = group["Timestamp"][:]
                n = len(ts)

                if n == 0:
                    continue

                # Start with Timestamp
                group_data = {
                    "Timestamp": ts
                }

                # Channels available in this group
                available_channels = {
                    ch.name: ch for ch in group.channels()
                }

                # Add every requested key
                for key in keys:

                    if key not in available_channels:
                        # Channel missing in this group:
                        # preserve alignment with NaNs
                        group_data[key] = np.full(n, np.nan)
                        continue

                    values = available_channels[key][:]

                    if len(values) != n:
                        raise ValueError(
                            f"Length mismatch in file '{filename}', "
                            f"group '{group.name}', channel '{key}': "
                            f"{len(values)} values vs {n} timestamps."
                        )

                    group_data[key] = values

                dfs.append(pd.DataFrame(group_data))

    if not dfs:
        return pd.DataFrame(columns=["Timestamp", *keys])

    df = pd.concat(dfs, ignore_index=True)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        unit="s",
        origin=datetime(1904, 1, 1)
    )

    return (
        df
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )


""" Older version, here for reference. Kept for backward compatibility.
def read_trend_df(data_dir, startdate, enddate, keys):
    columns = {}

    files = list_tdms_files(data_dir, "Trend")
    files = filter_by_date(files, startdate, enddate)

    for filename, _ in files:
        path = data_dir / filename

        with TdmsFile.open(path) as tdms:
            
            for group in tqdm(tdms.groups(), desc=f"{filename}", leave=False):
            # for group in tdms.groups():
                ts = group["Timestamp"][:]

                if "Timestamp" not in columns:
                    columns["Timestamp"] = []
                columns["Timestamp"].append(ts)

                for ch in group.channels():
                    name = ch.name
                    if name == "Timestamp":
                        continue
                    if name in keys:
                        if name not in columns:
                            columns[name] = []
                        columns[name].append(ch[:])

    if not columns:
        return pd.DataFrame()

    data = {}
    for k, v in columns.items():
        data[k] = np.concatenate(v)

    df = pd.DataFrame(data)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s", origin=datetime(1904, 1, 1))
    return df.sort_values("Timestamp").reset_index(drop=True)
"""