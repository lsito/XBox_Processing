import pandas as pd
from tqdm import tqdm

from nptdms import TdmsFile

from .tdms_scan import list_tdms_files, filter_by_date
from ..processing.xbox3 import group_data_xbox3
from ..processing.xbox2 import group_data_xbox2_polarix



def read_event_df(data_dir, startdate, enddate, stand, xbox):
    if stand == 2:
        token = "EventDataB"
    elif stand == 1:
        token = "EventDataA"
    else:
        token = "EventData" # For XBOX2, which doesn't have separate stands

    files = list_tdms_files(data_dir, token)
    files = filter_by_date(files, startdate, enddate)

    columns = {}   # name -> list of scalars

    for filename, _ in files:
        path = data_dir / filename

        with TdmsFile.open(path) as tdms:
            prev = None
            for group in tqdm(tdms.groups(), desc=f"{filename}", leave=False):
                try:
                    #for group in tdms.groups():
                    if prev is None:
                        prev = group
                        continue

                    if xbox == 3:
                        row = group_data_xbox3(group, prev, stand)
                    else:
                        # raise RuntimeError("Unsupported Xbox version")
                        row = group_data_xbox2_polarix(group, prev)

                    # accumulate into columns
                    for k, v in row.items():
                        if k not in columns:
                            columns[k] = []
                        columns[k].append(v)

                    prev = group
                except:
                    print(f"Error processing group {group.name} in file {filename}. Skipping this group.")
                    continue
    # build dataframe
    if not columns:
        return pd.DataFrame()

    return pd.DataFrame(columns)
