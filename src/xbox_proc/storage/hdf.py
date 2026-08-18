import pandas as pd
from pathlib import Path

# Deprecated version, we are not checking if part of the lines are already present
def save_hdf(df, path, key, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(path, key=key, mode=mode)


def load_hdf(path, key=None):
    if key is None:
        return pd.read_hdf(path)
    return pd.read_hdf(path, key)

from pathlib import Path
import pandas as pd


def save_hdf(df, path, key, mode="a", subset=None):
    """
    Save a DataFrame to HDF5.

    If mode == "w":
        overwrite the file/key.

    If mode == "a":
        append only rows that are not already present under the same key.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    path : str or Path
        HDF5 file path.
    key : str
        HDF5 key, e.g. "event_data" or "trend_data".
    mode : str
        "w" to overwrite, "a" to append missing rows.
    subset : list[str] or None
        Columns used to detect duplicates.
        If None, the full row is compared.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    key_norm = "/" + key.strip("/")

    if mode == "w" or not path.exists():
        df.to_hdf(
            path,
            key=key,
            mode="w",
            format="table",
            data_columns=True,
        )
        return

    if mode != "a":
        raise ValueError(f"Unsupported HDF mode: {mode!r}. Use 'a' or 'w'.")

    with pd.HDFStore(path, mode="a") as store:
        if key_norm in store.keys():
            old_df = store[key]

            combined = pd.concat([old_df, df], axis=0)

            combined = combined.drop_duplicates(
                subset=subset,
                keep="first",
            )

            store.put(
                key,
                combined,
                format="table",
                data_columns=True,
            )
        else:
            store.put(
                key,
                df,
                format="table",
                data_columns=True,
            )