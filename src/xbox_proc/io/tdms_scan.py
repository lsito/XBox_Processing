import os

def extract_date(filename):
    stem = filename.replace(".tdms", "")
    try:
        return int(stem.split("_")[-1])
    except:
        return None


def list_tdms_files(data_dir, token):
    files = []
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith(".tdms"):
            continue
        if token not in fn:
            continue
        date = extract_date(fn)
        if date is not None:
            files.append((fn, date))
    return files


def filter_by_date(files, start, end):
    return [(fn, d) for fn, d in files if start <= d < end]