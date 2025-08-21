import xarray as xr
import fsspec
from itertools import islice
import time
import matplotlib.pyplot as plt

def preprocess_timeseries_df(df, time_col='time'):
    df = df.copy()
    for col in df.columns:
        if col != time_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values(by=time_col)
    return df

def resample_to_daily(df, value_col, time_col='time',time_is_index=False, new_col_name=None):
    df = df.copy()
    if not time_is_index:
        df = df.set_index(time_col).resample('D')[value_col].sum().reset_index()
    if new_col_name:
        df.columns = ['time', new_col_name]
    return df


def plot_monthly_daily_totals(df, variable, time_col='time'):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    daily_data = df.set_index(time_col).resample('D')[variable].sum().reset_index()
    daily_data['month'] = daily_data[time_col].dt.to_period('M')

    for month, group in daily_data.groupby('month'):
        plt.figure(figsize=(12, 4))
        plt.plot(group[time_col], group[variable], marker='o', linestyle='-', color='blue')
        plt.title(f' {variable.capitalize()}  daily totals  - {month.strftime("%B %Y")}')
        plt.xlabel('Date')
        plt.ylabel(variable.capitalize())
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

def _batch_iterable(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch

def _open_with_fallback(f_obj):
    """Open a file object, trying NetCDF4 first, then NetCDF3 (scipy)."""
    try:
        ds = xr.open_dataset(f_obj, engine="h5netcdf", decode_times=True, chunks={"time": 50})
    except ValueError:
        ds = xr.open_dataset(f_obj, engine="scipy", decode_times=True, chunks={"time": 50})
    return ds

def load_imerg_from_HF(subdir="SouthAfrica_Limpopo", token=None, batch_size=20):
    repo_id = "musamthembu84/imerg"
    pattern = f"hf://datasets/{repo_id}/{subdir}/*.nc4"
    files = fsspec.open_files(pattern, mode="rb", token=token)

    datasets = []

    for batch_files in _batch_iterable(files, batch_size):
        ds_list = []

        for f in batch_files:
            f_obj = f.open()
            time.sleep(0.1)  # gentle delay
            ds = _open_with_fallback(f_obj)
            ds_list.append(ds["precipitation"])

        ds_batch = xr.concat(ds_list, dim="time")
        datasets.append(ds_batch)

    ds = xr.concat(datasets, dim="time")
    return ds