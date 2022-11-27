
import datetime
import json
import os
import re
import string

from bs4 import BeautifulSoup
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from shapely import affinity, geometry
from tqdm import tqdm

from tif_files import TifFile

DATETIME_FORMAT = "%Y_%m_%d__%H_%M_%S"
PUNCTUATION = {p: "_" for p in string.punctuation}
DATE_FORMAT = "%Y_%m_%d"

DIR, FILENAME = os.path.split(__file__)
DATA_DIR = os.path.join(os.path.dirname(DIR), "data")
MASK_DIR = os.path.join(DATA_DIR, "masks")
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)
BASIN_DIR = os.path.join(DATA_DIR, "usgs_basins")
if not os.path.exists(BASIN_DIR):
    os.mkdir(BASIN_DIR)
SAMPLE_DIR = os.path.join(DATA_DIR, "sample_images")
if not os.path.exists(SAMPLE_DIR):
    os.mkdir(SAMPLE_DIR)


def list_streamgage_files(directory: str):
    """List all files in directory with pattern `{streamgage}_xyx_{datetime}."""
    files = [f for f in os.listdir(directory) if "__" in f]
    files = list(filter(lambda s: len(re.findall("\d{8}_", s)) > 0, files))
    gage_ids = [re.findall("\d{8}_", f)[0][:-1] for f in files]
    timestamps = [f"2022" + f.split("2022")[-1] for f in files]
    timestamps = [ts.split(".")[0] for ts in timestamps]
    timestamps = [datetime.datetime.strptime(ts, DATETIME_FORMAT) for ts in timestamps]
    df = pd.DataFrame({"file": files, "gage_id": gage_ids, "timestamp": timestamps})
    df = df.sort_values(by=["gage_id", "timestamp"], ascending=False)
    df["latest"] = df["gage_id"] != df["gage_id"].shift(1)
    return df


def get_latest_streamgage_file(directory: str, streamgage: str):
    files_df = list_streamgage_files(directory)
    files_df = files_df[(files_df["gage_id"] == str(streamgage)) & (files_df["latest"] == True)]
    assert len(files_df) == 1, f"No files found for gage: {streamgage}"
    return files_df.iloc[0]["file"]


def remove_punctuation(s: str):
    return s.translate(str.maketrans(PUNCTUATION))


def get_polygon(left: float, bottom: float, right: float, top: float,
                buffer_percent: float = 0.05):
    coords = [
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
        [left, top],
    ]
    shape = geometry.Polygon(coords)
    if buffer_percent > 0:
        shape = geometry.Polygon(affinity.scale(
            shape, xfact=(1. + buffer_percent), yfact=(1. + buffer_percent)
        ).exterior.coords)
    return shape


def fmap(*lat_lon: tuple, labels: list = None, zoom_start: int = 14):
    """Folium map of lat-lon coordinates.

    Args:
        lat_lon: tuple of latitude, longitude.
        labels: optional marker labels (length should match lat_lon if passed).
        zoom_start: passed to folium.Map
    """
    first_lat, first_lon = lat_lon[0]
    my_map = folium.Map(location=[first_lat, first_lon], zoom_start=zoom_start)
    for i, (lat, lon) in enumerate(lat_lon):
        if labels is None:
            folium.Marker(location=[lat, lon], radius=10).add_to(my_map)
        else:
            folium.Marker(location=[lat, lon], radius=10, popup=labels[i]).add_to(my_map)
    return my_map


def update_json(fp: str, key: str, value: object):
    if not os.path.exists(fp):
        data = dict()
        with open(fp, "w") as f:
            json.dump(data, f)

    with open(fp, "r") as f:
        data = json.load(f)
        data[key] = value

    with open(fp, "w") as f:
        json.dump(data, f)

    print(f"Updated JSON file {fp}:\n  key = {str(key)}\n  value = {str(value)}")


def get_usgs_site_info(gage: str):
    """Get the full site location information from the USGS website."""
    url = f"https://waterdata.usgs.gov/nwis/wys_rpt/?site_no={gage}&agency_cd=USGS"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    full_title = soup.find_all("div", {"id": "station_full_nm"})[0].contents[0]
    location = soup.find_all("div", {"id": "location"})[0].contents[1]
    area = soup.find_all("div", {"id": "drainage_area"})[0].contents[1]
    return {"full_title": full_title, "location": f"LOCATION{location}", "area": f"DRAINAGE AREA{area}"}


def sat_img_filelist_df(filelist: list):
    """Construct a DataFrame from a list of satellite image files with this
    specific file format:

        '/subdirs/{crs}__{scale}__{satellite}__{band}__{date}.tif'
    """
    df = pd.DataFrame({"filename": filelist})
    df["filename_ext"] = df["filename"].map(lambda s: s.split(".")[-1])
    df["filename_prefix"] = df["filename"].map(lambda s: s.split("/")[-1])
    df["filename_prefix"] = df["filename_prefix"].map(lambda s: s.split(".")[0])
    df["subdir"] = df["filename"].map(lambda s: "/".join(s.split("/")[:-1]))
    filename_parts = ["crs", "scale", "satellite", "band"]
    for i, part in enumerate(filename_parts):  # NOQA
        df[part] = df["filename_prefix"].map(lambda s: s.split("__")[i])
    df["scale"] = df["scale"].str.replace("_", ".").astype(float)
    dates = df["filename_prefix"].map(lambda s: s.split("__")[-1])
    # Remove version numbers and hours if present:
    df["date_str"] = dates.map(lambda s: s[:10])
    df["date"] = pd.to_datetime(df["date_str"], format=DATE_FORMAT)
    return df


def get_y_data(gage: str, from_date: str, to_date: str):
    """Query USGS website to get target variable data for a specific
    streamgage location.

    Args:
        gage: streamgage location name.
        from_date: minimum date in format YYYY_MM_DD.
        to_date: maximum date in format YYYY_MM_DD.
    """
    from_dt = datetime.datetime.strptime(from_date, DATE_FORMAT)
    to_dt = datetime.datetime.strptime(to_date, DATE_FORMAT)

    url = f"https://waterdata.usgs.gov/nwis/dv?cb_all_=on&cb_00010=on&cb_00060=on&format=rdb&site_no={gage}" \
          f"&referred_module=sw&period=&begin_date={from_dt.strftime('%Y-%m-%d')}" \
          f"&end_date={to_dt.strftime('%Y-%m-%d')}"

    r = requests.get(url)
    str_data = r.content.decode("utf-8")
    lines = str_data.split("\n")
    first_line = 0
    while not lines[first_line].startswith("agency_cd"):
        first_line += 1
    lines = lines[first_line:]
    columns = lines[0].split("\t")
    dtypes = lines[1].split("\t")
    data = lines[2:]
    split_data = [(line.split("\t")) for line in data]
    df = pd.DataFrame(split_data, columns=columns).dropna(subset=["site_no", "datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d")
    for dtype, col in zip(dtypes, columns):
        if "n" in dtype:
            df[col] = df[col].replace("", np.nan).astype(float)
            df["ft"] = df[col]
    return df


def open_y_data():
    """Load the saved CSV of outcome data."""
    fp = os.path.join(DATA_DIR, "streamgage-full.csv")
    df = pd.read_csv(fp, encoding="utf-8")
    df["date"] = pd.to_datetime(df["time"])
    df["gage"] = df["gage"].astype(str)
    return df


def expected_image_dates(from_date: datetime.datetime,
                         to_date: datetime.datetime, freq: int) -> set:
    """Get all dates of a day frequency that fall between the from/to dates.
    Assumes that satellite always starts images on Jan-1st (as MODIS does).

    Args:
        from_date: minimum inclusive start date.
        to_date: maximum inclusive end date.
        freq: frequency in days
    """
    dates = list()
    for year in range(from_date.year, to_date.year + 1, 1):
        year_dates = [datetime.datetime(year, 1, 1)]
        while True:
            next_date = year_dates[-1] + datetime.timedelta(days=freq)
            if next_date.year == year:
                year_dates.append(next_date)
            else:
                break
        dates += year_dates
    return {dt for dt in dates if (dt >= from_date) and (dt <= to_date)}


def pixel_mean_std(*img_fp):
    """Calculate mean and STD of all image pixels across multiple images.
    Automatically fills in NaN values with zero."""
    # Source: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
    total_img_sum = 0
    total_img_sum_sq = 0
    total_num_pixels = 0

    print(f"Calculating pixel mean/std from {len(img_fp)} images")
    for fp in tqdm(img_fp):
        if fp.endswith(".tif"):  # .tif file
            tif = TifFile(fp)
            np_arr = tif.as_numpy_zero_nan
        elif fp.endswith(".npy"):  # .npy file
            np_arr = open_npy_file(fp)
        else:
            raise TypeError(f"Invalid filetype: {fp}")
        img_sum = np_arr.sum()
        total_img_sum += img_sum
        img_sum_sq = (np_arr ** 2).sum()
        total_img_sum_sq += img_sum_sq
        total_num_pixels += (np_arr.shape[0] * np_arr.shape[1])

    pixel_mean = total_img_sum / total_num_pixels
    pixel_var = (total_img_sum_sq / total_num_pixels) - (pixel_mean ** 2)
    pixel_std = np.sqrt(pixel_var)

    return {"pixel_mean": float(pixel_mean), "pixel_std": float(pixel_std)}


def convert_datetime(date_obj: object, fmt: str = DATE_FORMAT):
    """Convert possible string representation to datetime.datetime."""
    if isinstance(date_obj, datetime.datetime):
        return date_obj
    elif isinstance(date_obj, str):
        return datetime.datetime.strptime(date_obj, fmt)


def open_npy_file(fp: str):
    """Open a numpy array saved as .npy file. Automatically fills in NaN values
    with zero."""
    arr = np.load(fp)
    arr[np.isnan(arr)] = 0
    return arr


def extract_filename_data(fn: str):
    """Extract gage, band, and date from a filename using regex."""
    streamgage = re.findall("\d{8}", fn)  # NOQA
    assert len(streamgage) == 1, f"No 8-digit streamgage found in filename: {fn}"
    datestr = re.findall("\d{4}_\d{2}_\d{2}", fn)  # NOQA
    date = datetime.datetime.strptime(datestr[0], DATE_FORMAT)
    assert len(datestr) == 1, f"No datestring found in filename: {fn}"
    band = False
    for b in ("swe", "et", "temperature_2m", "total_precipitation", "dem"):
        if b in fn.lower():
            assert not band, f"Multiple bands in filename: {fn}"
            band = b
    assert band, f"No band in filename: {fn}"
    return {"fn": fn, "band": band, "date": date, "streamgage": streamgage[0]}


def open_swe_file(fp: str):
    """Open a swe .npy file. Automatically fills in NaN values with zero.
    Flip images to correct lat-lon alignment."""
    arr = np.load(fp)
    arr[np.isnan(arr)] = 0
    return np.rot90(arr, k=1, axes=(0, 1))


def get_swe_mask(swe_fp: str):
    """For an input SWE image, get the mask of its intersection with a GeoJson
    file of an irregular geometry (i.e. a watershed area)."""
    # Get gage name from filename:
    fn = swe_fp.split("/")[-1]
    gage = re.findall("\d{8}", fn)  # NOQA
    assert len(gage) == 1
    gage = gage[0]

    # See if mask is already saved:
    swe_arr = open_swe_file(swe_fp)
    h, w = swe_arr.shape
    mask_name = f"{gage}__swe__h{h}_w{w}.npy"
    fp = os.path.join(MASK_DIR, mask_name)
    if os.path.exists(fp):
        return open_npy_file(fp)
    else:
        return calculate_swe_mask(swe_fp, gage)


def calculate_swe_mask(swe_fp: str, gage: str):
    """Calculate the intersection mask of an SWE image's pixels with a GeoJson
    file of an irregular geometry (i.e. a watershed area)."""
    swe_arr = open_swe_file(swe_fp)

    num_rows, num_columns = swe_arr.shape

    # Get the bounding box of the SWE numpy array:
    bboxes_fp = os.path.join(DATA_DIR, "watershed_bounding_boxes.json")
    with open(bboxes_fp, "r") as f:
        bboxes = json.load(f)
    bbox = bboxes[gage]
    min_lon, min_lat, max_lon, max_lat = bbox

    # Calculate the lon/lat size of each pixel:
    pixel_width_lon = (max_lon - min_lon) / num_columns
    pixel_height_lat = (max_lat - min_lat) / num_rows

    # Load the basin geometry:
    geojson_fp = os.path.join(BASIN_DIR, f"{gage}.geojson")
    assert os.path.exists(geojson_fp), f"USGS basin file not found: {geojson_fp}"
    gdf = gpd.read_file(geojson_fp)
    basin_geo = gdf[gdf["id"] == "globalwatershed"]["geometry"].iloc[0]

    def pixel_shape(r, c):
        left = min_lon + (pixel_width_lon * c)
        right = min_lon + (pixel_width_lon * (c+1))
        bottom = min_lat + (pixel_height_lat * r)
        top = min_lat + (pixel_height_lat * (r+1))
        return get_polygon(left, bottom, right, top, buffer_percent=0)

    # Compute the mask:
    ratio_list = list()
    for row in list(range(swe_arr.shape[0]))[::-1]:  # Iterate rows backwards so they are appended top to bottom.
        row_values = list()
        for col in range(swe_arr.shape[1]):
            pxl = pixel_shape(row, col)
            intersection = pxl.intersection(basin_geo).area
            ratio = intersection / pxl.area
            row_values.append(ratio)
        ratio_list.append(row_values)
    mask = np.array(ratio_list)

    # Save the mask:
    h, w = mask.shape
    mask_name = f"{gage}__swe__h{h}_w{w}.npy"
    fp = os.path.join(MASK_DIR, mask_name)
    np.save(fp, mask, allow_pickle=False)

    return mask


def plot_masks(gage: str):
    """Plot all masks for the givenn streamgage."""
    fig, axes = plt.subplots(4, 5, figsize=(8, 10))

    sample_files = [f for f in os.listdir(SAMPLE_DIR) if gage in f]
    mask_files = [f for f in os.listdir(MASK_DIR) if gage in f]

    flat_axes = axes.flatten()
    for i, band in enumerate(("swe", "et", "precip", "temp")):

        sample = [f for f in sample_files if band in f.lower()][0]
        sample_fp = os.path.join(SAMPLE_DIR, sample)
        if sample_fp.endswith(".tif"):
            tif = TifFile(sample_fp)
            sample_arr = tif.as_numpy
            vmax = tif.pixel_nanmax
        elif sample_fp.endswith(".npy"):
            sample_arr = open_swe_file(sample_fp)
            vmax = np.nanmax(sample_arr)
        else:
            raise TypeError(f"Unknown file type: {sample_fp}")

        mask_fn = [f for f in mask_files if band in f.lower()][0]
        mask_arr = open_npy_file(os.path.join(MASK_DIR, mask_fn))

        sample_ix = i * 5
        ax = flat_axes[sample_ix]
        ax.imshow(sample_arr, vmin=0, vmax=vmax)
        ax.set_title(f"{band.upper()}")
        ax.set_xticks([])
        ax.set_yticks([])

        mask_ix = sample_ix + 1
        ax = flat_axes[mask_ix]
        ax.imshow(mask_arr, vmin=0, vmax=1)
        ax.set_title("Ratio Mask")
        ax.set_xticks([])
        ax.set_yticks([])

        masked_ix = mask_ix + 1
        ax = flat_axes[masked_ix]
        ratio_masked = np.where(mask_arr == 0, 0, mask_arr * sample_arr)
        ax.imshow(ratio_masked, vmin=0, vmax=vmax)
        ax.set_title("Ratio Masked")
        ax.set_xticks([])
        ax.set_yticks([])

        bool_mask = np.where(mask_arr, 1, 0)
        bool_mask_ix = masked_ix + 1
        ax = flat_axes[bool_mask_ix]
        ax.imshow(bool_mask, vmin=0, vmax=1)
        ax.set_title("Bool Mask")
        ax.set_xticks([])
        ax.set_yticks([])

        bool_masked_ix = bool_mask_ix + 1
        ax = flat_axes[bool_masked_ix]
        bool_masked = np.where(bool_mask == 0, 0, sample_arr)
        ax.imshow(bool_masked, vmin=0, vmax=vmax)
        ax.set_title("Bool Masked")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Gage = {gage}")

    return fig


def tif_to_npy(fp: str):
    """Save data from a .tif file as a numpy.array with same name but .npy ext.

    Args:
        fp: path to .tif file
    """
    assert os.path.exists(fp)
    assert fp.endswith(".tif")
    tif = TifFile(fp)
    arr = tif.as_numpy
    assert isinstance(arr, np.ndarray)
    basename = os.path.basename(fp)
    new_basename = basename.replace(".tif", ".npy")
    dir_name = os.path.dirname(fp)
    new_fp = os.path.join(dir_name, new_basename)
    np.save(new_fp, arr, allow_pickle=False, fix_imports=False)
    return new_fp


def convert_training_tif_files(train_dir: str):
    """Convert all .tif files in the training directory to .npy files.

    Args:
        train_dir: path to training directory containing .tif files.

    Returns:
        list: list of new .npy filepaths.
    """
    np_files = list()
    files = os.listdir(train_dir)
    files = list(filter(lambda fn: fn.endswith(".tif"), files))
    for f in files:
        fp = os.path.join(train_dir, f)
        np_fp = tif_to_npy(fp)
        np_files.append(np_fp)
    return np_files
