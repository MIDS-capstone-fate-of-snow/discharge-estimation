
import datetime
import json
import os
import re
import string

from bs4 import BeautifulSoup
import folium
import numpy as np
import pandas as pd
import requests
from shapely import affinity, geometry

DATETIME_FORMAT = "%Y_%m_%d__%H_%M_%S"
PUNCTUATION = {p: "_" for p in string.punctuation}
DATE_FORMAT = "%Y_%m_%d"


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


def get_polygon(left: float, bottom: float, right: float, top: float, buffer_percent: float = 0.05):
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
    split_data = [(l.split("\t")) for l in data]
    df = pd.DataFrame(split_data, columns=columns).dropna(subset=["site_no", "datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d")
    for dtype, col in zip(dtypes, columns):
        if "n" in dtype:
            df[col] = df[col].replace("", np.nan).astype(float)
            df["ft"] = df[col]
    return df
