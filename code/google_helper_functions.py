"""Helper functions for working with Google Earth Engine and Google Drive
service accounts."""

import datetime
import os
from typing import Tuple
import warnings

import ee
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import pydrive2
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from shapely import geometry

from tqdm import tqdm

from utils import get_polygon, remove_punctuation

DATETIME_FORMAT = "%Y_%m_%d__%H_%M_%S"


def connect_to_service_account_gdrive(key_fp: str):
    """Establish a connection to the service account's G:Drive for accessing
    saved GEE images.

    Args:
        key_fp: path to service account's JSON key file.

    Returns:
        pydrive2.drive.GoogleDrive
    """
    gauth = GoogleAuth()
    scopes = ['https://www.googleapis.com/auth/drive']
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(key_fp, scopes=scopes)
    drive = GoogleDrive(gauth)
    return drive


def authenticate_gee(service_account: str, key_fp: str):
    """Authenticate GEE using a service account:
    See: https://developers.google.com/earth-engine/guides/service_account

    Returns:
        google.oauth2.service_account.Credentials
    """
    credentials = ee.ServiceAccountCredentials(service_account, key_fp)
    ee.Initialize(credentials)
    return credentials


def download_files_from_gdrive(drive: pydrive2.drive.GoogleDrive, filename: str,
                               save_to: str, download_all: bool = True):
    """Download files from G:Drive.

    Args:
        drive: authenticated connected GoogleDrive instance.
        filename: filename to match on.
        save_to: local directory to download to.
        download_all: if True and mutliple matching files are found, download
            them all. If False, only downloads first matching file found.
    """
    file_list = drive.ListFile({"q": f"title contains '{filename}'"}).GetList()
    n_files = len(file_list)
    assert n_files > 0, f"No matching files found for string: {filename}"
    if n_files > 1 and download_all:
        warnings.warn(f"Downloading {n_files:,} files matching string: {filename}")
    elif n_files > 1:
        warnings.warn(f"Downloading first matching file of {n_files:,} found matching string: {filename}")
        file_list = file_list[:1]
    downloaded = list()
    for f in tqdm(file_list):
        target = os.path.join(save_to, f["title"])
        f.GetContentFile(target)
        downloaded.append(target)
    return downloaded


def clear_up_service_account_gdrive(drive):
    files = drive.ListFile({"q": f"trashed=false"}).GetList()
    files = [f for f in files if f["title"].endswith(".tif") and "__" in f["title"] and "elv" in f["title"]]
    gage_ids = [f["title"].split("_")[0] for f in files]
    timestamps = [f"2022" + f["title"].split("2022")[-1][:-4] for f in files]
    timestamps = [datetime.datetime.strptime(ts, DATETIME_FORMAT) for ts in timestamps]
    df = pd.DataFrame({"file": files, "gage_id": gage_ids, "timestamp": timestamps})
    df = df.sort_values(by=["gage_id", "timestamp"], ascending=False)
    df["keep"] = df["gage_id"] != df["gage_id"].shift(-1)
    delete = df[~df["keep"]]
    for drive_file in delete["file"]:
        drive_file.Delete()
    return delete


DATE_FORMAT = "%Y_%m_%d"


def get_ee_img_collection(sat_name: str, polygon: geometry.Polygon,
                          date_from: str, date_to: str):
    """Get ee.ImageCollection for a satellite.

    Args:
        sat_name: name of satellite in GEE datasets catalog.
        polygon: geographical region to get data for.
        date_from: start date for data.
        date_to: end date for data.
    """
    raise DeprecationWarning()
    gee_polygon = ee.Geometry.Polygon(list(polygon.boundary.coords))
    date_from = datetime.datetime.strptime(date_from, DATE_FORMAT)
    date_to = datetime.datetime.strptime(date_to, DATE_FORMAT)
    return ee.ImageCollection(sat_name).filterDate(date_from, date_to).filterBounds(gee_polygon)


def export_image_collection_to_gdrive(bounding_box: Tuple[float], sat_name: str,
                                      band: str, date_from: str, date_to: str,
                                      buffer_percent: float = 0.05,
                                      crs: str = "epsg:4326",
                                      scale: float = None):
    raise DeprecationWarning()

    # Define the region of interest:
    left, bottom, right, top = bounding_box  # NOQA
    polygon = get_polygon(left, bottom, right, top, buffer_percent=buffer_percent)

    # Get the GEE image collection:
    ic = get_ee_img_collection(sat_name=sat_name, polygon=polygon, date_from=date_from, date_to=date_to)

    # Number of images to download from the image collection:
    num_img = ic.size().getInfo()
    print(f"Satellite {sat_name}: {num_img:,.0f} images")

    # Convert to a list of images:
    img_list = ic.toList(num_img)

    # Export the images one by one:
    all_tasks, filenames = list(), list()
    for i in range(num_img):

        # Select the image:
        img = ee.Image(img_list.get(i))

        # Date of the image:
        img_date = img.date().getInfo()  # NOQA
        hdate = datetime.datetime.utcfromtimestamp(img_date["value"] / 1000).strftime(DATE_FORMAT)
        print(f"> Task {str(i+1).zfill(7)} - {hdate}")

        # Select a band:
        img_band = img.select(band)

        # Get the original scale if not rescaling:
        if scale is None:
            scale = img_band.projection().nominalScale().getInfo()

        # Get the GEE polygon shape
        gee_polygon = ee.Geometry.Polygon(list(polygon.boundary.coords))

        # Reproject:
        reprojection = img_band.reproject(crs=crs, scale=scale)

        # Export to G:Drive:
        sat_name_c = remove_punctuation(sat_name)
        crs_c = remove_punctuation(crs)
        band_c = remove_punctuation(band)
        hdate_c = remove_punctuation(hdate)
        scale_c = remove_punctuation(f"{scale:.2f}")
        filename = f"{crs_c}__{scale_c}__{sat_name_c}__{band_c}__{hdate_c}"
        task = ee.batch.Export.image.toDrive(
            reprojection.toFloat(),
            description=f"Image {filename}",
            folder=f"{sat_name_c}",
            fileNamePrefix=filename,
            region=gee_polygon,
            fileFormat="GeoTIFF",
            maxPixels=1e10
        )
        task.start()

        all_tasks.append(task)
        filenames.append(filename)

    return all_tasks, filenames
