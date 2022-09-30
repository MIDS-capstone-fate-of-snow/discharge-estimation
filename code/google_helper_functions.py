"""Helper functions for working with Google Earth Engine and Google Drive
service accounts."""

import datetime
import os
import warnings

import ee
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import pydrive2
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from tqdm import tqdm

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
