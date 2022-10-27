"""User API for discharge estimation."""

import datetime
import json
import os
import platform
import time
import warnings

import multiprocess
from multiprocess import Process  # NOQA
import pandas as pd

from aws_s3_client import S3Client
from gdrive_client import GDriveClient
from gee_client import GEEClient
from utils import get_y_data, sat_img_filelist_df

DATE_FORMAT = "%Y_%m_%d"


class DischargeAPI:

    def __init__(self, local_data_dir: str, s3_bucket: str, gdrive_keys: str,
                 service_account: str):
        """

        Args:
            local_data_dir: local data directory where files will be stored.
            s3_bucket: name of S3 bucket data will be stored in.
            gdrive_keys: path to keys JSON file for GDrive service account.
            service_account: GEE service account name, e.g.
                username@project.iam.gserviceaccount.com
        """
        if not os.path.isdir(local_data_dir):
            raise NotADirectoryError(local_data_dir)
        self.local_data_dir = local_data_dir
        temp_dir = os.path.join(self.local_data_dir, "temp")
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        self.temp_dir = temp_dir
        self.gdrive_keys = gdrive_keys
        self.service_account = service_account
        self.s3_bucket = s3_bucket
        self.S3Client = S3Client(self.local_data_dir, self.s3_bucket)
        self.GEEAPI = GEEAPI(local_dir=self.temp_dir, gdrive_keys=self.gdrive_keys,
                             service_account=self.service_account, s3_bucket=self.s3_bucket)

        # Shortuts to util functions:
        self.get_y_data = get_y_data

    def create_pixel_agg_csv(self, gage: str,
                             sat_name: str = "MODIS_006_MOD16A2",
                             band: str = "ET"):
        """Create CSV of the aggregate pixel values for each image in a
        satellite band for a target gage.

        Args:
            gage: target gage name.
            sat_name: satellite name as in S3 image filenames.
            band: satellite name as in S3 image filenames.
        """
        tifs = self.S3Client.list_gee_tif_files(gage)
        tifs = tifs[(tifs["satellite"] == sat_name) & (tifs["band"] == band)]
        agg_df = self.S3Client.agg_img_pixels(*tifs["filepath"])
        fl_df = sat_img_filelist_df(tifs["filepath"])
        df = pd.concat([fl_df.set_index("filename"), agg_df.set_index("filepath")], axis=1)
        df.index.name = "filepath"
        df = df.reset_index()
        df["gage"] = gage
        target_dir = self.local_data_dir
        for subdir in (sat_name, band):
            target_dir = os.path.join(target_dir, subdir)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
        filename = f"{gage}__{sat_name}__{band}.csv"
        fp = os.path.join(target_dir, filename)
        df.to_csv(fp, encoding="utf-8", index=False)
        print(f"{len(df)} rows saved to CSV: {fp}")
        return fp


class GEEAPI:

    def __init__(self, local_dir: str, gdrive_keys: str, service_account: str,
                 s3_bucket: str = None):
        """API for managing training data for models.

        Args:
            local_dir: local directory to save images to.
            gdrive_keys: path to keys JSON file for GDrive service account.
            service_account: GEE service account name, e.g.
                username@project.iam.gserviceaccount.com
            s3_bucket: name of S3 bucket data will be stored in.
        """
        if not os.path.isdir(local_dir):
            raise NotADirectoryError(local_dir)
        self.local_dir = local_dir
        if not os.path.exists(gdrive_keys):
            raise FileNotFoundError(gdrive_keys)
        self.gdrive_keys = gdrive_keys
        assert isinstance(service_account, str)
        self.service_account = service_account
        self.s3_bucket = s3_bucket

    def local_subdir(self, *subdir):
        """Create a local subdirectory of the main class `local_dir` attribute.

        Args:
            *subdir: subdirectory name

        Returns:
            Full path to subdirectory.
        """
        fp = self.local_dir
        for d in subdir:
            fp = os.path.join(fp, d)
            if not os.path.exists(fp):
                os.mkdir(fp)
        return fp

    def get_gee_images(self, sat: str, band: str, bounding_box: tuple,
                       date_from: str, date_to: str,
                       delete_local: bool = False,
                       local_subdir: str = None, to_s3: bool = True,
                       s3_dir: str = None, crs: str = None,
                       buffer_percent: float = 0.05, scale: float = None,
                       hourly: bool = False, h_d_agg: str = None,
                       **filters):
        """Spawn the following processes to scrape GEE for satellite images:
            1. Send requests to GEE for images meeting criteria;
            2. Create a GDrive client to download images to local;
            3. Create an AWS S3 client to move files from local to an S3 bucket.

        The 3rd stage is optional - images can just be stored locally.

        Args:
            sat: GEE satellite name.
            band: satellite band.
            bounding_box: tuple of (left, bottom, right, top) defining region.
            date_from: inclusive start date.
            date_to: exclusive end date.
            delete_local: if True and uploading to S3, delete from local after.
            local_subdir: optional subdir of main instance `local_dir`.
            to_s3: if True, upload files to S3, else just keep locally.
            s3_dir: if uploading to S3, optional sub-directory to save to.
            buffer_percent: percent of bounding box area to add as buffer.
            crs: optional coordinate reference system, otherwise the satellite
                band's default is used.
            scale: optional desired image resolution, otherwise the satellite
                band's default is used.
            hourly: if True, append hour to file date stamp for hourly data.
            h_d_agg: optional agg function to convert hourly data to daily;
                either `mean` or `sum`.
            filters: key-value pairs of additional filters to apply to
                properties of ImageCollection (e.g. hour from hourly datasets).
        """
        t1 = time.time()  # Runtime timer.
        start_time = datetime.datetime.now(datetime.timezone.utc)  # Min time for checking all files scraped.

        if local_subdir is not None:
            local_dir = self.local_subdir(local_subdir)
        else:
            local_dir = self.local_dir

        # Client to make requests to GEE:
        gee = GEEClient(self.service_account, self.gdrive_keys)

        # Client to scan service account GDrive for files, download to local, and delete in GDrive:
        gdrive = GDriveClient(key_fp=self.gdrive_keys, local_dir=local_dir)

        # Client to upload files to S3:
        if to_s3:
            s3 = S3Client(directory=local_dir, bucket=self.s3_bucket, test=False)

        # Start the processes for continually downloading from GDrive and uploading to S3:
        if platform.system() == "Darwin":
            multiprocess.set_start_method("spawn")  # NOQA
        process_gd = Process(target=gdrive, kwargs={"file_extensions": ["tif"]})
        process_gd.start()
        if to_s3:
            process_s3 = Process(target=s3, kwargs={"s3_directory": s3_dir, "file_ext": "tif",  # NOQA
                                                    "delete_local": delete_local})
            process_s3.start()

        # Create tasks for satellite-band to download to GDrive:
        gee.export_ic_to_gdrive(
            bounding_box=bounding_box, sat_name=sat, band=band, date_from=date_from, date_to=date_to,
            buffer_percent=buffer_percent, crs=crs, scale=scale, hourly=hourly, h_d_agg=h_d_agg, **filters
        )

        # Wait for all EE tasks to complete:
        while gee.active:
            time.sleep(5)

        # Check for GEE failures:
        failures = gee.failures
        if len(failures):
            warnings.warn(f"{len(failures)} GEE task(s) failed - returning task objects for debugging.")
            if to_s3:
                process_s3.kill()  # NOQA
            process_gd.kill()
            return failures

        # Wait for all files to be uploaded to S3:
        if to_s3:
            ee_files = list(gee.tasks["filename"])
            filenames = [f"{fn}.tif" for fn in ee_files]
            while not s3.filenames_in_bucket(filenames, from_time=start_time, directory_name=s3_dir):
                time.sleep(5)
            process_s3.kill()  # NOQA

        # Wait for all Gdrive files to finish downloading:
        while len(gdrive.to_download(file_extensions=['tif'])):
            time.sleep(2)
        process_gd.kill()

        elapsed_time = time.time() - t1
        print(f"Finished - elapsed_time = {elapsed_time:,.2f} seconds")
        return gee._tasks  # NOQA


if __name__ == "__main__":

    DIR = os.getcwd()
    DATA_DIR = os.path.join(os.path.dirname(DIR), "data")
    TEMP_DIR = os.path.join(DATA_DIR, "temp")

    GAGE_NAMES = ["11202710", "11185500", "11189500", "11208000", "11318500"]  # "11402000", "11266500"

    for gage_name in GAGE_NAMES:

        bbox_fp = os.path.join(DATA_DIR, "watershed_bounding_boxes.json")
        with open(bbox_fp, "r") as f:
            BBOXES = json.load(f)

        for year in range(2010, 2021, 1):

            GDRIVE_KEYS = os.path.join(os.path.expanduser("~"), "snow-capstone-4a3c9603fcf0.json")
            SERVICE_ACCT = "capstone-gee-account@snow-capstone.iam.gserviceaccount.com"
            BUCKET = "w210-snow-fate"
            BOUNDING_BOX = BBOXES[gage_name]

            gee_api = GEEAPI(local_dir=TEMP_DIR, gdrive_keys=GDRIVE_KEYS,
                             service_account=SERVICE_ACCT, s3_bucket=BUCKET)

            # Request raw for MODIS-ET:
            _ = gee_api.get_gee_images(
                sat="MODIS/006/MOD16A2",
                band="ET",
                bounding_box=BOUNDING_BOX,
                date_from=f"{year}_01_01",
                date_to=f"{year+1}_01_01",
                delete_local=True,
                local_subdir=None,
                to_s3=True,
                s3_dir=gage_name,
                crs="EPSG:4326",
                buffer_percent=0.05,
                scale=None,
                hourly=False,
            )

            # Request mean images for temperature:
            _ = gee_api.get_gee_images(
                sat="ECMWF/ERA5_LAND/HOURLY",
                band="temperature_2m",
                bounding_box=BOUNDING_BOX,
                date_from=f"{year}_01_01",
                date_to=f"{year+1}_01_01",
                delete_local=True,
                local_subdir=None,
                to_s3=True,
                s3_dir=gage_name,
                crs=None,
                buffer_percent=0.05,
                scale=None,
                hourly=False,
                h_d_agg="mean"
            )

            # Request sum images for precipitation:
            _ = gee_api.get_gee_images(
                sat="ECMWF/ERA5_LAND/HOURLY",
                band="total_precipitation",
                bounding_box=BOUNDING_BOX,
                date_from=f"{year}_01_01",
                date_to=f"{year+1}_01_01",
                delete_local=True,
                local_subdir=None,
                to_s3=True,
                s3_dir=gage_name,
                crs=None,
                buffer_percent=0.05,
                scale=None,
                hourly=False,
                h_d_agg="sum"
            )
