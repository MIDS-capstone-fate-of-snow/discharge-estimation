"""User API for discharge estimation."""

import datetime
import os.path
import platform
import time

import multiprocess
from multiprocess import Process  # NOQA

from aws_s3_client import S3Client
from gdrive_client import GDriveClient
from gee_client import GEEClient


class DataAPI:

    def __init__(self, local_dir: str, gdrive_keys: str, service_account: str,
                 s3_bucket: str = None):
        """

        Args:
            local_dir: local directory to save images to.
            gdrive_keys: path to keys JSON file for GDrive service account.
            service_account: GEE service account name, e.g.
                username@project.iam.gserviceaccount.com
            s3_bucket:
        """
        if not os.path.isdir(local_dir):
            raise NotADirectoryError(local_dir)
        self.local_dir = local_dir
        if not os.path.exists(gdrive_keys):
            raise FileNotFoundError(gdrive_keys)
        self.gdrive_keys = gdrive_keys
        assert isinstance(service_account, str)
        self.service_account = service_account
        assert isinstance(s3_bucket, str)
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
                       s3_dir: str = None, crs: str = "epsg:4326",
                       buffer_percent: float = 0.05, scale: float = None,
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
            crs: coordinate reference system.
            scale: optional desired image resolution, otherwise default is used.
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
            buffer_percent=buffer_percent, crs=crs, scale=scale, **filters
        )

        # Wait for all EE tasks to complete:
        while gee.active:
            time.sleep(5)

        # Wait for all files to be uploaded to S3:
        if to_s3:
            ee_files = list(gee.tasks["filename"])
            filenames = [f"{f}.tif" for f in ee_files]
            while not s3.filenames_in_bucket(filenames, from_time=start_time, directory_name=s3_dir):
                time.sleep(5)
            process_s3.kill()  # NOQA

        # Wait for all Gdrive files to finish downloading:
        while len(gdrive.to_download(file_extensions=['tif'])):
            time.sleep(5)
        process_gd.kill()

        elapsed_time = time.time() - t1
        print(f"Finished - elapsed_time = {elapsed_time:,.2f} seconds")


if __name__ == "__main__":

    for year in range(2010, 2021, 1):

        DIR = os.getcwd()
        TEMP_DIR = os.path.join(os.path.dirname(DIR), "data", "temp")
        GDRIVE_KEYS = os.path.join(os.path.expanduser("~"), "snow-capstone-4a3c9603fcf0.json")
        SERVICE_ACCT = "capstone-gee-account@snow-capstone.iam.gserviceaccount.com"
        BUCKET = "w210-snow-fate"
        BOUNDING_BOX = (-119.67587142994948, 37.5937475200786, -119.25727819655671, 37.90260095290917)

        api = DataAPI(local_dir=TEMP_DIR, gdrive_keys=GDRIVE_KEYS, service_account=SERVICE_ACCT, s3_bucket=BUCKET)

        # Request once-daily images for temperature:
        api.get_gee_images(
            sat="ECMWF/ERA5_LAND/HOURLY",
            band="temperature_2m",
            bounding_box=BOUNDING_BOX,
            date_from=f"{year}_01_01",
            date_to=f"{year}_12_31",
            delete_local=True,
            local_subdir=None,
            to_s3=True,
            s3_dir="11266500",
            hour=12
        )