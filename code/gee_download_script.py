"""Script to download images from GEE and upload to AWS S3."""

import os
import platform
import time

from multiprocess import Process  # NOQA
import multiprocess

from aws_helpers import S3Uploader
from gdrive_downloader import GDriveFileDownloader
from gee_client import GEEClient

DIR, FILENAME = os.path.split(__file__)


# Variables:
# TODO: these can be collected with argparse.
STREAMGAGE_NAME = "11266500"
TEMP_DIR = os.path.join(os.path.dirname(DIR), "data", "temp")
GDRIVE_KEYS = os.path.join(os.path.expanduser("~"), "snow-capstone-4a3c9603fcf0.json")
BUCKET = "w210-snow-fate"
SERVICE_ACCOUNT = "capstone-gee-account@snow-capstone.iam.gserviceaccount.com"
BOUNDING_BOX = (-119.67587142994948, 37.5937475200786, -119.25727819655671, 37.90260095290917)
DATE_FROM = "2020_01_01"
DATE_TO = "2020_01_31"
BUFFER_PERCENT = 0.05
CRS = "epsg:4326"
SCALE = 100
SAT_BANDS = [
    ("MODIS/006/MOD16A2", "ET"),
]

if __name__ == "__main__":

    # Object to scan temp dir for tif files, upload to S3, and delete locally:
    s3_uploader = S3Uploader(directory=TEMP_DIR, file_ext="tif", bucket=BUCKET, test=False)

    # Object to scan service account GDrive for files, download to temp dir, and delete in GDrive:
    gdrive_downloader = GDriveFileDownloader(key_fp=GDRIVE_KEYS, local_dir=TEMP_DIR)

    # Client to make download requests to GEE:
    ee_client = GEEClient(SERVICE_ACCOUNT, GDRIVE_KEYS)

    # Start the processes for continually downloading from GDRive and uploading to S3:
    if platform.system() == "Darwin":  # TODO: check this works on AWS VM.
        multiprocess.set_start_method("spawn")  # NOQA

    process_gd = Process(target=gdrive_downloader, kwargs={"file_extensions": ["tif"]})
    process_s3 = Process(target=s3_uploader, kwargs={"s3_directory": STREAMGAGE_NAME})
    process_gd.start()
    process_s3.start()

    # Iterate through satellite-bands creating download tasks to GDrive:
    for sat, band in SAT_BANDS:
        ee_client.export_ic_to_gdrive(
            bounding_box=BOUNDING_BOX, sat_name=sat, band=band, date_from=DATE_FROM, date_to=DATE_TO,
            buffer_percent=BUFFER_PERCENT, crs=CRS, scale=SCALE
        )

    # Wait for all tasks to complete:
    while ee_client.active:
        time.sleep(1)
    process_s3.kill()
    process_gd.kill()
