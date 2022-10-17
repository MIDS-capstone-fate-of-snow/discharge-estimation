"""Script to download images from GEE and upload to AWS S3."""

import datetime
import os
import platform
import time

from multiprocess import Process  # NOQA
import multiprocess

from aws_s3_client import S3Client
from gdrive_client import GDriveClient
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
# DATE_FROM = "2010_01_01"
# DATE_TO = "2019_12_31"
DATE_FROM = "2020_01_01"
DATE_TO = "2020_01_02"
BUFFER_PERCENT = 0.05
CRS = "epsg:4326"
SCALE = None  # None means don't rescale.
SAT_BANDS = [
    # ("MODIS/006/MOD16A2", "ET"),
    ("ECMWF/ERA5_LAND/HOURLY", "total_precipitation"),
    # ("ECMWF/ERA5_LAND/HOURLY", "temperature_2m"),
]


if __name__ == "__main__":

    t1 = time.time()  # Runtime timer.

    start_time = datetime.datetime.now(datetime.timezone.utc)

    # Object to scan temp dir for tif files, upload to S3, and delete locally:
    s3 = S3Client(directory=TEMP_DIR, bucket=BUCKET, test=False)

    # Object to scan service account GDrive for files, download to temp dir, and delete in GDrive:
    gdrive = GDriveClient(key_fp=GDRIVE_KEYS, local_dir=TEMP_DIR)

    # Client to make download requests to GEE:
    gee = GEEClient(SERVICE_ACCOUNT, GDRIVE_KEYS)

    # Start the processes for continually downloading from GDrive and uploading to S3:
    if platform.system() == "Darwin":
        multiprocess.set_start_method("spawn")  # NOQA
    process_gd = Process(target=gdrive, kwargs={"file_extensions": ["tif"]})
    process_s3 = Process(target=s3, kwargs={"s3_directory": STREAMGAGE_NAME, "file_ext": "tif"})
    process_gd.start()
    process_s3.start()

    # Iterate through satellite-bands creating download tasks to GDrive:
    for sat, band in SAT_BANDS:
        gee.export_ic_to_gdrive(
            bounding_box=BOUNDING_BOX, sat_name=sat, band=band,
            date_from=DATE_FROM, date_to=DATE_TO, buffer_percent=BUFFER_PERCENT, crs=CRS, scale=SCALE
        )

    # Wait for all EE tasks to complete:
    while gee.active:
        time.sleep(5)

    # Wait for all files to be uploaded to S3:
    ee_files = list(gee.tasks["filename"])
    filenames = [f"{f}.tif" for f in ee_files]
    while not s3.filenames_in_bucket(filenames, from_time=start_time, directory_name=STREAMGAGE_NAME):
        time.sleep(5)

    process_s3.kill()
    process_gd.kill()

    elapsed_time = time.time() - t1
    print(f"Finished - elapsed_time = {elapsed_time:,.2f} seconds")
