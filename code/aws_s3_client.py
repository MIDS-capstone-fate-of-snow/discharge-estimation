"""Client for working with files stored in AWS S3."""

import datetime
import os
import tempfile
import time
import warnings

import boto3
from botocore.exceptions import ClientError
import pandas as pd
from tqdm import tqdm

from logger import logging
from tif_files import TifFile
from utils import expected_image_dates, sat_img_filelist_df

DATE_FORMAT = "%Y_%m_%d"


def upload_file_to_s3(fp: str, bucket: str, object_name: str = None,
                      s3_directory: str = None) -> bool:
    """Upload a file to an S3 bucket

    Args:
        fp: path to local file to upload.
        bucket: S3 bucket to upload to.
        object_name: S3 object name. If not specified then file_name is used.
        s3_directory: optional subdirectory in S3 to save to.
    """
    # If S3 object_name was not specified, use file_name:
    if object_name is None:
        object_name = os.path.basename(fp)
    if s3_directory is not None:
        object_name = f"{s3_directory}/{object_name}"

    # Upload the file:
    s3_client = boto3.client("s3")
    try:
        return s3_client.upload_file(fp, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False


class S3Client:

    def __init__(self, directory: str, bucket: str, test: bool = True):
        """Client for managing files in an AWS S3 bucket.

        Args:
            directory: local directory to download files to.
            bucket: name of S3 bucket client connects to.
            test: whether or not to run in test mode, which means files won't
                actually be deleted.
        """
        assert os.path.isdir(directory), f"Invalid directory path: {directory}"
        self.directory = directory
        self.bucket = bucket
        self.test = test
        self.test_file_prefix = "TEST__" if self.test else ""
        self.test_msg_prefix = "(TEST) " if self.test else ""
        self.test_deleted = list()

    def list_bucket(self, directory_name: str = None):
        """List files in the S3 bucket."""
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(self.bucket)
        if directory_name is not None:
            objects = bucket.objects.filter(Prefix=f"{directory_name}/")
        else:
            objects = bucket.objects.all()
        name_date_owner_size = [(o.key, o.last_modified, o.owner, o.size) for o in objects]
        df = pd.DataFrame(name_date_owner_size, columns=["filepath", "last_modified_date", "owner", "size"])
        df["filename"] = [fn.split("/")[-1] for fn in df["filepath"]]
        df["filename_prefix"] = [fn.split(".")[0] for fn in df["filename"]]
        df["filename_ext"] = [fn.split(".")[-1] for fn in df["filename"]]
        return df

    def filenames_in_bucket(self, filenames: list,
                            from_time: datetime.datetime = None,
                            directory_name: str = None) -> bool:
        """Check all filenames are in the S3 bucket.

        Args:
            filenames: filenames with file extensions.
            from_time: optional minimum time when files were last modified.
            directory_name: optional S3 directory name to limit search to.
        """
        df = self.list_bucket(directory_name=directory_name)
        if from_time is not None:
            df = df[df["last_modified_date"] >= from_time]
        missing = set(filenames) - set(df["filename"])
        return not len(missing)

    def make_filepath(self, filename: str):
        return os.path.join(self.directory, filename)

    def list_local_files(self, file_ext: str = None):
        files = os.listdir(self.directory)
        # Skip part files:
        files = [f for f in files if not f.endswith(".part")]  # NOQA
        if file_ext is not None:
            files = list(filter(lambda fn: fn.endswith(f".{file_ext}"), files))
        if self.test:
            files = list(set(files) - set(self.test_deleted))
        return files

    def delete_local_file(self, filename: str):
        fp = self.make_filepath(filename)
        if self.test:
            self.test_deleted.append(filename)
        else:
            os.remove(fp)
        msg = f"{self.test_msg_prefix}Deleted local file: {fp}"
        logging.info(msg)
        print(msg)

    def delete_s3_file(self, *filepath: str):
        """Delete files on S3.

        Args:
            filepath: full relative filepath in S3 bucket.
        """
        client = boto3.client("s3")
        out = []
        for fp in filepath:
            out.append(client.delete_object(Bucket=self.bucket, Key=fp))
        return out

    def list_gee_tif_files(self, directory_name: str = None):
        """List all GeoTiff files saved in S3 from GEE, with specific filename
        formats."""
        df = self.list_bucket(directory_name)
        df = df[df["filename_ext"] == "tif"]
        filename_parts = ["crs", "scale", "satellite", "band"]
        for i, part in enumerate(filename_parts):  # NOQA
            df[part] = df["filename_prefix"].map(lambda s: s.split("__")[i])
        df["scale"] = df["scale"].str.replace("_", ".").astype(float)
        dates = df["filename_prefix"].map(lambda s: s.split("__")[-1])
        # Remove version numbers and hours if present:
        df["date"] = dates.map(lambda s: s[:10])
        df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT)
        return df

    def download_to_local(self, *filename, skip_existing: bool = True,
                          custom_dir: str = None, shhh: bool = False):
        """Download files from the S3 bucket to the local directory.

        Args:
            filename: full S3 filename including subdirectories.
            skip_existing: if True, skip files which already exist locally.
            custom_dir: optional custom location to store files, otherwise
                `self.directory` is used.
            shhh: if True, suppress print statuses.
        """
        files = list(set(filename))

        s3 = boto3.client("s3")
        for fname in files:
            subdirectories = fname.split("/")
            base_dir = self.directory if custom_dir is None else custom_dir
            # Create sub-directories locally if they don't exist:
            for subdir in subdirectories[:-1]:
                base_dir = os.path.join(base_dir, subdir)
                if not os.path.exists(base_dir):
                    os.mkdir(base_dir)
            local_fname = fname.split("/")[-1]
            target = os.path.join(base_dir, local_fname)
            if skip_existing and os.path.exists(target):
                pass
            else:
                s3.download_file(Bucket=self.bucket, Key=fname, Filename=target)
                if not shhh:
                    print(f"Downloaded S3 file to: {target}")

    def agg_img_pixels(self, *filepath,
                       delete_after: bool = True) -> pd.DataFrame:
        """Compute (not-nan) sum, mean, min, max of all pixels in images.

        Args:
            filepath: full S3 filepath to image.
            delete_after: if True, delete local file after computation.
        """
        temp_dir = tempfile.gettempdir()
        results = list()
        for fp in tqdm(filepath):
            self.download_to_local(fp, custom_dir=temp_dir, shhh=True)
            local_fp = os.path.join(temp_dir, fp)
            tif = TifFile(local_fp)
            results.append((fp, tif.pixel_nansum, tif.pixel_nanmean, tif.pixel_nanmin, tif.pixel_nanmax))
            if delete_after:
                os.remove(local_fp)
        columns = ["filepath"] + [f"pixel_{op}" for op in ("sum", "mean", "min", "max")]
        return pd.DataFrame(results, columns=columns)

    def download_training_data(self, gage_name: str, date_from: str,
                               date_to: str, skip_existing: bool = True):
        """Download training data from S3 to local for model training. Performs
        checks to make sure all required data for the periods given is present.

        Args:
            gage_name: streamgage name (directory name in S3 bucket).
            date_from: inclusive start date to get training data for.
            date_to: inclusive end data to get training data for.
            skip_existing: if True, skip local files which already exist.
        """

        # Convert date strings to datetime objects, and define all dates in range:
        dt_from = datetime.datetime.strptime(date_from, DATE_FORMAT)
        dt_to = datetime.datetime.strptime(date_to, DATE_FORMAT)
        assert dt_to >= dt_from

        # List all tif files in the bucket directory for the gage:
        tifs = self.list_gee_tif_files(directory_name=gage_name)

        # Filter to target date period:
        tifs = tifs[(tifs["date"] >= dt_from) & (tifs["date"] <= dt_to)]

        # Make the directory structure for downloading into:
        target_dir = self.directory
        for subdir in ("training_data", "raw"):
            target_dir = os.path.join(target_dir, subdir)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

        # Run the report to check for missing data:
        report = self.gage_data_report(gage_name, date_from, date_to)

        # Get the data for each band:
        for band, freq in [("total_precipitation", 1), ("temperature_2m", 1), ("ET", 8)]:

            df = tifs[tifs["band"] == band]

            # Warn missing dates:
            missing = report[band]["missing_dates"]
            if len(missing):
                warnings.warn(f"Missing {band} data for dates:\n{sorted(missing)}")

            # Dedupe multiple files for the same date:
            if report[band]["has_dupes"]:
                dupe_dates = report[band]["dupes"].index
                warnings.warn(f"Duplicate {band} files found for below dates. (keeping most recently "
                              f"updated file but this may not be correct):\n{sorted(dupe_dates)}")
                df = df.sort_values(by=["date", "last_modified_date"], ascending=[True, False])
                df = df.drop_duplicates(subset=["date"], keep="first")

            # Download to local:
            print(f"Downloading {gage_name}, {band} data to {target_dir}")
            for fp in tqdm(df["filepath"]):
                self.download_to_local(fp, custom_dir=target_dir, skip_existing=skip_existing, shhh=True)

    def gage_data_report(self, gage_name: str, date_from: str,
                         date_to: str) -> dict:
        """Get a report of what data is in S3 for the gage in the date range.

        Args:
            gage_name: name of streamgage.
            date_from: inclusive start date.
            date_to: inclusive end date.
        """
        # Convert date strings to datetime objects, and define all dates in range:
        dt_from = datetime.datetime.strptime(date_from, DATE_FORMAT)
        dt_to = datetime.datetime.strptime(date_to, DATE_FORMAT)
        assert dt_to >= dt_from

        # Get dataframe of all images currently in bucket for streamgage:
        df = self.list_bucket(gage_name)
        extracted_columns = sat_img_filelist_df(df["filepath"]).rename(columns={"filename": "filepath"})
        keep_cols = "filepath subdir crs scale satellite band date_str date".split()
        df = pd.merge(df, extracted_columns[keep_cols], left_on="filepath", right_on="filepath")

        # Create the 'report' as a dict of metadata about each band's data:
        report = dict()
        for band, freq in (("total_precipitation", 1), ("temperature_2m", 1), ("ET", 8)):
            data = dict(freq=freq)
            subset = df[df["band"] == band]

            data["empty_files"] = len(subset[subset["size"] == 0])
            subset = subset[subset["size"] > 0]

            data["min_date"] = min(subset["date"])
            data["max_date"] = max(subset["date"])
            data["count"] = subset["date"].count()
            data["nunique"] = subset["date"].nunique()
            data["has_dupes"] = data["nunique"] < data["count"]
            dates_expected = expected_image_dates(dt_from, dt_to, freq=data["freq"])
            data["missing_dates"] = dates_expected - set(subset["date"])
            value_counts = subset["date"].value_counts().sort_values(ascending=False)
            data["dupes"] = value_counts[value_counts > 1]
            data["dates_with_dupes_count"] = len(data["dupes"])
            data["subset"] = subset
            report[band] = data

        return report

    def __call__(self, s3_directory: str = None, file_ext: str = None,
                 delete_local: bool = True):
        while True:
            local_files = self.list_local_files(file_ext=file_ext)
            if len(local_files):
                for filename in local_files:
                    fp = self.make_filepath(filename)
                    object_name = f"{self.test_file_prefix}{filename}"
                    upload_file_to_s3(fp, self.bucket, object_name, s3_directory)
                    msg = f"{self.test_msg_prefix}Uploaded local file to S3: {object_name}"
                    if delete_local:
                        self.delete_local_file(filename)
                        msg = f"{msg} (deleted from local)"
                    logging.info(msg)
                    print(msg)
            else:
                time.sleep(1)


if __name__ == "__main__":

    # Generate 10 empty test files, and run the uploader in non-test mode (so
    # that they actually get deleted from local after being uploaded).

    import string
    import random

    # Get this directory location:
    DIR, FILENAME = os.path.split(__file__)
    test_directory = os.path.join(os.path.dirname(DIR), "tests")

    if not os.path.isdir(test_directory):
        os.mkdir(test_directory)
    for i in range(10):
        test_filename = "".join(random.choices(string.ascii_uppercase+string.ascii_lowercase, k=10))
        test_filename = f"TEST__{test_filename}.txt"
        test_fp = os.path.join(test_directory, test_filename)
        with open(test_fp, "w") as f:
            pass

    s3uploader = S3Client(test_directory, bucket="w210-snow-fate", test=False)
    s3uploader(s3_directory="test", file_ext="txt")
