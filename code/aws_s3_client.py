
import datetime
import os
import time

import boto3
from botocore.exceptions import ClientError
import pandas as pd

from logger import logging


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
        names = [o.key for o in objects]
        modified_dates = [o.last_modified for o in objects]
        df = pd.DataFrame({"filepath": names, "last_modified_date": modified_dates})
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

    def __call__(self, s3_directory: str = None, file_ext: str = None):
        while True:
            local_files = self.list_local_files(file_ext=file_ext)
            if len(local_files):
                for filename in local_files:
                    fp = self.make_filepath(filename)
                    object_name = f"{self.test_file_prefix}{filename}"
                    upload_file_to_s3(fp, self.bucket, object_name, s3_directory)
                    msg = f"{self.test_msg_prefix}Uploaded local file to S3: {object_name}"
                    logging.info(msg)
                    print(msg)
                    self.delete_local_file(filename)
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
