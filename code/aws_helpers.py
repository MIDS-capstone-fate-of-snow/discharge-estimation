
import os
import time

import boto3
from botocore.exceptions import ClientError

from logger import logging


def upload_file_to_s3(fp: str, bucket: str, object_name: str = None) -> bool:
    """Upload a file to an S3 bucket

    Args:
        fp: path to local file to upload.
        bucket: S3 bucket to upload to.
        object_name: S3 object name. If not specified then file_name is used.
    """
    # If S3 object_name was not specified, use file_name:
    if object_name is None:
        object_name = os.path.basename(fp)

    # Upload the file:
    s3_client = boto3.client("s3")
    try:
        return s3_client.upload_file(fp, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False


class S3Uploader:

    def __init__(self, directory: str, file_ext: str, bucket: str,
                 test: bool = True):
        assert os.path.isdir(directory), f"Invalid directory path: {directory}"
        self.directory = directory
        self.file_ext = file_ext
        self.bucket = bucket
        self.test = test
        self.test_file_prefix = "TEST__" if self.test else ""
        self.test_msg_prefix = "(TEST) " if self.test else ""
        self.test_deleted = list()

    def make_filepath(self, filename: str):
        return os.path.join(self.directory, filename)

    @property
    def list_local_files(self):
        files = os.listdir(self.directory)
        files = list(filter(lambda fn: fn.endswith(f".{self.file_ext}"), files))
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

    def __call__(self):
        while True:
            local_files = self.list_local_files
            if len(local_files):
                for filename in local_files:
                    fp = self.make_filepath(filename)
                    object_name = f"{self.test_file_prefix}{filename}"
                    upload_file_to_s3(fp, self.bucket, object_name)
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

    s3uploader = S3Uploader(test_directory, "txt", bucket="w210-snow-fate", test=False)
    s3uploader()
