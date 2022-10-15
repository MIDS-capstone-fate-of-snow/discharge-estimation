"""Class for working with GDrive to download files to local."""

import os
import shutil
import time

import numpy as np
import pandas as pd

from utils import remove_punctuation

from google_helper_functions import connect_to_service_account_gdrive


class GDriveClient:
    """Class for managing files on GDrive."""

    def __init__(self, key_fp: str, local_dir: str):
        assert os.path.exists(key_fp), f"Invalid filepath: {key_fp}"
        self.key_fp = key_fp
        assert os.path.isdir(local_dir), f"Not a directory: {local_dir}"
        self.local_dir = local_dir
        self._downloaded = list()
        self._deleted = list()
        self._callback = None

    @property
    def downloaded(self):
        return pd.DataFrame(self._downloaded)

    @property
    def deleted(self):
        return pd.DataFrame(self._deleted)

    def connect(self):
        """Returns a connection to the GDrive."""
        return connect_to_service_account_gdrive(self.key_fp)

    @property
    def undeleted_files(self):
        drive = self.connect()
        file_list = drive.ListFile({"q": f"trashed=false"}).GetList()
        if len(file_list):
            file_df = pd.DataFrame([dict(f) for f in file_list])
            file_df["file"] = file_list
        else:
            # Construct an empty dataframe so no errors raised elsewhere:
            file_df = pd.DataFrame()
        for mandatory_column in ["title", "fileExtension", "file", "originalFilename", "fileSize", "webContentLink"]:
            if mandatory_column not in file_df.columns:
                file_df[mandatory_column] = np.nan
        return file_df

    def delete_all_files(self):
        """Delete all files currently in the GDrive."""
        files = self.undeleted_files
        for f in files["file"]:
            f.Delete()
        return files

    @property
    def local_files(self):
        return os.listdir(self.local_dir)

    def to_download(self, file_extensions: list = ("tif", )):
        file_df = self.undeleted_files
        file_df = file_df[
            (file_df["fileExtension"].isin(file_extensions)) &
            (file_df["fileSize"].astype(float) > 0) &
            (file_df["webContentLink"].notna())
            ]
        return file_df

    def target_filepath(self, filename: str, allow_duplicates: bool = True,
                        zfill: int = 5):
        """Construct a filepath to save a file in the local directory.

        Args:
            filename: name of file to save including extension.
            allow_duplicates: if True and the filename matches an existing file,
                appends a version number to the filename. If False returns None.
            zfill: number of zeros to pad version numbers with.
        """
        # Remove punctuation and replace with underscore:
        name, extension = filename.split(".")
        name = remove_punctuation(name)
        filename = f"{name}.{extension}"

        if filename in self.local_files:
            if not allow_duplicates:
                # Don't construct filepath:
                return None
            else:
                # Append version number for duplicate filenames:
                i = 1
                while True:
                    version = f"{i}".zfill(zfill)
                    new_filename = f"{name}_v{version}.{extension}"
                    if new_filename not in self.local_files:
                        filename = new_filename
                        break
                    else:
                        i += 1

        return os.path.join(self.local_dir, filename)

    def download_by_file_ext(self, file_extensions: list = ("tif",),
                             allow_duplicates: bool = True,
                             delete: bool = False):
        """Download files with specific file extensions from GDrive.

        Args:
            file_extensions: extensions of files to download.
            allow_duplicates: if False and filename already exists locally don't
                download again. If True a version number will be appended.
            delete: if True, delete the source file from GDrive after download.

        Returns:
            Dict of GDrive files downloaded, deleted, and skipped.
        """
        file_df = self.to_download(file_extensions)
        skipped = list()
        if len(file_df):
            for ix in list(file_df.index):
                self._callback = None
                row = file_df.loc[ix]
                title = row["title"]
                target_fp = self.target_filepath(title, allow_duplicates=allow_duplicates)
                if target_fp is not None:
                    part_fp = f"{target_fp}.part"

                    def callback(total_transferred, file_size):
                        """Log the file size of the downloaded file; used to
                        identify when the file has finished downloading."""
                        self._callback = total_transferred, file_size

                    gdrive_file_object = row["file"]
                    gdrive_file_object.GetContentFile(part_fp, callback=callback)
                    while self._callback is None:
                        time.sleep(0.1)  # Wait until file finishes downloading before removing .part extension
                    shutil.move(part_fp, target_fp)
                    metadata = dict(gdrive_file_object)
                    metadata["target_fp"] = target_fp
                    metadata["timestamp"] = time.time()
                    self._downloaded.append(metadata)
                    if delete:
                        gdrive_file_object.Delete()
                        self._deleted.append(metadata)
                else:
                    skipped.append({"title": title, "originalFilename": row["originalFilename"]})

        return {"skipped": skipped}

    def __call__(self, file_extensions: list = ("tif", ),
                 allow_duplicates: bool = True, delete: bool = True,
                 sleep: int = 1):
        while True:
            self.download_by_file_ext(file_extensions=file_extensions,
                                      allow_duplicates=allow_duplicates, delete=delete)
            time.sleep(sleep)
