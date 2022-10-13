"""Class for working with GDrive to download files to local."""

import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import remove_punctuation

from google_helper_functions import connect_to_service_account_gdrive


class GDriveFileDownloader:
    """Class for downloading files on GDrive to the local machine."""

    def __init__(self, key_fp: str, local_dir: str):
        self.drive = connect_to_service_account_gdrive(key_fp)
        assert os.path.isdir(local_dir), f"Not a directory: {local_dir}"
        self.local_dir = local_dir

    @property
    def undeleted_gdrive_files(self):
        file_list = self.drive.ListFile({"q": f"trashed=false"}).GetList()
        if len(file_list):
            file_df = pd.DataFrame([dict(f) for f in file_list])
            file_df["file"] = file_list
        else:
            # Construct an empty dataframe so no errors raised elsewhere:
            file_df = pd.DataFrame()
        for mandatory_column in ["title", "fileExtension", "file", "originalFilename"]:
            if mandatory_column not in file_df.columns:
                file_df[mandatory_column] = np.nan
        return file_df

    @property
    def local_files(self):
        return os.listdir(self.local_dir)

    def to_download(self, file_extensions: list = ("tif", )):
        file_df = self.undeleted_gdrive_files
        file_df = file_df[file_df["fileExtension"].isin(file_extensions)]
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

    def download_from_gdrive(self, file_extensions: list = ("tif", ),
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
        downloaded, deleted, skipped = list(), list(), list()
        for ix in tqdm(list(file_df.index)):
            row = file_df.loc[ix]
            title = row["title"]
            target_fp = self.target_filepath(title, allow_duplicates=allow_duplicates)
            if target_fp is not None:
                gdrive_file_object = row["file"]
                gdrive_file_object.GetContentFile(target_fp)
                downloaded.append(target_fp)
                if delete:
                    gdrive_file_object.Delete()
                    deleted.append(row["originalFilename"])
            else:
                skipped.append({"title": title, "originalFilename": row["originalFilename"]})

        return {"downloaded": downloaded, "deleted": deleted, "skipped": skipped}

    def __call__(self, file_extensions: list = ("tif", ),
                 allow_duplicates: bool = True, delete: bool = True,
                 sleep: int = 1):
        while True:
            self.download_from_gdrive(file_extensions=file_extensions,
                                      allow_duplicates=allow_duplicates, delete=delete)
            time.sleep(sleep)
