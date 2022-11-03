"""Code for training a CNN-based sequence model using raw satellite images."""

import datetime
import os
import re

import pandas as pd

from utils import expected_image_dates

DATE_FORMAT = "%Y_%m_%d"


class CNNSeqDataset:

    def __init__(self,
                 precip_dirs: list,
                 temp_dirs: list,
                 et_dirs: list,
                 swe_dirs: list,
                 y_fp: str,
                 y_col: str = "m3",
                 n_d_precip: int = 7,
                 n_d_temp: int = 7,
                 n_d_et: int = 8,
                 swe_d_rel: list = (7, 14, 21, 28),
                 n_d_y: int = 14,
                 min_date: str = "2010_01_01",
                 max_date: str = "2016_12_31",
                 val_start: str = "2015_01_01",
                 test_start: str = "2016_01_01"):
        """Construct training dataset of images for training models."""

        # Store variables:
        self.n_d_precip = n_d_precip
        self.n_d_temp = n_d_temp
        assert n_d_et >= 8, f"MODIS data is only every 8 days, so must use 8 days minimum"
        self.n_d_et = n_d_et
        self.swe_d_rel = swe_d_rel
        self.n_d_y = n_d_y

        self.min_date = datetime.datetime.strptime(min_date, DATE_FORMAT)
        self.max_date = datetime.datetime.strptime(max_date, DATE_FORMAT)
        self.val_start = datetime.datetime.strptime(val_start, DATE_FORMAT)
        self.test_start = datetime.datetime.strptime(test_start, DATE_FORMAT)

        # Expected dates for MODIS:
        modis_dates = pd.Series(sorted(expected_image_dates(self.min_date, self.max_date, 8)))
        modis_dates.name = "modis_dates"
        self.modis_dates = modis_dates

        # File formats:
        self.file_formats = {
            "precip": "tif",
            "temp": "tif",
            "et": "tif",
            "swe": "npy",
        }

        # Directories where files for each image source are stored:
        self.dirs = {
            "precip": precip_dirs,
            "temp": temp_dirs,
            "et": et_dirs,
            "swe": swe_dirs,
        }

        # Metadata of all files for each band:
        self.file_metadata = dict()
        for band, dirs in self.dirs.items():
            file_metadata = list()
            ffmt = self.file_formats[band]
            for dr in dirs:
                filenames = [f for f in os.listdir(dr) if band.lower() in f.lower() and f.endswith(f".{ffmt}")]
                for fn in filenames:
                    metadata = self.extract_filename_data(fn)
                    metadata["full_filepath"] = os.path.join(dr, fn)
                    file_metadata.append(metadata)
            self.file_metadata[band] = file_metadata

        # Dataframe of all metadata:
        dfs = list()
        for band, metadata in self.file_metadata.items():
            df = pd.DataFrame(metadata)
            df["band"] = band
            dfs.append(df)
        self.metadata_df = pd.concat(dfs, axis=0).reset_index(drop=True)

        # Lookup for filepaths:
        self.fp_lookup = self.metadata_df[
            ["streamgage", "band", "date", "full_filepath"]
        ].set_index(["streamgage", "band", "date"]).sort_index()

        # Outcome data:
        self.y_col = y_col
        self.y_fp = y_fp
        self.y_df = pd.read_csv(y_fp, encoding="utf-8")
        self.y_df["gage"] = self.y_df["gage"].astype(str)
        self.y_df["date"] = pd.to_datetime(self.y_df["time"])
        self.y = self.y_df.set_index(["gage", "date"])[self.y_col]

        # Gages and bands in the data:
        self.gages = self.y_df["gage"].unique()
        self.bands = self.metadata_df["band"].unique()

        # Calculate which dates can be used for each gage:
        self.date_bounds = self.get_xy_date_bounds()
        self.y_dates = pd.date_range(self.date_bounds["y_min_date"], self.date_bounds["y_max_date"])

    @staticmethod
    def extract_filename_data(fn: str):
        """Extract gage, band, and date components from a filename using regex."""
        streamgage = re.findall("\d{8}", fn)  # NOQA
        assert len(streamgage) == 1, f"No 8-digit streamgage found in filename: {fn}"
        datestr = re.findall("\d{4}_\d{2}_\d{2}", fn)  # NOQA
        date = datetime.datetime.strptime(datestr[0], DATE_FORMAT)
        assert len(datestr) == 1, f"No datestring found in filename: {fn}"
        band = False
        for b in ("swe", "et", "temperature_2m", "total_precipitation"):
            if b in fn.lower():
                assert not band, f"Multiple bands in filename: {fn}"
                band = b
        assert band, f"No band in filename: {fn}"
        return {"fn": fn, "band": band, "date": date, "streamgage": streamgage[0]}

    def get_xy_date_bounds(self):
        """Calculate date boundaries for dataset based on X-y frequencies."""
        dates = dict()

        # Minimum y-variable dates based on how much prior X data is required:
        dates["min_temp"] = self.min_date + datetime.timedelta(days=self.n_d_temp)
        dates["min_precip"] = self.min_date + datetime.timedelta(days=self.n_d_precip)
        dates["min_swe"] = self.min_date + datetime.timedelta(days=max(self.swe_d_rel))
        dates["min_et"] = self.min_date + datetime.timedelta(days=self.n_d_et)
        y_min_date = max(dates.values())
        dates["y_min_date"] = y_min_date

        # Max y_date based on number of forward observations predicting:
        dates["y_max_date"] = self.max_date - datetime.timedelta(days=self.n_d_y)

        return dates

    def get_required_dates(self, y_date: datetime.datetime):
        """Get required feature dates for a target y-date."""
        if isinstance(y_date, str):
            y_date = datetime.datetime.strptime(y_date, DATE_FORMAT)

        req_dates = dict()

        # Outcome variable dates:
        req_dates["y"] = pd.date_range(y_date, y_date + datetime.timedelta(days=self.n_d_y - 1))

        # Daily temperature, precipitation dates:
        req_dates["temp"] = pd.date_range(y_date - datetime.timedelta(days=self.n_d_temp),
                                          y_date - datetime.timedelta(1))
        req_dates["precip"] = pd.date_range(y_date - datetime.timedelta(days=self.n_d_precip),
                                            y_date - datetime.timedelta(1))

        # Identify MODIS dates within range:
        et_max = y_date
        et_min = y_date - datetime.timedelta(days=self.n_d_et)
        et_dates = self.modis_dates[(self.modis_dates >= et_min) & (self.modis_dates < et_max)]
        req_dates["et"] = et_dates.values

        # SWE dates:
        swe_dates = list()
        for day in self.swe_d_rel:
            swe_dates.append(y_date - datetime.timedelta(day))
        req_dates["swe"] = sorted(swe_dates)

        return req_dates

    def get_required_filepaths(self, y_date: datetime.datetime, gage: str):
        """Get required image filepaths for a target y-date. Raises errors for
        missing files."""
        filepaths = dict()
        req_dates = self.get_required_dates(y_date)
        for band, dates in req_dates.items():
            if band != "y":
                try:
                    filepaths[band] = list(self.fp_lookup.loc[(gage, band)].loc[dates, "full_filepath"])
                except KeyError:
                    raise FileNotFoundError(f"Missing images for {gage=}, {band=}, {dates=}")
            else:
                try:
                    filepaths[band] = self.y.loc[gage].loc[dates]
                except KeyError:
                    raise FileNotFoundError(f"Missing images for {gage=}, {band=}, {dates=}")

        return filepaths

    def date_generator(self):
        for y_date in self.y_dates:
            yield self.get_required_dates(y_date)

    def dataset_generator(self):
        for gage in self.gages:
            for y_date in self.y_dates:
                yield self.get_required_filepaths(y_date, gage)
