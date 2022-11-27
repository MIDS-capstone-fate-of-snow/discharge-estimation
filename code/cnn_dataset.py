"""Code for training a CNN-based sequence model using raw satellite images."""

from collections import defaultdict
import datetime
from itertools import product
import os
import random
import warnings
import yaml

import numpy as np
import pandas as pd

from tif_files import TifFile
from utils import convert_datetime, expected_image_dates, extract_filename_data, open_npy_file, open_swe_file, \
    pixel_mean_std

DIR, FILENAME = os.path.split(__file__)
DATA_DIR = os.path.join(os.path.dirname(DIR), "data")
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
                 swe_d_rel: list = range(7, 85, 7),
                 n_d_y: int = 14,
                 y_seq: bool = True,
                 min_date: str = "2010_01_01",
                 max_date: str = "2016_12_31",
                 val_start: str = "2015_01_01",
                 test_start: str = "2016_01_01",
                 use_masks: bool = True,
                 gages: list = None,
                 random_seed: int = 42,
                 shuffle_train: bool = True):
        """Construct image training dataset for training CNN sequence models.

        Args:
            precip_dirs: directory(/ies) where precipitation images are saved.
            temp_dirs: directory(/ies) where temperature images are saved.
            et_dirs: directory(/ies) where ET images are saved.
            swe_dirs: directory(/ies) where SWE images are saved.
            y_fp: full filepath to CSV of streamgage measurement outcome data.
            y_col: column in `y_fp` to use as the outcome variable.
            n_d_precip: number of days to look back for precipitation images.
            n_d_temp: number of days to look back for temperature images.
            n_d_et: number of days to look back for ET images (since they are
                only every 8 days, must be at least 8).
            swe_d_rel: specific days back in the past to select SWE images from.
            n_d_y: number of days forward to predict y measurements for.
            y_seq: whether or not y should be a sequence of all days up to
                `n_d_y`, or just a single day's measurement at `n_d_y`.
            min_date: global minimum training dataset date.
            max_date: global maximum training dataset date.
            val_start: date cutoff to start the validation set at.
            test_start: date cutoff to start the test set at.
        """

        # Store variables:
        self.n_d_precip = n_d_precip
        self.n_d_temp = n_d_temp
        assert n_d_et >= 8, f"MODIS data is only every 8 days, so must use 8 days minimum"
        self.n_d_et = n_d_et
        self.swe_d_rel = list(swe_d_rel)
        self.n_d_y = n_d_y
        self.y_seq = y_seq

        self.min_date = convert_datetime(min_date)
        self.max_date = convert_datetime(max_date)
        self.val_start = convert_datetime(val_start)
        self.test_start = convert_datetime(test_start)

        self.use_masks = use_masks

        self.random_seed = random_seed
        self.shuffle_train = shuffle_train

        # Outcome data:
        self.y_col = y_col
        self.y_fp = y_fp
        self.y_df = pd.read_csv(y_fp, encoding="utf-8")
        self.y_df["gage"] = self.y_df["gage"].astype(str)
        self.y_df["date"] = pd.to_datetime(self.y_df["time"])
        self.y = self.y_df.set_index(["gage", "date"])[self.y_col]

        # Gages:
        if gages is None:
            self.gages = [str(g) for g in self.y_df["gage"].unique()]
        else:
            assert isinstance(gages, list), f"`gages` must be list of int/str"
            self.gages = [str(g) for g in gages]

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
                    metadata = extract_filename_data(fn)
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

        self.bands = self.metadata_df["band"].unique()

        # Calculate which dates can be used for each gage:
        self.date_bounds = self.get_xy_date_bounds()
        self.y_dates = pd.date_range(self.date_bounds["y_min_date"], self.date_bounds["y_max_date"])

        # Calculate the gage-date pairs for training/validation/test:
        self.train_pairs = self.get_training_pairs(min(self.y_dates), self.val_start - datetime.timedelta(days=1))
        if self.shuffle_train:
            random.seed(random_seed)
            random.shuffle(self.train_pairs)
        self.val_pairs = self.get_training_pairs(self.val_start, self.test_start - datetime.timedelta(days=1))
        self.test_pairs = self.get_training_pairs(self.test_start, max(self.y_dates))

        # Compute image normalization values for training period (if not already saved):
        self._norm_fp = os.path.join(DATA_DIR, "img_norm_values.yaml")
        if not os.path.exists(self._norm_fp):
            with open(self._norm_fp, "w") as f:
                yaml.safe_dump(dict(), f)
        self.img_norm = dict()
        for band in self.bands:
            self.img_norm[band] = self.get_img_norm(band, self.min_date,
                                                    (self.val_start - datetime.timedelta(days=1)),
                                                    recompute=False)

        # DEM dataset:
        dem_dir = os.path.join(DATA_DIR, "dem")
        dem_files = os.listdir(dem_dir)
        self.dem_fps = dict()
        for gage in self.gages:
            gage_dems = [f for f in dem_files if gage in f]
            assert len(gage_dems) == 1
            self.dem_fps[gage] = os.path.join(dem_dir, gage_dems[0])
        self.dem_norm_values = pixel_mean_std(*self.dem_fps.values())
        self.dems_raw, self.dems_norm = dict(), dict()
        for gage, fp in self.dem_fps.items():
            tif = TifFile(fp)
            self.dems_raw[gage] = tif.as_numpy
            self.dems_norm[gage] = (tif.as_numpy - self.dem_norm_values["pixel_mean"]) / \
                self.dem_norm_values["pixel_std"]

        if self.use_masks:
            self.ratio_masks, self.bool_masks = self.get_watershed_masks()
        else:
            self.ratio_masks, self.bool_masks = None, None

    @property
    def saved_img_norm(self):
        with open(self._norm_fp, "r") as f:
            return yaml.safe_load(f)

    def get_img_norm(self, band: str, date_from: datetime.datetime,
                     date_to: datetime.datetime, recompute: bool = False):
        """Compute pixel normalization values across all images within the dates
        given for the given band. Only uses images for gages at `self.gages`."""
        # Key to identify gages used:
        gage_key = "_".join(sorted(self.gages))

        # Filter metadata to get required image filepaths:
        date_from = convert_datetime(date_from, DATE_FORMAT)
        date_to = convert_datetime(date_to, DATE_FORMAT)
        df = self.metadata_df[
            (self.metadata_df["date"] >= date_from) &
            (self.metadata_df["date"] <= date_to) &
            (self.metadata_df["band"] == band.strip().lower()) &
            (self.metadata_df["streamgage"].isin(self.gages))
            ]

        # Get the actual min/max dates possible with the data:
        date_from, date_to = min(df["date"]), max(df["date"])
        date_key = f"{date_from.strftime(DATE_FORMAT)}_to_{date_to.strftime(DATE_FORMAT)}"

        if not recompute:  # Try to fetch precomputed values:
            try:
                return self.saved_img_norm[gage_key][band][date_key]
            except KeyError:
                pass

        # Calculate values:
        mu_std = pixel_mean_std(*df["full_filepath"].values)

        # Save values:
        saved = self.saved_img_norm
        if gage_key not in saved.keys():
            saved[gage_key] = dict()
        if band not in saved[gage_key].keys():
            saved[gage_key][band] = dict()
        saved[gage_key][band][date_key] = mu_std
        with open(self._norm_fp, "w") as f:
            yaml.safe_dump(saved, f)
        print(f"Save image pixel mean/std for gages `{gage_key}`, band `{band}`, dates `{date_key}`")

        return mu_std

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
        y_date = convert_datetime(y_date, DATE_FORMAT)

        req_dates = dict()

        # Outcome variable dates:
        if self.y_seq:  # y is a sequence of observations.
            req_dates["y"] = pd.date_range(y_date, y_date + datetime.timedelta(days=self.n_d_y - 1))
        else:  # y is a single observation.
            req_dates["y"] = [y_date + datetime.timedelta(days=self.n_d_y - 1)]

        # Daily temperature, precipitation dates:
        req_dates["temp"] = pd.date_range(y_date - datetime.timedelta(days=self.n_d_temp),
                                          y_date - datetime.timedelta(1))
        req_dates["precip"] = pd.date_range(y_date - datetime.timedelta(days=self.n_d_precip),
                                            y_date - datetime.timedelta(1))

        # Identify MODIS dates within range:
        et_max = y_date
        et_min = y_date - datetime.timedelta(days=self.n_d_et)
        et_dates = self.modis_dates[(self.modis_dates >= et_min) & (self.modis_dates < et_max)]
        n_et = self.n_d_et // 8
        et_dates = et_dates.values[-n_et:]
        req_dates["et"] = et_dates

        # SWE dates:
        swe_dates = list()
        for day in self.swe_d_rel:
            swe_dates.append(y_date - datetime.timedelta(day))
        req_dates["swe"] = sorted(swe_dates)

        return req_dates

    def get_required_filepaths(self, y_date: datetime.datetime, gage: str):
        """Get required image filepaths for a target y-date. Raises errors for
        missing files.

        Args:
            y_date: the first y-target date.
            gage: name of the streamgage location.
        """
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
        filepaths["debug_data"] = dict(y_date=y_date, gage=gage)
        return filepaths

    def filepaths_to_data(self, fps: dict):
        """Convert output of `get_required_filepaths` to actual data values.

        Args:
            fps: dict output from `get_required_filepaths` method.
        """
        data = dict()
        data["y"] = fps["y"].values
        for band in ("temp", "precip", "et", "swe"):
            arrays = list()
            for fp in fps[band]:
                if self.file_formats[band] == "tif":
                    try:
                        tif = TifFile(fp)
                    except Exception as e:
                        debug = fps['debug_data']
                        debug["band"] = band
                        warnings.warn(f"{debug} - error loading fp: {fp}")
                        raise e
                    arr = tif.as_numpy_zero_nan
                elif band == "swe":
                    arr = open_swe_file(fp)
                elif self.file_formats[band] == "npy":
                    raise NotImplementedError()
                    # arr = open_npy_file(fp)
                else:
                    raise ValueError(f"Invalid band: {band}")
                norm_arr = (arr - self.img_norm[band]["pixel_mean"]) / self.img_norm[band]["pixel_std"]
                arrays.append(norm_arr)
            data[band] = np.array(arrays)
        gage = fps["debug_data"]["gage"]
        data["dem"] = np.array([self.dems_norm[gage]])
        data["debug_data"] = fps["debug_data"]

        # Apply masking if required:
        if self.use_masks:
            data = self.apply_watershed_masks(data)

        return data

    def get_training_pairs(self, min_date: datetime.datetime,
                           max_date: datetime.datetime):
        """Get training pairs of (streamgage, date) within the date bounds.

        Args:
            min_date: inclusive minimum date.
            max_date: inclusive maximum date.
        """
        min_date = convert_datetime(min_date, DATE_FORMAT)
        max_date = convert_datetime(max_date, DATE_FORMAT)
        y_dates = self.y_dates[(self.y_dates >= min_date) & (self.y_dates <= max_date)]
        return list(product(self.gages, y_dates))

    def get_watershed_masks(self):
        """Get the watershed masks for each image type."""
        mask_dir = os.path.join(DATA_DIR, "masks")
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".npy")]
        ratio_masks, bool_masks = defaultdict(dict), defaultdict(dict)
        for band in ("temp", "dem", "precip", "et", "swe"):
            for gage in self.gages:
                files = list(filter(lambda f: (gage in f.lower()) and (band in f.lower()), mask_files))
                assert len(files) == 1
                arr = open_npy_file(os.path.join(mask_dir, files[0]))
                ratio_masks[band][gage] = arr
                bool_masks[band][gage] = np.where(arr, 1, 0)

        return ratio_masks, bool_masks

    def apply_watershed_masks(self, data: dict):
        """Apply watershed masks to training data images."""
        gage = data["debug_data"]["gage"]
        masked_data = data.copy()
        for band in ("temp", "dem", "precip", "et", "swe"):
            if band not in data:
                continue
            mask = None
            if band in ("temp", "dem"):  # Apply boolean masking:
                mask = self.bool_masks[band][gage]
            elif band in ("precip", "et", "swe"):  # Apply ratio masking:
                mask = self.ratio_masks[band][gage]
            if mask is not None:
                arrays = data[band]
                masked_arrays = list()
                for i, arr in enumerate(arrays):
                    assert mask.shape == arr.shape, f"Mask doesn't match array shape: {gage=}, {band=}, {i=}"
                    masked_arr = arr * mask
                    masked_arrays.append(masked_arr)
                masked_data[band] = np.array(masked_arrays)
        return masked_data

    def date_generator(self):
        for y_date in self.y_dates:
            yield self.get_required_dates(y_date)

    def train_filepath_generator(self):
        for gage, y_date in self.train_pairs:
            yield self.get_required_filepaths(y_date, gage)

    def val_filepath_generator(self):
        for gage, y_date in self.val_pairs:
            yield self.get_required_filepaths(y_date, gage)

    def test_filepath_generator(self):
        for gage, y_date in self.test_pairs:
            yield self.get_required_filepaths(y_date, gage)

    def train_data_generator(self):
        for gage, y_date in self.train_pairs:
            fps = self.get_required_filepaths(y_date, gage)
            try:
                yield self.filepaths_to_data(fps)
            except OSError:
                print(f"OSError in train_data_generator: gage={gage}, y_date={y_date}")
                continue

    def val_data_generator(self):
        for gage, y_date in self.val_pairs:
            fps = self.get_required_filepaths(y_date, gage)
            try:
                yield self.filepaths_to_data(fps)
            except OSError:
                print(f"OSError in val_data_generator: gage={gage}, y_date={y_date}")
                continue

    def test_data_generator(self):
        for gage, y_date in self.test_pairs:
            fps = self.get_required_filepaths(y_date, gage)
            try:
                yield self.filepaths_to_data(fps)
            except OSError:
                print(f"OSError in test_data_generator: gage={gage}, y_date={y_date}")
                continue
