
import os
from PIL import Image  # NOQA

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show


class TifFile:

    def __init__(self, fp: str):
        self.fp = fp
        self.tif_data = rasterio.open(fp)
        self.as_numpy = np.array(Image.open(self.fp))

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        tif_data = rasterio.open(self.fp)
        show(tif_data, with_bounds=True, ax=ax)

    @property
    def shape(self):
        return self.as_numpy.shape

    @property
    def num_pixels(self):
        return self.shape[0] * self.shape[1]

    @property
    def num_notna_pixels(self):
        return (~np.isnan(self.as_numpy)).astype(int).sum()

    @property
    def contain_nans(self):
        return self.num_pixels != self.num_notna_pixels

    @property
    def pixel_mean(self):
        """Mean of all pixel values."""
        return self.as_numpy.mean()

    @property
    def pixel_nanmean(self):
        return np.nanmean(self.as_numpy)

    @property
    def pixel_sum(self):
        """Mean of all pixel values."""
        return self.as_numpy.sum()

    @property
    def pixel_nansum(self):
        return np.nansum(self.as_numpy)

    @property
    def crs(self):
        """Coordinate reference system."""
        return self.tif_data.crs["init"]

    @property
    def bounds(self):
        """Image lat-lon bounds."""
        return self.tif_data.bounds

    @property
    def min_lon(self):
        """Image minimum lon."""
        return self.tif_data.bounds[0]

    @property
    def min_lat(self):
        """Image minimum lat."""
        return self.tif_data.bounds[1]

    @property
    def max_lon(self):
        """Image maximum lon."""
        return self.tif_data.bounds[2]

    @property
    def max_lat(self):
        """Image maximum lat."""
        return self.tif_data.bounds[3]


class TifDir:

    def __init__(self, fp: str):
        assert os.path.isdir(fp), f"Not a directory: {fp}"
        self.fp = fp
        self.tif_fps = [os.path.join(self.fp, f) for f in os.listdir(self.fp) if f.endswith(".tif")]
        self.tifs = [TifFile(p) for p in self.tif_fps]
        self.count = len(self.tifs)

    @property
    def shapes_match(self):
        return len(set([t.shape for t in self.tifs])) == 1

    @property
    def contain_nans(self):
        return any([t.contain_nans for t in self.tifs])

    def plot(self, num_col: int = 4, fig_width: float = 8.0):
        num_row = self.count // num_col
        if (num_row * num_col) < self.count:
            num_row += 1
        fig_height = fig_width * (num_col / num_row)
        fig, axes = plt.subplots(num_row, num_col, figsize=(fig_width, fig_height))
        for t, ax in zip(self.tifs, axes.flatten()):
            t.plot(ax=ax)
        return fig

    @property
    def elementwise_mean(self):
        return np.mean([t.as_numpy for t in self.tifs], axis=0)

    @property
    def mean(self):
        return np.mean([t.as_numpy for t in self.tifs])

    @property
    def elementwise_max(self):
        return np.max([t.as_numpy for t in self.tifs], axis=0)

    @property
    def max(self):
        return np.max([t.as_numpy for t in self.tifs])

    @property
    def elementwise_min(self):
        return np.min([t.as_numpy for t in self.tifs], axis=0)

    @property
    def min(self):
        return np.min([t.as_numpy for t in self.tifs])

    @property
    def elementwise_sum(self):
        return np.sum([t.as_numpy for t in self.tifs], axis=0)

    @property
    def sum(self):
        return np.sum([t.as_numpy for t in self.tifs])

    def __getitem__(self, item):
        return self.tifs[item]
