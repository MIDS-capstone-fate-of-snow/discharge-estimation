
import os
from PIL import Image  # NOQA

import folium
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
        """Mean of all non-nan pixel values."""
        return np.nanmean(self.as_numpy)

    @property
    def pixel_sum(self):
        """Sum of all pixel values."""
        return self.as_numpy.sum()

    @property
    def pixel_nansum(self):
        """Sum of all non-nan pixel values."""
        return np.nansum(self.as_numpy)

    @property
    def pixel_min(self):
        """Min of all pixel values."""
        return self.as_numpy.min()

    @property
    def pixel_nanmin(self):
        """Min of all non-nan pixel values."""
        return np.nanmin(self.as_numpy)

    @property
    def pixel_max(self):
        """Max of all pixel values."""
        return self.as_numpy.max()

    @property
    def pixel_nanmax(self):
        """Max of all non-nan pixel values."""
        return np.nanmax(self.as_numpy)

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

    def plot_pixel_grid(self, my_map: folium.folium.Map = None,
                        color: str = "blue", weight: int = 1,
                        fill_opacity: float = 0.0, **kwargs):
        """Plot a grid of the pixel locations on a folium map."""
        if my_map is None:
            lat = self.min_lat + ((self.max_lat - self.min_lat) / 2)
            lon = self.min_lon + ((self.max_lon - self.min_lon) / 2)
            my_map = folium.Map(location=[lat, lon], **kwargs)
        width = self.max_lon - self.min_lon
        height = self.max_lat - self.min_lat
        cell_w = width / self.shape[1]
        cell_h = height / self.shape[0]

        for w in range(self.shape[0]):
            for h in range(self.shape[1]):
                min_lat = self.min_lat + (w * cell_w)
                max_lat = self.min_lat + ((w+1) * cell_w)
                min_lon = self.min_lon + (h * cell_h)
                max_lon = self.min_lon + ((h+1) * cell_h)
                folium.Rectangle([(min_lat, min_lon), (max_lat, max_lon)],
                                 color=color, weight=weight, fill=True, fill_opacity=fill_opacity).add_to(my_map)
        return my_map


class TifDir:

    def __init__(self, fp: str):
        self.__ix = 0  # Index for generator.
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

    def __next__(self):
        v = self.tifs[self.__ix]
        self.__ix += 1
        return v

    def __iter__(self):
        for tif in self.tifs:
            yield tif

    def __len__(self):
        return len(self.tif_fps)
