
import os
from PIL import Image  # NOQA
import re

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show

DIR, FILENAME = os.path.split(__file__)
DATA_DIR = os.path.join(os.path.dirname(DIR), "data")
MASK_DIR = os.path.join(DATA_DIR, "masks")
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)
BASIN_DIR = os.path.join(DATA_DIR, "usgs_basins")
if not os.path.exists(BASIN_DIR):
    os.mkdir(BASIN_DIR)


class TifFile:

    def __init__(self, fp: str):
        self.fp = fp
        self.filename = os.path.basename(fp)
        with rasterio.open(fp) as f:
            self.tif_data = f
        with Image.open(self.fp) as f:
            self.as_numpy = np.array(f)
        # Version of the array with all NaNs filled with zero:
        self.as_numpy_zero_nan = np.where(np.isnan(self.as_numpy), 0, self.as_numpy)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        show(self.tif_data, with_bounds=True, ax=ax)

    @property
    def gage(self):
        """Returns 8-digit gage number if found in filename, else None."""
        streamgage = list(set(re.findall("\d{8}", self.filename)))
        if len(streamgage) == 1:
            return str(streamgage[0])
        else:
            return None

    @property
    def band(self):
        """Returns band name if found in filename, else None."""
        found = list()
        for band in ("et", "precip", "temp", "swe", "dem"):
            if band in self.filename.lower():
                found.append(band)
        if len(found) == 1:
            return found[0]
        else:
            return None

    @property
    def shape(self):
        return self.as_numpy.shape

    @property
    def num_rows(self):
        """Number of pixel rows in the image, i.e. magnitude of y-axis."""
        return self.shape[0]

    @property
    def num_columns(self):
        """Number of pixel columns in the image, i.e. magnitude of x-axis."""
        return self.shape[1]

    @property
    def pixel_width_lon(self):
        """Width of a pixel in units of longitude."""
        return (self.bounds.right - self.bounds.left) / self.num_columns

    @property
    def pixel_height_lat(self):
        """Height of a pixel in units of latitude."""
        return (self.bounds.top - self.bounds.bottom) / self.num_rows

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

    def pixel_shape(self, row: int, col: int):
        """Get Polygon of the shape of an individual pixel in the image.

        Args:
            row: vertical pixel index of the row.
            col: horizontal pixel index of the column.
        """
        from utils import get_polygon

        assert 0 <= col <= self.num_columns
        assert 0 <= row <= self.num_rows

        left = self.min_lon + (self.pixel_width_lon * col)
        right = self.min_lon + (self.pixel_width_lon * (col+1))
        bottom = self.min_lat + (self.pixel_height_lat * row)
        top = self.min_lat + (self.pixel_height_lat * (row+1))

        return get_polygon(left, bottom, right, top, buffer_percent=0)

    @property
    def as_polygon(self):
        """Return the full area of the image as a Polygon."""
        from utils import get_polygon
        return get_polygon(self.bounds.left, self.bounds.bottom, self.bounds.right, self.bounds.top, buffer_percent=0)

    def calculate_mask(self, geojson_fp: str):
        """Calculate the intersection mask of this image's pixels with a GeoJson
        file of an irregular geometry (i.e. a watershed area)."""
        gdf = gpd.read_file(geojson_fp)
        basin_geo = gdf[gdf["id"] == "globalwatershed"]["geometry"].iloc[0]
        ratio_list = list()
        for row in list(range(self.num_rows))[::-1]:  # Iterate rows backwards so they are appended top to bottom.
            row_values = list()
            for col in range(self.num_columns):
                pxl = self.pixel_shape(row, col)
                intersection = pxl.intersection(basin_geo).area
                ratio = intersection / pxl.area
                row_values.append(ratio)
            ratio_list.append(row_values)
        mask = np.array(ratio_list)
        return mask

    def get_mask(self, gage: str, band: str):
        mask = self.load_mask(gage, band)
        if mask is None:
            geojson_fp = os.path.join(BASIN_DIR, f"{gage}.geojson")
            assert os.path.exists(geojson_fp), f"USGS basin file not found: {geojson_fp}"
            mask = self.calculate_mask(geojson_fp)
            self.save_mask(mask, gage, band)
        return mask

    def plot_mask(self, gage: str, band: str, suptitle: str = None):
        """Plot the tif's image masked by an irregular geometry from a GeoJSON
        file (i.e. a watershed area)."""
        mask = self.get_mask(gage, band)

        fig, axes = plt.subplots(1, 5, figsize=(13, 3))

        vmax = self.pixel_nanmax

        # Original image:
        ax = axes[0]
        self.plot(ax)
        ax.imshow(self.as_numpy, vmin=0, vmax=vmax)
        ax.set_title("Original Image")

        # Ratio mask:
        ax = axes[1]
        ax.imshow(mask, vmin=0, vmax=1)
        ax.set_title("Ratio Mask")

        # Ratio masked image:
        ratio_masked_img = np.where(mask == 0, 0, self.as_numpy * mask)
        ax = axes[2]
        ax.imshow(ratio_masked_img, vmin=0, vmax=vmax)
        ax.set_title("Ratio Masked Image")

        # Boolean mask:
        ax = axes[3]
        bool_mask = np.where(mask, 1, 0)
        ax.imshow(bool_mask, vmin=0, vmax=1)
        ax.set_title("Bool Mask")

        # Boolean masked image:
        ax = axes[4]
        bool_masked_img = np.where(bool_mask, self.as_numpy, 0)
        ax.imshow(bool_masked_img, vmin=0, vmax=vmax)
        ax.set_title("Bool Masked Image")

        for ax in axes.flatten()[1:]:
            ax.set_xticks([])
            ax.set_yticks([])

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig

    @staticmethod
    def save_mask(mask: np.array, gage: str, band: str):
        """Save image mask as a numpy.array"""
        h, w = mask.shape
        mask_name = f"{gage}__{band.lower()}__h{h}_w{w}.npy"
        fp = os.path.join(MASK_DIR, mask_name)
        np.save(fp, mask, allow_pickle=False)
        return fp

    def load_mask(self, gage: str, band: str):
        """Attempt to load a saved mask matching the current image's shape.
        Returns None if mask doesn't exist."""
        from utils import open_npy_file
        h, w = self.shape
        mask_name = f"{gage}__{band.lower()}__h{h}_w{w}.npy"
        fp = os.path.join(MASK_DIR, mask_name)
        try:
            return open_npy_file(fp)
        except FileNotFoundError:
            return None


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
