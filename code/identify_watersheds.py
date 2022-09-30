"""Method to identify the upstream watershed from a given lat-long point."""

import datetime
import os
import time

import ee
import folium
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import pickle
from pysheds.grid import Grid

from google_helper_functions import authenticate_gee, connect_to_service_account_gdrive, download_files_from_gdrive

DATETIME_FORMAT = "%Y_%m_%d__%H_%M_%S"

# Get the directory location:
DIR, FILENAME = os.path.split(__file__)
DATA_DIR = os.path.join(os.path.dirname(DIR), "data")
TARGET_GAGES = pd.read_csv(os.path.join(DATA_DIR, "target_gages.csv"), encoding="utf-8")


def add_ee_layer(self, ee_image_object, vis_params, name):
    """Adds a method for displaying Earth Engine image tiles to folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)


def map_elevation_point(lat: float, lon: float, name: str, buffer: int = 10_000,
                        zoom_start: int = 7, width: object = "60%",
                        height: object = "60%"):
    """Helper function to produce a folium map showing the point of interest on
    an elevation map, with a circle drawn around it of radius=`buffer`.

    Args:
        lat: decimal latitude.
        lon: decimal longitude.
        name: label to add to marker on map.
        buffer: radius of area in metres to draw around lat-long point.
        zoom_start: folium map zoom.
        width: folium map width.
        height: folium map height.

    Returns:
        folium.folium.Map
    """
    # Import the USGS ground elevation image.
    elv = ee.Image("USGS/SRTMGL1_003")

    # Add Earth Engine drawing method to folium.
    folium.Map.add_ee_layer = add_ee_layer

    # Set visualization parameters for ground elevation.
    elv_vis_params = {
        'min': 0, 'max': 4000,
        'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']
    }

    # Create a new map.
    my_map = folium.Map(location=[lat, lon], zoom_start=zoom_start, width=width, height=height)

    my_map.add_ee_layer(elv, elv_vis_params, "Elevation")  # NOQA

    # Define a region of interest with a buffer zone:
    poi = ee.Geometry.Point(lon, lat)
    roi = poi.buffer(buffer)

    # Draw the marker:
    folium.Marker(location=[lat, lon], radius=10, popup=f"{name} ({lat}, {lon})").add_to(my_map)

    # Draw the buffer zone:
    folium.GeoJson(roi.getInfo()).add_to(my_map)

    return my_map


def save_gee_elv_to_drive(lat: float, lon: float, name: str,
                          buffer: int = 10_000, scale: int = 30):
    """

    Args:
        lat: decimal latitude.
        lon: decimal longitude.
        name: location identifier e.g. streamgage name.
        buffer: radius around point of interest to get.
        scale: pixel resolution in metres of image to save.

    Returns:
        ee.batch.Task, str
    """
    # Credit to this SO answer:
    # https://stackoverflow.com/questions/71834208/why-is-the-export-to-the-drive-empty-when-using-google-earth-engine-in-python-w
    # TODO: This can be factored out to a general function for scraping other GEE satellite datasets.
    # Import the USGS ground elevation image.
    elv = ee.Image("USGS/SRTMGL1_003")

    # Define a region of interest as a circular buffer zone around a point:
    poi = ee.Geometry.Point(lon, lat)
    roi = poi.buffer(buffer)

    description = f"Elevation data export for streamgage {name}"
    now = datetime.datetime.now().strftime(DATETIME_FORMAT)
    filename = f"{name}_{scale:.0f}m_elv_{now}"

    # Export the geotiff to GDrive for the elevation image in the highlighted polygon:
    task = ee.batch.Export.image.toDrive(
        image=elv,
        description=description,
        scale=scale,
        region=roi,
        fileNamePrefix=filename,
        crs="EPSG:4326",
        fileFormat="GeoTIFF"
    )
    task.start()
    return task, filename


class WatershedIdentifier:
    """Use pyshed to preprocess elevation data and identify a watershed."""

    def __init__(self, tif_fp: str, lat: float, lon: float,
                 min_acc: float = 1.0):
        self.tif_fp = tif_fp
        self.grid = Grid.from_raster(tif_fp)
        self.dem = self.grid.read_raster(tif_fp)

        # Condition DEM
        # ----------------------
        # Fill pits in DEM:
        self.pit_filled_dem = self.grid.fill_pits(self.dem)

        # Fill depressions in DEM:
        self.flooded_dem = self.grid.fill_depressions(self.pit_filled_dem)

        # Resolve flats in DEM:
        self.inflated_dem = self.grid.resolve_flats(self.flooded_dem)

        # Determine D8 flow directions from DEM
        # Specify directional mapping:
        self.dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

        # Compute flow directions:
        self.fdir = self.grid.flowdir(self.inflated_dem, dirmap=self.dirmap)

        # Calculate flow accumulation:
        self.acc = self.grid.accumulation(self.fdir, dirmap=self.dirmap)

        # Delineate a catchment
        self.calculate_catchment(lat, lon, min_acc)

    def calculate_catchment(self, lat: float, lon: float, min_acc: float = 1.0):
        self.lat, self.lon, self.min_acc = lat, lon, min_acc  # NOQA

        # Move the lat-lon to a higher pour point if necessary:
        self.snap_lon, self.snap_lat = self.snap_lat_lon_on_acc(lat, lon, min_acc)  # NOQA

        # Delineate the catchment:
        self.catch = self.grid.catchment(x=self.snap_lon, y=self.snap_lat, fdir=self.fdir, # NOQA
                                         dirmap=self.dirmap, xytype="coordinate")

        # Create clipped version:
        self.clipped_grid = Grid.from_raster(self.tif_fp)  # NOQA
        self.clipped_grid.clip_to(self.catch)
        self.clipped_catch = self.clipped_grid.view(self.catch)  # NOQA

    @property
    def catchment_max_width(self):
        return self.clipped_catch.shape[1]

    @property
    def catchment_max_height(self):
        return self.clipped_catch.shape[0]

    @property
    def catchment_total_pixels(self):
        return self.clipped_catch.sum()

    @property
    def catchment_touches_grid_edge(self):
        """Identify if any of the catchment area identified sits on the
        boundaries of the input TIF file (meaning the file may need to be
        enlarged to capture the full watershed)."""
        catchment_arr = np.array(self.catch)
        first_row_sum = catchment_arr[1, :].astype(int).sum()
        last_row_sum = catchment_arr[-2, :].astype(int).sum()
        first_col_sum = catchment_arr[:, 1].astype(int).sum()
        last_col_sum = catchment_arr[:, -2].astype(int).sum()
        return any([first_row_sum, last_row_sum, first_col_sum, last_col_sum])

    def summarize_catchment(self, pixel_m: float = 30.0):
        pixel_km = pixel_m / 1000
        summary = {
            "max_width_km": self.catchment_max_width * pixel_m / 1000,
            "max_height_km": self.catchment_max_height * pixel_m / 1000,
            "total_sq_m": float(self.catchment_total_pixels * (pixel_m * pixel_m)),
            "total_sq_km": float(self.catchment_total_pixels * (pixel_km * pixel_km))
        }
        return summary

    def snap_lat_lon_on_acc(self, lat: float, lon: float,
                            min_acc: float = 1_000.0):
        x, y = lon, lat
        snap_lon, snap_lat = self.grid.snap_to_mask(self.acc > min_acc, (x, y))
        return snap_lon, snap_lat

    def plot_dem(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_alpha(0)
        plt.imshow(self.dem, extent=self.grid.extent, cmap='terrain', zorder=1)
        plt.colorbar(label='Elevation (m)')
        plt.grid(zorder=0)
        plt.title('Digital elevation map', size=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        ax.xaxis.set_major_formatter("{x:.3f}")

        ax.scatter([self.lon], [self.lat], marker="x", color="r", zorder=10, s=100, linewidths=3.5)
        x_snap, y_snap = self.snap_lat_lon_on_acc(self.lat, self.lon, min_acc=self.min_acc)
        ax.scatter([x_snap], [y_snap], marker="+", color="b", zorder=20, s=100, linewidths=3.5)

        return fig

    def plot_flow_dir(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_alpha(0)
        plt.imshow(self.fdir, extent=self.grid.extent, cmap='viridis', zorder=2)
        boundaries = ([0] + sorted(list(self.dirmap)))
        plt.colorbar(boundaries=boundaries, values=sorted(self.dirmap))
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Flow direction grid', size=14)
        plt.grid(zorder=-1)
        plt.tight_layout()
        ax.xaxis.set_major_formatter("{x:.3f}")

        ax.scatter([self.lon], [self.lat], marker="x", color="r", zorder=10, s=100, linewidths=3.5)
        x_snap, y_snap = self.snap_lat_lon_on_acc(self.lat, self.lon, min_acc=self.min_acc)
        ax.scatter([x_snap], [y_snap], marker="+", color="b", zorder=20, s=100, linewidths=3.5)

        return fig

    def plot_flow_acc(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_alpha(0)
        plt.grid('on', zorder=0)
        im = ax.imshow(self.acc, extent=self.grid.extent, zorder=2, cmap='cubehelix',
                       norm=colors.LogNorm(1, self.acc.max()), interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='Upstream Cells')
        plt.title('Flow Accumulation', size=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        ax.xaxis.set_major_formatter("{x:.3f}")

        ax.scatter([self.lon], [self.lat], marker="x", color="r", zorder=10, s=100, linewidths=3.5)
        x_snap, y_snap = self.snap_lat_lon_on_acc(self.lat, self.lon, min_acc=self.min_acc)
        ax.scatter([x_snap], [y_snap], marker="+", color="b", zorder=20, s=100, linewidths=3.5)

        return fig

    def plot_catchment(self, base_map: str = None, clipped: bool = False):

        if clipped:
            extent = self.clipped_grid.extent
            catch = self.clipped_catch
        else:
            extent = self.grid.extent
            catch = self.catch

        # Plot the catchment:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_alpha(0)

        plt.grid('on', zorder=0)
        cmap = "Greys_r"
        title = "Delineated Catchment"
        if base_map == "dem":
            if clipped:
                plt.imshow(self.clipped_grid.view(self.dem), extent=extent, cmap='terrain', zorder=1)
            else:
                plt.imshow(self.dem, extent=extent, cmap='terrain', zorder=1)
            title = f"{title} - {base_map.upper()}"
        elif base_map == "fdir":
            if clipped:
                plt.imshow(self.clipped_grid.view(self.fdir), extent=extent, cmap='viridis', zorder=1)
            else:
                plt.imshow(self.fdir, extent=extent, cmap='viridis', zorder=2)
            cmap = "YlGn"
            title = f"{title} - {base_map.upper()}"
        elif base_map == "acc":
            if clipped:
                plt.imshow(self.clipped_grid.view(self.acc), extent=extent, cmap='cubehelix', zorder=1)
            else:
                ax.imshow(self.acc, extent=extent, zorder=2, cmap='cubehelix',
                          norm=colors.LogNorm(1, self.acc.max()), interpolation='bilinear')
            cmap = "YlGn"
            title = f"{title} - {base_map.upper()}"

        ax.imshow(np.where(catch, catch, np.nan), extent=extent, zorder=2, cmap=cmap, alpha=0.5)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(title, size=14)
        ax.scatter([self.lon], [self.lat], marker="x", color="r", zorder=10, s=50, linewidths=1.5)
        x_snap, y_snap = self.snap_lat_lon_on_acc(self.lat, self.lon, min_acc=self.min_acc)
        ax.scatter([x_snap], [y_snap], marker="+", color="b", zorder=20, s=50, linewidths=1.5)
        return fig

    def save(self, directory: str, name: str):
        """Pickle the current instance."""
        now = datetime.datetime.now().strftime(DATETIME_FORMAT)
        fp = os.path.join(directory, f"{name}__{now}")
        with open(fp, "wb") as f:
            pickle.dump(self, f)
        return fp


def clear_up_gage_tiff_files():
    """Delete all but the most recent TIFF elevation file for each gage."""
    tif_dir = os.path.join(DATA_DIR, "gage_tiff_files")
    files = [f for f in os.listdir(tif_dir) if f.endswith(".tif")]
    gage_ids = [f.split("_")[0] for f in files]
    timestamps = [f"2022" + f.split("2022")[-1][:-4] for f in files]
    timestamps = [datetime.datetime.strptime(ts, DATETIME_FORMAT) for ts in timestamps]
    df = pd.DataFrame({"file": files, "gage_id": gage_ids, "timestamp": timestamps})
    df = df.sort_values(by=["gage_id", "timestamp"], ascending=False)
    df["keep"] = df["gage_id"] != df["gage_id"].shift(-1)
    delete = df[~df["keep"]]
    for filename in delete["file"]:
        fp = os.path.join(tif_dir, filename)
        os.remove(fp)
    return delete


def clear_up_watershed_files():
    """Delete all but the most recent watershed file for each gage."""
    ws_dir = os.path.join(DATA_DIR, "watersheds")
    files = [f for f in os.listdir(ws_dir) if "__" in f]
    gage_ids = [f.split("_")[0] for f in files]
    timestamps = [f"2022" + f.split("2022")[-1] for f in files]
    timestamps = [datetime.datetime.strptime(ts, DATETIME_FORMAT) for ts in timestamps]
    df = pd.DataFrame({"file": files, "gage_id": gage_ids, "timestamp": timestamps})
    df = df.sort_values(by=["gage_id", "timestamp"], ascending=False)
    df["keep"] = df["gage_id"] != df["gage_id"].shift(-1)
    delete = df[~df["keep"]]
    for filename in delete["file"]:
        fp = os.path.join(ws_dir, filename)
        os.remove(fp)
    return delete


if __name__ == "__main__":

    # Authenticate with service account:
    service_acct = "capstone-gee-account@snow-capstone.iam.gserviceaccount.com"
    keys = os.path.join(os.path.expanduser("~"), "snow-capstone-4a3c9603fcf0.json")
    credentials = authenticate_gee(service_acct, keys)

    gages = pd.read_csv(os.path.join(DATA_DIR, "target_gages.csv"), encoding="utf-8")
    for ix, row in gages.iterrows():

        lat, lon = row["dec_lat_va"], row["dec_long_va"]
        name = row["site_no"]
        print(f"Streamgage: {name}")

        # Radius in metres of the area around the streamgage to get data for.
        # Start small and increase by 5k each time until full watershed is found:
        buffer = 10_000
        catchment_found = False

        while not catchment_found:

            # Export the geotiff to GDrive for the elevation image in the highlighted polygon:
            task, filename = save_gee_elv_to_drive(lat, lon, name=name, buffer=buffer, scale=30)
            while task.status()["state"] != "COMPLETED":
                time.sleep(1)
            print("TIF file saved to service account G:Drive")

            # Download the file from G:Drive to a local copy:
            drive = connect_to_service_account_gdrive(keys)
            save_to = os.path.join(DATA_DIR, "gage_tiff_files")
            local_files = download_files_from_gdrive(drive, filename, save_to, download_all=True)
            tif_fp = local_files[0]

            # Delineate the watershed and save it:
            # This variable moves the streamgage location to a higher accumulation point:
            min_acc = 1
            ws = WatershedIdentifier(tif_fp, lat, lon, min_acc)
            print(ws.summarize_catchment(30))

            # Save the watershed and confirm it can be reloaded:
            pickled_fp = ws.save(os.path.join(DATA_DIR, "watersheds"), f"{name}")


            def pickle_load(fp: str):
                with open(fp, "rb") as f:
                    return pickle.load(f)


            unpickled_ws = pickle_load(pickled_fp)
            unpickled_ws.summarize_catchment()

            catchment_found = not ws.catchment_touches_grid_edge
            buffer += 5_000
