"""Client to download GEE images and save to G:Drive."""

import datetime
from typing import Tuple

import ee
import pandas as pd
from shapely import geometry
from tqdm import tqdm

from google_helper_functions import connect_to_service_account_gdrive
from utils import get_polygon, remove_punctuation

DATETIME_FORMAT = "%Y_%m_%d__%H_%M_%S"
DATE_FORMAT = "%Y_%m_%d"


class GEEClient:

    def __init__(self, service_account: str, key_fp: str):
        self.service_account = service_account
        self.key_fp = key_fp
        self.drive = connect_to_service_account_gdrive(key_fp)
        credentials = ee.ServiceAccountCredentials(service_account, key_fp)
        ee.Initialize(credentials)
        self._tasks, self._filenames = list(), list()

    @staticmethod
    def get_img_collection(sat_name: str, polygon: geometry.Polygon,
                           date_from: str, date_to: str) -> ee.ImageCollection:
        """Get ee.ImageCollection for a satellite.

        Args:
            sat_name: name of satellite in GEE datasets catalog.
            polygon: geographical region to get data for.
            date_from: start date for data.
            date_to: end date for data.

        Returns:
            ee.ImageCollection
        """
        gee_polygon = ee.Geometry.Polygon(list(polygon.boundary.coords))
        date_from = datetime.datetime.strptime(date_from, DATE_FORMAT)
        date_to = datetime.datetime.strptime(date_to, DATE_FORMAT)
        return ee.ImageCollection(sat_name).filterDate(date_from, date_to).filterBounds(gee_polygon)

    def export_ic_to_gdrive(self, bounding_box: Tuple, sat_name: str,
                            band: str, date_from: str, date_to: str,
                            buffer_percent: float = 0.05,
                            crs: str = "epsg:4326",
                            scale: float = 100) -> None:
        """Export all files in an Image Collection to GDrive.

        Args:
            bounding_box: lat-lon coordinates defining the bounding box edges.
            sat_name: satellite name in GEE catalog.
            band: satellite band to get images from.
            date_from: start date to get images from.
            date_to: end date to get images to.
            buffer_percent: percent of bounding box area to add as buffer.
            crs: coordinate reference system.
            scale: desired image resolution.
        """
        # Define the region of interest:
        left, bottom, right, top = bounding_box  # NOQA
        polygon = get_polygon(left, bottom, right, top, buffer_percent=buffer_percent)

        # Get the GEE image collection:
        ic = self.get_img_collection(sat_name=sat_name, polygon=polygon, date_from=date_from, date_to=date_to)

        # Number of images to download from the image collection:
        num_img = ic.size().getInfo()  # NOQA
        print(f"Satellite `{sat_name}`, band `{band}`: creating export tasks for {num_img:,.0f} images")

        # Convert to a list of images:
        img_list = ic.toList(num_img)  # NOQA

        # Export the images one by one:
        tasks, filenames = list(), list()
        for i in tqdm(range(num_img)):

            # Select the image:
            img = ee.Image(img_list.get(i))

            # Date of the image:
            img_date = img.date().getInfo()  # NOQA
            hdate = datetime.datetime.utcfromtimestamp(img_date["value"] / 1000).strftime(DATE_FORMAT)

            # Select a band:
            img_band = img.select(band)

            # Get the GEE polygon shape
            gee_polygon = ee.Geometry.Polygon(list(polygon.boundary.coords))

            # Reproject:
            reprojection = img_band.reproject(crs=crs, scale=scale)

            # Export to G:Drive:
            sat_name_c = remove_punctuation(sat_name)
            crs_c = remove_punctuation(crs)
            band_c = remove_punctuation(band)
            hdate_c = remove_punctuation(hdate)
            filename = f"{crs_c}__{scale}__{sat_name_c}__{band_c}__{hdate_c}"
            task = ee.batch.Export.image.toDrive(
                reprojection.toFloat(),
                description=f"Image {filename}",
                folder=f"{sat_name_c}",
                fileNamePrefix=filename,
                region=gee_polygon,
                fileFormat="GeoTIFF",
                maxPixels=1e10
            )
            task.start()

            tasks.append(task)
            filenames.append(filename)

        self._tasks += tasks
        self._filenames += filenames

    @property
    def tasks(self):
        task_df = pd.DataFrame([t.status() for t in self._tasks])
        task_df["filename"] = self._filenames
        return task_df

    @property
    def incomplete_tasks(self):
        task_df = self.tasks
        task_df = task_df[task_df["state"] != "COMPLETED"]
        return task_df

    @property
    def active(self):
        return bool(len(self.incomplete_tasks))
