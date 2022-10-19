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
                           date_from: str, date_to: str,
                           band: str = None) -> ee.ImageCollection:
        """Get ee.ImageCollection for a satellite.

        Args:
            sat_name: name of satellite in GEE datasets catalog.
            polygon: geographical region to get data for.
            date_from: start date for data.
            date_to: end date for data.
            band: optional satellite band to select.

        Returns:
            ee.ImageCollection
        """
        gee_polygon = ee.Geometry.Polygon(list(polygon.boundary.coords))
        date_from = datetime.datetime.strptime(date_from, DATE_FORMAT)
        date_to = datetime.datetime.strptime(date_to, DATE_FORMAT)
        ic = ee.ImageCollection(sat_name).filterDate(date_from, date_to).filterBounds(gee_polygon)
        if band is not None:
            ic = ic.select(band)
        return ic

    def export_ic_to_gdrive(self, bounding_box: Tuple, sat_name: str,
                            band: str, date_from: str, date_to: str,
                            buffer_percent: float = 0.05,
                            crs: str = "epsg:4326",
                            scale: float = None, hourly: bool = False) -> None:
        """Export all files in an Image Collection to GDrive.

        Args:
            bounding_box: lat-lon coordinates defining the bounding box edges.
            sat_name: satellite name in GEE catalog.
            band: satellite band to get images from.
            date_from: start date to get images from.
            date_to: end date to get images to.
            buffer_percent: percent of bounding box area to add as buffer.
            crs: coordinate reference system.
            scale: optional desired image resolution, otherwise default is used.
            hourly: if True, append the hour to the datetime for hourly data.
        """
        # Define the region of interest:
        left, bottom, right, top = bounding_box  # NOQA
        polygon = get_polygon(left, bottom, right, top, buffer_percent=buffer_percent)
        gee_polygon = ee.Geometry.Polygon(list(polygon.boundary.coords))

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

            # Get the original scale if not rescaling:
            if scale is None:
                scale = img_band.projection().nominalScale().getInfo()
            assert isinstance(scale, float)

            # Reproject:
            reprojection = img_band.reproject(crs=crs, scale=scale)
            if hourly:
                hour = reprojection.getInfo()["properties"]["hour"]
                hdate = hdate + "_" + f"{hour}".zfill(2)

            # Export to G:Drive:
            sat_name_c = remove_punctuation(sat_name)
            crs_c = remove_punctuation(crs)
            band_c = remove_punctuation(band)
            hdate_c = remove_punctuation(hdate)
            scale_c = remove_punctuation(f"{scale:.2f}")
            filename = f"{crs_c}__{scale_c}__{sat_name_c}__{band_c}__{hdate_c}"
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

    def export_hourly_to_daily(self, bounding_box: Tuple, sat_name: str,
                               band: str, agg: str, date_from: str,
                               date_to: str,  buffer_percent: float = 0.05,
                               crs: str = "epsg:4326",
                               scale: float = None) -> None:
        """Alternate export that handles transforming hourly data to daily.

        Args:
            bounding_box: lat-lon coordinates defining the bounding box edges.
            sat_name: satellite name in GEE catalog.
            band: satellite band to get images from.
            agg: hourly aggregation to daily - either 'sum' or 'mean'.
            date_from: start date to get images from.
            date_to: end date to get images to.
            buffer_percent: percent of bounding box area to add as buffer.
            crs: coordinate reference system.
            scale: optional desired image resolution, otherwise default is used.
        """
        # TODO! Factor out this function and the other export function so that there is less duplication.
        # Define the region of interest:
        left, bottom, right, top = bounding_box  # NOQA
        polygon = get_polygon(left, bottom, right, top, buffer_percent=buffer_percent)

        # Get the GEE image collection:
        ic = self.get_img_collection(sat_name=sat_name, polygon=polygon, date_from=date_from, date_to=date_to, band=band)

        # Convert hourly data to daily:
        start_date = datetime.datetime.strptime(date_from, DATE_FORMAT).strftime("%Y-%m-%d")
        end_date = datetime.datetime.strptime(date_to, DATE_FORMAT).strftime("%Y-%m-%d")
        ic = self.hourly_to_daily(ic, start_date=start_date, end_date=end_date, agg=agg)

        # Number of images to download from the image collection:
        num_img = ic.size().getInfo()  # NOQA
        print(f"Satellite `{sat_name}`, band `{band}`: creating export tasks for {num_img:,.0f} images")

        # Convert to a list of images:
        img_list = ic.toList(num_img)  # NOQA

        # Export the images one by one:
        tasks, filenames = list(), list()
        img_date = datetime.datetime.strptime(date_from, DATE_FORMAT)
        for i in tqdm(range(num_img)):

            # Select the image:
            img = ee.Image(img_list.get(i))

            # Date of the image:
            hdate = img_date.strftime(DATE_FORMAT)
            img_date += datetime.timedelta(days=1)

            # Get the original scale if not rescaling:
            if scale is None:
                scale = img.projection().nominalScale().getInfo()  # NOQA
            assert isinstance(scale, float)

            # Get the GEE polygon shape
            gee_polygon = ee.Geometry.Polygon(list(polygon.boundary.coords))

            # Reproject CRS and scale:
            reprojection = img.reproject(crs=crs, scale=scale)  # NOQA

            # Export to G:Drive:
            sat_name_c = remove_punctuation(sat_name)
            crs_c = remove_punctuation(crs)
            band_c = remove_punctuation(band)
            hdate_c = remove_punctuation(hdate)
            scale_c = remove_punctuation(f"{scale:.2f}")
            filename = f"{crs_c}__{scale_c}__{sat_name_c}__{band_c}__{hdate_c}"
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

    @staticmethod
    def hourly_to_daily(collection: ee.imagecollection.ImageCollection,
                        start_date: str, end_date: str, agg: str = "mean"):
        """Aggregate hourly ImageCollection to daily with sum or mean.

        Args:
            collection: hourly ImageCollection object.
            start_date: string start date in format 'YYY-MM-DD'.
            end_date: string end date in format 'YYY-MM-DD'.
            agg: method for aggregating hourly data - either 'sum' or 'mean'.
        """
        # Credit - adapted from here:
        # https://gis.stackexchange.com/questions/358520/calculating-daily-average-using-hourly-data-in-google-earth-engine
        start_date = ee.Date(start_date)
        end_date = ee.Date(end_date)

        def day_offset(d: int):
            start = start_date.advance(d, "days")  # NOQA
            end = start.advance(1, "days")
            if agg == "mean":
                return collection.filterDate(start, end).mean()
            elif agg == "sum":
                return collection.filterDate(start, end).sum()
            else:
                raise ValueError(f"Invalid agg: {agg}")

        num_days = end_date.difference(start_date, "days")  # NOQA
        daily = ee.ImageCollection(ee.List.sequence(0, num_days.subtract(1)).map(day_offset))  # NOQA
        return daily

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
