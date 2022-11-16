"""Create a GIF showing the changes in X and y variables over time."""

import datetime
import os

import imageio
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from tif_files import TifFile
from utils import extract_filename_data, open_npy_file, open_y_data

DATE_FORMAT = "%Y_%m_%d"


class Giffer:

    def __init__(self, src_dir: str, target_dir: str, dem_dir: str):
        self.target_directory = target_dir
        self.src_directory = src_dir
        files = os.listdir(self.src_directory)
        self.files = list(filter(lambda f: f.endswith(".tif") or f.endswith(".npy"), files))
        file_data = [extract_filename_data(fn) for fn in self.files]
        self.file_df = pd.DataFrame(file_data)
        self.bands = ['total_precipitation', 'et', 'swe', 'temperature_2m']
        self.filenames = {band: None for band in self.bands}
        self.min_date, self.max_date = min(self.file_df["date"]), max(self.file_df["date"])
        self.date_range = pd.date_range(self.min_date, self.max_date)
        self.y = open_y_data()
        self.y_max = self.y.groupby(["gage"])["m3"].max()

        # Load the DEMs:
        self.dem_dir = dem_dir
        dem_files = os.listdir(self.dem_dir)
        dem_files = list(filter(lambda f: f.endswith(".tif"), dem_files))
        self.dem_fps = {extract_filename_data(f)["streamgage"]: os.path.join(dem_dir, f) for f in dem_files}
        self.dems = dict()
        for k, v in self.dem_fps.items():
            tif = TifFile(v)
            self.dems[k] = tif.as_numpy_zero_nan

    def create_frame(self, gage: str, date: datetime.datetime):
        df = self.file_df
        df = df[(df["date"] == date) & (df["streamgage"] == gage)]
        fig, axes = plt.subplots(1, 6, figsize=(12, 3))

        for band in self.bands:
            if band in df["band"].values:
                fn = df[df["band"] == band]["fn"].iloc[0]
                self.filenames[band] = fn

        for ax, (band, filename) in zip(axes.flatten()[1:], self.filenames.items()):
            if filename is not None:
                fp = os.path.join(self.src_directory, filename)
                if filename.endswith("npy"):
                    data = open_npy_file(fp)
                elif filename.endswith("tif"):
                    data = TifFile(fp).as_numpy_zero_nan
                else:
                    raise ValueError(filename)
                ax.imshow(data)

            ax.set_title(band)
            ax.set_xticks([])
            ax.set_yticks([])

        # Plot the DEM:
        ax = axes.flatten()[0]
        ax.imshow(self.dems[gage])
        ax.set_title("DEM")
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot the y-variable:
        ax = axes.flatten()[-1]
        y = self.y[(self.y["gage"] == gage) & (self.y["date"] == date)]["m3"].iloc[0]
        ax.bar(0, y)
        ax.set_ylim(0, self.y_max[gage])
        ax.set_xticks([])
        ax.set_title("discharge m3")
        ax.yaxis.tick_right()

        img_fp = os.path.join(self.target_directory, f"{gage}_{date.strftime(DATE_FORMAT)}.png")
        fig.suptitle(f"{gage} - {date.strftime('%d %b %Y')}", y=1.05)
        fig.savefig(img_fp, bbox_inches="tight")
        plt.close()
        return img_fp

    def create_gif(self, gage: str, fps: int = 25):
        png_fps, frames = list(), list()
        print(f"Creating {fps} fps GIF for gage {gage}")
        for date in tqdm(self.date_range):
            img_fp = self.create_frame(gage, date)
            png_fps.append(img_fp)
            image = imageio.v2.imread(img_fp)
            frames.append(image)
        gif_fp = os.path.abspath(os.path.join(self.target_directory, f"{gage}.gif"))
        imageio.mimsave(gif_fp, frames, fps=fps)
        print(gif_fp)
        # Clean up png files:
        for fp in png_fps:
            os.remove(fp)
        return gif_fp


if __name__ == "__main__":
    DIR, FILENAME = os.path.split(__file__)
    DATA_DIR = os.path.join(os.path.dirname(DIR), "data")
    TRAIN_DIR = os.path.join(DATA_DIR, "training_data")
    IMG_DIR = os.path.join(DATA_DIR, "sat_img_viz")
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)
    DEM_DIR = os.path.join(DATA_DIR, "dem")
    giffer = Giffer(src_dir=TRAIN_DIR, target_dir=IMG_DIR, dem_dir=DEM_DIR)
    gages = ["11185500", "11189500", "11202710", "11208000", "11266500", "11318500", "11402000"]
    for g in gages:
        gage_gif_fp = giffer.create_gif(g)
