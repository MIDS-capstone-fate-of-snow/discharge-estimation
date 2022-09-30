
import datetime
import os
import re

import pandas as pd

DATETIME_FORMAT = "%Y_%m_%d__%H_%M_%S"


def list_streamgage_files(directory: str):
    """List all files in directory with pattern `{streamgage}_xyx_{datetime}."""
    files = [f for f in os.listdir(directory) if "__" in f]
    files = list(filter(lambda s: len(re.findall("\d{8}_", s)) > 0, files))
    gage_ids = [re.findall("\d{8}_", f)[0][:-1] for f in files]
    timestamps = [f"2022" + f.split("2022")[-1] for f in files]
    timestamps = [ts.split(".")[0] for ts in timestamps]
    timestamps = [datetime.datetime.strptime(ts, DATETIME_FORMAT) for ts in timestamps]
    df = pd.DataFrame({"file": files, "gage_id": gage_ids, "timestamp": timestamps})
    df = df.sort_values(by=["gage_id", "timestamp"], ascending=False)
    df["latest"] = df["gage_id"] != df["gage_id"].shift(1)
    return df


def get_latest_streamgage_file(directory: str, streamgage: str):
    files_df = list_streamgage_files(directory)
    files_df = files_df[(files_df["gage_id"] == str(streamgage)) & (files_df["latest"] == True)]
    assert len(files_df) == 1, f"No files found for gage: {streamgage}"
    return files_df.iloc[0]["file"]
