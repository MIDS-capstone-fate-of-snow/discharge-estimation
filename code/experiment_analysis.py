"""Code for analyzing results of neural network experiments."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yaml

from utils import open_y_data

pd.set_option("display.max_colwidth", 500)

DIR = os.getcwd()
EXPERIMENT_DIR = os.path.join(os.path.dirname(DIR), "experiments")
if not os.path.exists(EXPERIMENT_DIR):
    os.mkdir(EXPERIMENT_DIR)


def experiment_path(name: str):
    return os.path.join(EXPERIMENT_DIR, name)


class ExperimentAnalysis:
    """Class for analyzing all experiment results."""

    def __init__(self):
        self.y = open_y_data()
        self.y["date"] = self.y["date"].dt.date

        experiment_list = os.listdir(EXPERIMENT_DIR)
        experiment_dirs = list()
        experiment_names = list()
        for name in experiment_list:
            fp = experiment_path(name)
            if os.path.isdir(fp) and ("__trained_model" not in name) and ("." not in name):
                experiment_dirs.append(fp)
                experiment_names.append(name)
        self.experiment_names = experiment_names

        yamls = [f"{name}__metadata.yaml" for name in experiment_names]
        metadata = list()
        for yml in yamls:
            try:
                fp = experiment_path(yml)
                with open(fp, "r") as f:
                    data = yaml.safe_load(f)
                    metadata.append(data)
            except FileNotFoundError:
                continue
        experiments = pd.DataFrame(metadata)
        for time_col in [c for c in experiments.columns if "time" in c]:
            experiments[time_col] = pd.to_datetime(experiments[time_col], format="%Y_%m_%d__%H_%M_%S")
        # Only keep the rows which accurately saved metadata:
        self.experiments = experiments.dropna(subset=["y_col"]).reset_index(drop=True).set_index("id")

    def merge_pred_and_truth(self, experiment_id: str, pred_type: str = "val"):
        """Join a dataset of predictions to the true values."""
        experiment_data = self.experiments.loc[experiment_id]
        fp = experiment_path(f"{experiment_id}__{pred_type}_pred.csv")
        df = pd.read_csv(fp, encoding="utf-8")
        df["gage"] = df["gage"].astype(str)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        merged = pd.concat([self.y.set_index(["gage", "date"]),
                            df.set_index(["gage", "date"])], axis=1)
        y_col = experiment_data["y_col"]
        pred_cols = [c for c in merged.columns if "y_day" in c]

        if y_col == "m3":
            pass
        # If predicted area-normalized values, convert back to actual value:
        elif y_col == "m3_per_area_km":
            for col in pred_cols:
                merged[col] = merged[col] * merged["area_km"]
        elif y_col == "m3_per_area_miles":
            for col in pred_cols:
                merged[col] = merged[col] * merged["area_miles"]
        elif y_col == "m3_per_area_m":
            for col in pred_cols:
                merged[col] = merged[col] * merged["area_m"]
        else:
            raise ValueError(f"Code not written for y_col: {y_col}")

        merged = merged.sort_index()

        # Shift the prediction columns to align with the true values:
        for col in pred_cols:
            n_days = int(col.replace("y_day_", ""))
            merged[f"{col}_aligned"] = merged[col].shift(n_days)

        subset = [c for c in merged.columns if "y_day_" in c]
        return merged.dropna(subset=subset, how="all")

    @staticmethod
    def plot_pred_vs_true(merged_df: pd.DataFrame, gage: str,
                          days: list = (1, 7, 14)):
        """Plot predicted discharge values against true values."""
        n_rows = len(days)
        fig, axes = plt.subplots(n_rows, 1, figsize=(8, n_rows*4), dpi=200)
        plot_df = merged_df.reset_index().copy()
        plot_df = plot_df[plot_df["gage"] == gage].sort_values(by=["date"])
        for day, ax in zip(days, axes.flatten()):
            col_name = f"y_day_{day}_aligned"
            plot_df = plot_df.dropna(subset=[col_name])
            ax.plot(plot_df["date"].values, plot_df["m3"].values, color="b", label="ground truth")
            ax.plot(plot_df["date"].values, plot_df[col_name].values, color="r", label="prediction")
            ax.set_title(f"Gage {gage}: Discharge Predictions from {day} Day{'s' if day > 1 else ''} Prior")
            ax.legend()
            ax.set_ylabel("Discharge m3")
        return fig

    @staticmethod
    def compute_experiment_rmse(merged_df: pd.DataFrame):
        """Compute RMSE results for a single experiment."""
        cols = [c for c in merged_df.columns if "y_day_" in c and "_aligned" in c]
        rmse = dict()
        all_pred, all_actual = list(), list()
        for col in cols:
            day = int(col.replace("y_day_", "").replace("_aligned", ""))
            df = merged_df[["m3", col]].dropna(how="any")
            actual = df["m3"].values
            predicted = df[col].values
            score = mean_squared_error(actual, predicted, squared=False)
            rmse[f"rmse_{day}day"] = score
            all_actual += list(actual)
            all_pred += list(predicted)
        rmse["rmse_avg"] = np.mean(list(rmse.values()))
        rmse["rmse_total"] = mean_squared_error(all_actual, all_pred, squared=False)
        return rmse

    def compute_rmse_results(self):
        """Save a CSV of all experiment RMSE results."""
        experiment_results = self.experiments.copy()
        for experiment_id in self.experiments.index:
            merged = self.merge_pred_and_truth(experiment_id)
            rmse = self.compute_experiment_rmse(merged)
            for col, value in rmse.items():
                if col not in experiment_results.columns:
                    experiment_results[col] = np.nan
                experiment_results.loc[experiment_id, col] = value
        experiment_results = experiment_results.reset_index()
        experiment_results = experiment_results.sort_values(by=["rmse_total"], ascending=True)
        first = ["gages", "epochs", "hidden_dim", "enc_num_heads", "dec_num_heads",
                 "n_days_et", "n_days_precip", "n_days_temp", "n_days_y", "n_et", "n_swe"]
        order = first + [c for c in experiment_results.columns if c not in first]
        experiment_results = experiment_results[order]
        results_fp = experiment_path("rmse_results.csv")
        experiment_results.to_csv(results_fp, encoding="utf-8", index=False)
        print(f"Results saved to:\n  {results_fp}")
        return experiment_results
