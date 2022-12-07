"""Code to load a saved experiment and work with it."""

from collections import defaultdict
import os
import warnings

from keras.models import load_model
from keras.utils import custom_object_scope
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from cnn_models import TransformerDecoder, TransformerEncoder
from cnn_dataset import CNNSeqDataset
from utils import open_y_data, score_mape, score_rmse, score_rrmse

DIR, FILENAME = os.path.split(__file__)
DATA_DIR = os.path.join(os.path.dirname(DIR), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "training_data")
EXPERIMENT_DIR = os.path.join(os.path.dirname(DIR), "experiments")
if not os.path.exists(EXPERIMENT_DIR):
    os.mkdir(EXPERIMENT_DIR)


def experiment_path(name: str):
    return os.path.join(EXPERIMENT_DIR, name)


class Experiment:
    """Work with a saved experiment."""

    def __init__(self, experiment_id: str):
        self.ground_truth = open_y_data()
        self.experiment_id = experiment_id
        metadata_fp = f"{self.experiment_id}__metadata.yaml"
        fp = experiment_path(metadata_fp)
        with open(fp, "r") as f:
            self.metadata = yaml.safe_load(f)
        for int_key in ("n_days_precip", "n_days_temp", "n_days_et", "n_days_y", "n_swe", "n_et"):
            self.metadata[int_key] = int(self.metadata[int_key])
        for eval_key in ("swe_days_relative", "y_seq", "use_masks", "gages", "shuffle_train", "log_transform_y"):
            try:
                self.metadata[eval_key] = eval(self.metadata[eval_key])
            except KeyError:  # Use defaults for earlier experiments which didn't have parameter:
                defaults = {"y_seq": True}
                self.metadata[eval_key] = defaults[eval_key]
        self.ckpt_dir = experiment_path(f"{experiment_id}__ckpts")
        try:
            self.ckpts = sorted(filter(lambda fn: fn.endswith(".hdf5"), os.listdir(self.ckpt_dir)))
        except FileNotFoundError:
            self.ckpts = []
        self.trained_model_dir = experiment_path(f"{experiment_id}__trained_model")
        self.cnn_dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.loaded_models = dict()
        self.predictions = defaultdict(dict)

    def load_trained_model(self, epoch: int = None):
        """Load the trained model.

        Args:
            epoch: which epoch's model to load. If not passed
                loads the final saved model.
        """
        if epoch is None:
            model_fp = self.trained_model_dir
        else:
            ckpt = self.ckpts[epoch-1]
            model_fp = os.path.join(self.ckpt_dir, ckpt)

        with custom_object_scope({
            'TransformerEncoder': TransformerEncoder,
            'TransformerDecoder': TransformerDecoder
        }):
            model = load_model(model_fp)
        self.loaded_models[epoch] = model

    def load_model_dataset(self):
        """Load the CNNSeqDataset with the correct parameters
        for the experiment model."""
        self.cnn_dataset = CNNSeqDataset(
            precip_dirs=[TRAIN_DIR],
            temp_dirs=[TRAIN_DIR],
            et_dirs=[TRAIN_DIR],
            swe_dirs=[TRAIN_DIR],
            y_fp=os.path.join(DATA_DIR, "streamgage-full.csv"),
            y_col=self.metadata["y_col"],
            n_d_precip=self.metadata["n_days_precip"],
            n_d_temp=self.metadata["n_days_temp"],
            n_d_et=self.metadata["n_days_et"],
            swe_d_rel=self.metadata["swe_days_relative"],
            n_d_y=self.metadata["n_days_y"],
            y_seq=self.metadata["y_seq"],
            min_date="2010_01_01",
            max_date="2016_12_31",
            val_start="2015_01_01",
            test_start="2016_01_01",
            use_masks=self.metadata["use_masks"],
            gages=self.metadata["gages"],
            random_seed=42,
            shuffle_train=self.metadata["shuffle_train"],
        )
        # Create the datasets:
        keras_datasets = self.cnn_dataset.keras_datasets()
        self.train_data = keras_datasets["train_data"]
        self.val_data = keras_datasets["val_data"]
        self.test_data = keras_datasets["test_data"]

    def get_predictions(self, predict: str = "test", epoch: int = None):
        """Make predictions using the save model.

        Args:
            predict: what to predict; either 'val' or 'test'.
            epoch: optional epoch's model to load, else just use final model.
        """
        try:
            return self.predictions[epoch][predict]
        except KeyError:
            pass
        if self.cnn_dataset is None:
            self.load_model_dataset()
        dataset = {"test": self.test_data,
                   "val": self.val_data,
                   "train": self.train_data}[predict]
        pairs = {"test": self.cnn_dataset.test_pairs,
                 "val": self.cnn_dataset.val_pairs,
                 "train": self.cnn_dataset.train_pairs}[predict]
        n_steps = len(pairs)
        if self.loaded_models.get(epoch) is None:
            self.load_trained_model(epoch)
        model = self.loaded_models[epoch]
        pred = model.predict(dataset, steps=n_steps)
        y_seq = self.metadata.get("y_seq", True)
        n_days_y = self.metadata["n_days_y"]
        if y_seq:
            columns = [f"y_day_{y+1}" for y in range(n_days_y)]
        else:
            columns = [f"y_day_{n_days_y}"]
        df = pd.DataFrame(pred, columns=columns)

        # Undo log transformations:
        if self.metadata["log_transform_y"]:
            for c in columns:
                df[c] = np.exp(df[c])

        df["gage"] = [t[0] for t in pairs]
        df["date"] = [t[1].to_pydatetime().date() for t in pairs]
        order = ["gage", "date"] + columns
        pred = df[order]
        self.predictions[epoch][predict] = pred
        return pred

    def load_predictions(self, predict: str = "test"):
        fp = experiment_path(f"{self.experiment_id}__{predict}_pred.csv")
        df = pd.read_csv(fp)
        df["gage"] = df["gage"].astype(str)
        return df

    def get_test_set(self, day: int = None):
        # Get the date range for the max prediction:

        # TODO: Only have these predictions for LSTM currently.
        gage = "11402000"

        # Load the model predictions:
        pred = self.load_predictions("test")
        pred["date"] = pd.to_datetime(pred["date"])  # NOQA
        available_cols = [c for c in pred.columns if c.startswith("y_day_")]
        max_day_available = max([int(c.replace("y_day_", "")) for c in available_cols])
        if (day is None) or (max_day_available < day):
            if isinstance(day, int):
                warnings.warn(f"Requested predictions for day {day}, but latest available is "
                              f"{max_day_available}. Returning those instead.")
            day = max_day_available
            col = f"y_day_{max_day_available}"
        else:
            col = f"y_day_{day}"

        pred = pred[pred["gage"] == gage].set_index("date")[col]
        pred.name = "pred"
        pred = pred.shift(day-1).dropna()

        # Get the true values:
        y = self.ground_truth
        y = y[y["gage"] == gage].set_index("date").loc[pred.index][self.metadata["y_col"]]
        y.name = "ground_truth"

        # Get the LSTM predictions:
        lstm_pred = self.lstm_results(day)

        test_set = pd.concat([y, pred, lstm_pred], axis=1).dropna()

        # Set minimum to zero for predictions:
        for col in ("pred", "lstm_pred"):
            test_set[col] = np.where(test_set[col] < 0, 0, test_set[col])

        return test_set

    @staticmethod
    def lstm_results(day: int):
        fp = os.path.join(DATA_DIR, "LSTM_11402000_test_pred.csv")
        lstm_pred = pd.read_csv(fp)
        lstm_pred["pred_date"] = pd.to_datetime(lstm_pred["pred_date"])
        col = f"day{day}_pred"
        lstm_pred = lstm_pred.set_index("pred_date")[col]
        lstm_pred.name = "lstm_pred"
        # TODO: confirm with Zixi logic for shifting dates:
        lstm_pred = lstm_pred.shift(day + 28).dropna()
        return lstm_pred

    def plot_test_pred_vs_lstm(self, day: int = 1, figsize=(14, 4), dpi=150):

        # Get the dates for the max day window for reference:
        reference_test_set = self.get_test_set(None)
        index = reference_test_set.index

        # TODO: Only have these predictions for LSTM currently.
        gage = "11402000"

        test_set = self.get_test_set(day).loc[index]
        print(len(test_set))

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        ax = axes[1]
        ax.plot(test_set["pred"], color="r", label="Prediction")
        ax.plot(test_set["ground_truth"], color="b", label="Ground Truth")
        rmse = score_rmse(test_set["ground_truth"], test_set["pred"])
        rrmse = score_rrmse(test_set["ground_truth"], test_set["pred"])
        mape = score_mape(test_set["ground_truth"], test_set["pred"])
        architecture = self.metadata["architecture"].replace("Architecture", "")
        ax.set_title(f"{architecture} Model\nRMSE={rmse:.2f}, RRMSE={rrmse:.2f}, MAPE={mape:.2f}")
        ax.legend()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
        ax.set_xlim(test_set.index.min(), test_set.index.max())
        ax.set_ylabel("Discharge m3")

        ax = axes[0]
        ax.plot(test_set["lstm_pred"], color="r", label="Prediction")
        ax.plot(test_set["ground_truth"], color="b", label="Ground Truth")
        rmse = score_rmse(test_set["ground_truth"], test_set["lstm_pred"])
        rrmse = score_rrmse(test_set["ground_truth"], test_set["lstm_pred"])
        mape = score_mape(test_set["ground_truth"], test_set["lstm_pred"])
        ax.set_title(f"Baseline LSTM Model\nRMSE={rmse:.2f}, RRMSE={rrmse:.2f}, MAPE={mape:.2f}")
        ax.legend()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
        ax.set_xlim(test_set.index.min(), test_set.index.max())
        ax.set_ylabel("Discharge m3")

        fig.suptitle(f"Gage {gage} 2016 Test Set Predictions, {day} Day{'s' if day > 1 else ''} Ahead", y=1.05)
        return fig
