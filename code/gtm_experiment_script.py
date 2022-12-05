"""Script to run multiple experiments."""

import datetime
import json
import os
import uuid

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import yaml

from cnn_dataset import CNNSeqDataset
from cnn_models import GAPTransMaxArchitecture
from utils import convert_datetime

pd.set_option("display.max_colwidth", 500)

DIR = os.getcwd()
DATA_DIR = os.path.join(os.path.dirname(DIR), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "training_data")
EXPERIMENT_DIR = os.path.join(os.path.dirname(DIR), "experiments")
if not os.path.exists(EXPERIMENT_DIR):
    os.mkdir(EXPERIMENT_DIR)


class GTMExperiment:

    def __init__(self, **params):

        self.default_params = dict(
            gages=['11402000'],  # '11266500', '11402000', '11189500', '11318500', '11202710', '11208000', '11185500'],
            y_col="m3",  # "m3_per_area_km", "m3_per_area_miles"
            n_days_precip=7,
            n_days_temp=7,
            n_days_et=8,
            swe_days_relative=[7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84],
            n_days_y=14,
            y_seq=True,
            use_masks=True,
            shuffle_train=True,

            # Model architecture params:
            enc_embed_dim=16, enc_dense_dim=32, enc_num_heads=2,
            dec_embed_dim=16, dec_dense_dim=32, dec_num_heads=2,
            kernal=(2, 2), strides=(1, 1),
            hidden_dim=16,
            dropout=0.5,
            pooling="avg",
            cnn_activation="relu",

            # Sample weighting params:
            sample_z_score=None,
            sample_weight=None,

            # Model training params:
            epochs=5,
            learning_rate=0.0001,
            opt=keras.optimizers.Adam,
            loss="mean_squared_error",
            tf_board_update_freq=100,
        )
        self.params = {**self.default_params, **params}

        # Constant/calculated parameters:
        self.params["n_swe"] = len(list(self.params["swe_days_relative"]))
        self.params["n_et"] = self.params["n_days_et"] // 8

        # Create the experiment:
        experiment = dict()
        experiment["start_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        experiment_id = str(uuid.uuid4())
        print(f"experiment['experiment_id'] = '{experiment_id}'")
        experiment["id"] = experiment_id

        # Make dir for logging tf objects:
        TF_LOG_DIR = os.path.join(EXPERIMENT_DIR, experiment_id)
        if not os.path.exists(TF_LOG_DIR):
            os.mkdir(TF_LOG_DIR)

        self.features = ("dem", "temp", "precip", "swe", "et")
        self.y_fp = os.path.join(DATA_DIR, "streamgage-full.csv")
        dates = {
            "train": {
                "start": "2010_01_01",
                "end": "2014_12_31",
            },
            "val": {
                "start": "2015_01_01",
                "end": "2015_12_31",
            },
            "test": {
                "start": "2016_01_01",
                "end": "2016_12_31",
            },
        }
        DATE_PARAMS = dict()
        for k, v in dates.items():
            DATE_PARAMS[k] = dict()
            for k2, v2 in v.items():
                DATE_PARAMS[k][k2] = convert_datetime(v2)

        # Create the dataset object:
        self.cnn_data = CNNSeqDataset(
            precip_dirs=[TRAIN_DIR],
            temp_dirs=[TRAIN_DIR],
            et_dirs=[TRAIN_DIR],
            swe_dirs=[TRAIN_DIR],
            y_fp=self.y_fp,
            y_col=self.params["y_col"],
            n_d_precip=self.params["n_days_precip"],
            n_d_temp=self.params["n_days_temp"],
            n_d_et=self.params["n_days_et"],
            swe_d_rel=self.params["swe_days_relative"],
            n_d_y=self.params["n_days_y"],
            min_date=DATE_PARAMS["train"]["start"],
            max_date=DATE_PARAMS["test"]["end"],
            val_start=DATE_PARAMS["val"]["start"],
            test_start=DATE_PARAMS["test"]["start"],
            use_masks=self.params["use_masks"],
            random_seed=42,
            shuffle_train=self.params["shuffle_train"],
            gages=self.params["gages"],
            sample_z_score=self.params["sample_z_score"],
            sample_weight=self.params["sample_weight"],
        )
        print(f"Num training examples = {len(self.cnn_data.train_pairs)}")

        assert len(self.params["gages"]) == 1
        experiment["architecture"] = "GAPTransMaxArchitecture"
        model = self.get_gaptransmax_model()
        fp = os.path.join(EXPERIMENT_DIR, f"{experiment_id}__model.png")
        keras.utils.plot_model(model, fp, show_shapes=True)

        # Train the model:
        experiment["train_start_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        datasets = self.cnn_data.keras_datasets()

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=TF_LOG_DIR,
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            write_steps_per_second=False,
            update_freq=self.params["tf_board_update_freq"],
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )

        ckpt_dir = os.path.join(EXPERIMENT_DIR, f"{experiment_id}__ckpts")
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        model_save_callback = tf.keras.callbacks.ModelCheckpoint(
            f"{ckpt_dir}/model__" + "{epoch:02d}-{val_loss:.2f}.hdf5",
            monitor="val_loss",
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            options=None,
            initial_value_threshold=None,
        )

        train_steps_per_epoch = len(self.cnn_data.train_pairs)
        n_val_steps = len(self.cnn_data.val_pairs)
        optimizer = self.params["opt"](self.params["learning_rate"])
        model.compile(optimizer=optimizer, loss=self.params["loss"])
        model.fit(
            datasets["train_data"], epochs=self.params["epochs"], batch_size=1,
            steps_per_epoch=train_steps_per_epoch,
            validation_data=datasets["val_data"], validation_steps=n_val_steps,
            validation_batch_size=1, validation_freq=1,
            callbacks=[tensorboard_callback, model_save_callback]
        )
        experiment["train_end_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        # Save the experiment results:

        # Save the validation predictions:
        experiment["val_pred_start_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        n_val_steps = len(self.cnn_data.val_pairs)
        val_pred = model.predict(datasets["val_data"], steps=n_val_steps)
        columns = [f"y_day_{y+1}" for y in range(self.params["n_days_y"])]
        df = pd.DataFrame(val_pred, columns=columns)
        df["gage"] = [t[0] for t in self.cnn_data.val_pairs]
        df["date"] = [t[1].to_pydatetime().date() for t in self.cnn_data.val_pairs]
        fp = os.path.join(EXPERIMENT_DIR, f"{experiment_id}__val_pred.csv")
        order = ["gage", "date"] + columns
        val_pred = df[order]
        val_pred.to_csv(fp, encoding="utf-8", index=False)
        experiment["val_pred_end_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        # Save the test predictions:
        experiment["test_pred_start_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        n_test_steps = len(self.cnn_data.test_pairs)
        test_pred = model.predict(datasets["test_data"], steps=n_test_steps)
        columns = [f"y_day_{y+1}" for y in range(self.params["n_days_y"])]
        df = pd.DataFrame(test_pred, columns=columns)
        df["gage"] = [t[0] for t in self.cnn_data.test_pairs]
        df["date"] = [t[1].to_pydatetime().date() for t in self.cnn_data.test_pairs]
        fp = os.path.join(EXPERIMENT_DIR, f"{experiment_id}__test_pred.csv")
        order = ["gage", "date"] + columns
        test_pred = df[order]
        test_pred.to_csv(fp, encoding="utf-8", index=False)
        experiment["test_pred_end_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        # Save the model JSON:
        model_json = model.to_json()
        fp = os.path.join(EXPERIMENT_DIR, f"{experiment_id}__model.json")
        with open(fp, "w") as f:
            json.dump(model_json, f)

        # Save the full trained model:
        fp = os.path.join(EXPERIMENT_DIR, f"{experiment_id}__trained_model")
        model.save(fp)

        # Save the experiment metadata:
        experiment["end_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        metadata = {**self.params, **experiment}
        str_metadata = {k: str(v) for k, v in metadata.items()}
        fp = os.path.join(EXPERIMENT_DIR, f"{experiment_id}__metadata.yaml")
        with open(fp, "w") as f:
            yaml.safe_dump(str_metadata, f)

        print(f"Finished experiment id = '{experiment_id}'")

    def get_gaptransmax_model(self):
        # Create the model architecture:
        architecture = GAPTransMaxArchitecture()

        model = architecture.get_model(
            kernal=self.params["kernal"],
            strides=self.params["strides"],
            n_days_precip=self.params["n_days_precip"],
            n_days_temp=self.params["n_days_temp"],
            n_swe=self.params["n_swe"],
            n_et=self.params["n_et"],
            enc_embed_dim=self.params["enc_embed_dim"],
            enc_dense_dim=self.params["enc_dense_dim"],
            enc_num_heads=self.params["enc_num_heads"],
            dec_embed_dim=self.params["dec_embed_dim"],
            dec_dense_dim=self.params["dec_dense_dim"],
            dec_num_heads=self.params["dec_num_heads"],
            n_y=self.params["n_days_y"],
            hidden_dim=self.params["hidden_dim"],
            dropout=self.params["dropout"],
            cnn_activation=self.params["cnn_activation"],
            pooling=self.params["pooling"],
        )
        return model


if __name__ == "__main__":

    for sz, sw in [
        (0.5, 5),
        (0.5, 2),
        (1, 10),
        (1, 5),
        (1, 2),
        (2, 10),
        (2, 5),
        (2, 2),
        (3, 10),
        (3, 5),
        (3, 2),
    ]:
        exp_params = dict(
            gages=['11402000', '11189500', '11318500', '11266500', '11202710'],  # '11208000', '11185500'
            y_col="m3",  # "m3_per_area_km", "m3_per_area_miles"
            n_days_precip=1,
            n_days_temp=21,
            n_days_et=8,
            swe_days_relative=[7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84],
            n_days_y=14,
            y_seq=True,
            use_masks=True,
            shuffle_train=True,

            # Model architecture params:
            enc_embed_dim=24, enc_dense_dim=48, enc_num_heads=4,
            dec_embed_dim=24, dec_dense_dim=48, dec_num_heads=4,
            kernal=(2, 2), strides=(1, 1),
            hidden_dim=24,
            dropout=0.5,
            pooling="avg",
            cnn_activation="relu",

            # Sample weighting params:
            sample_z_score=sz,
            sample_weight=sw,

            # Model training params:
            epochs=5,
            learning_rate=0.0001,
            opt=keras.optimizers.Adam,
            loss="mean_squared_error",
            tf_board_update_freq=100,
        )
        exp = GTMExperiment(**exp_params)
