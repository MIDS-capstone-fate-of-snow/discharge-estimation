"""Code to run neural network architecture experiments."""

import datetime
import json
import os
import uuid
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from cnn_dataset import CNNSeqDataset
from cnn_models import GAPTransArchitecture, GAPTransMaxArchitecture
from utils import convert_datetime

DIR, FILENAME = os.path.split(__file__)
DATA_DIR = os.path.join(os.path.dirname(DIR), "data")
EXPERIMENT_DIR = os.path.join(os.path.dirname(DIR), "experiments")
if not os.path.exists(EXPERIMENT_DIR):
    os.mkdir(EXPERIMENT_DIR)


class GAPModelExperiment:

    def __init__(self, **params):

        # The features, this shouldn't be changed:
        self.features = ("dem", "temp", "precip", "swe", "et")

        # Populated when experiment runs:
        self.experiment_id = None
        self.tf_log_dir = None
        self.val_pred = None
        self.model = None

        # Default experiment parameters:
        self.defaults = dict(
            gages=['11402000', '11318500', '11266500', '11208000', '11202710', '11185500', '11189500'],
            y_col="m3",
            n_days_precip=7,
            n_days_temp=7,
            n_days_et=8,
            swe_days_relative=range(7, 85, 7),
            n_days_y=14,
            use_masks=True,
            shuffle_train=True,

            # Model architecture params:
            dem_kernal=(5, 5), dem_strides=(1, 1),
            et_kernal=(4, 4), et_strides=(1, 1),
            temp_kernal=(2, 2), temp_strides=(1, 1),
            precip_kernal=(2, 2), precip_strides=(1, 1),
            swe_kernal=(5, 5), swe_strides=(1, 1),
            enc_embed_dim=16, enc_dense_dim=32, enc_num_heads=2,
            dec_embed_dim=16, dec_dense_dim=32, dec_num_heads=2,
            hidden_dim=16, dropout=0.5,

            # Model training params:
            epochs=5,
            learning_rate=0.0001,
            opt=keras.optimizers.Adam,
            loss="mean_squared_error",
            tf_board_update_freq=100,
        )
        self.params = {**self.defaults, **params}
        self.params["n_swe"] = len(list(self.params["swe_days_relative"]))  # NOQA
        self.params["n_et"] = self.params["n_days_et"] // 8
        self.train_dir = os.path.join(DATA_DIR, "training_data")
        self.y_fp = os.path.join(DATA_DIR, "streamgage-full.csv")
        date_params = {
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
        date_params_dt = dict()
        for k, v in date_params.items():
            date_params_dt[k] = dict()
            for k2, v2 in v.items():
                date_params_dt[k][k2] = convert_datetime(v2)
        self.date_params = date_params_dt

        # Create the dataset generator:
        self.cnn_data = CNNSeqDataset(
            precip_dirs=[self.train_dir],
            temp_dirs=[self.train_dir],
            et_dirs=[self.train_dir],
            swe_dirs=[self.train_dir],
            y_fp=self.y_fp,
            y_col=self.params["y_col"],  # NOQA
            n_d_precip=self.params["n_days_precip"],
            n_d_temp=self.params["n_days_temp"],
            n_d_et=self.params["n_days_et"],
            swe_d_rel=self.params["swe_days_relative"],  # NOQA
            n_d_y=self.params["n_days_y"],
            min_date=self.date_params["train"]["start"],
            max_date=self.date_params["test"]["end"],
            val_start=self.date_params["val"]["start"],
            test_start=self.date_params["test"]["start"],
            use_masks=self.params["use_masks"],  # NOQA
            random_seed=42,
            shuffle_train=self.params["shuffle_train"],  # NOQA
            gages=self.params["gages"],  # NOQA
        )
        self.params["num_training_samples"] = len(self.cnn_data.train_pairs)

    def new_model(self):
        # Create the model architecture:
        architecture = GAPTransArchitecture()
        model = architecture.get_model(
            dem_kernal=self.params["dem_kernal"], dem_strides=self.params["dem_strides"],
            et_kernal=self.params["et_kernal"], et_strides=self.params["et_strides"],
            temp_kernal=self.params["temp_kernal"], temp_strides=self.params["temp_strides"],
            precip_kernal=self.params["precip_kernal"], precip_strides=self.params["precip_strides"],
            swe_kernal=self.params["swe_kernal"], swe_strides=self.params["swe_strides"],
            n_days_precip=self.params["n_days_precip"],  n_days_temp=self.params["n_days_temp"],
            n_swe=self.params["n_swe"],
            enc_embed_dim=self.params["enc_embed_dim"], enc_dense_dim=self.params["enc_dense_dim"],
            enc_num_heads=self.params["enc_num_heads"],
            dec_embed_dim=self.params["dec_embed_dim"], dec_dense_dim=self.params["dec_dense_dim"],
            dec_num_heads=self.params["dec_num_heads"],
            n_y=self.params["n_days_y"], hidden_dim=self.params["hidden_dim"], dropout=self.params["dropout"],
        )
        self.model = model

    def keras_train_gen(self, debug: bool = False):
        """Construct Keras-compatible train generator."""
        train_data_gen = self.cnn_data.train_data_generator()
        i = 0
        while True:

            # Generate the training sample dict, making the generator infinite:
            try:
                sample = next(train_data_gen)
            except StopIteration:
                # Reset the generator:
                train_data_gen = self.cnn_data.train_data_generator()
                i = 0
                sample = next(train_data_gen)

            # Print debug info:
            if debug:
                print(f"Sample {i}, {sample['debug_data']}")

            # Yield data in format required by tensorflow:
            X = [sample[f] for f in self.features]
            X = tuple([np.expand_dims(np.expand_dims(x, -1), 0) for x in X])
            yield X, np.expand_dims(sample["y"], 0)
            i += 1

    def keras_val_gen(self, debug: bool = False):
        """Construct Keras-compatible validation generator."""
        data_gen = self.cnn_data.val_data_generator()
        i = 0
        while True:

            # Generate the training sample dict, making the generator infinite:
            try:
                sample = next(data_gen)
            except StopIteration:
                # Reset the generator:
                data_gen = self.cnn_data.val_data_generator()
                i = 0
                sample = next(data_gen)

            # Print debug info:
            if debug:
                print(f"Sample {i}, {sample['debug_data']}")

            # Yield data in format required by tensorflow:
            X = [sample[f] for f in self.features]
            X = tuple([np.expand_dims(np.expand_dims(x, -1), 0) for x in X])
            yield X, np.expand_dims(sample["y"], 0)
            i += 1

    def keras_test_gen(self, debug: bool = False):
        """Construct Keras-compatible test generator."""
        data_gen = self.cnn_data.test_data_generator()
        i = 0
        while True:

            # Generate the training sample dict, making the generator infinite:
            try:
                sample = next(data_gen)
            except StopIteration:
                # Reset the generator:
                data_gen = self.cnn_data.test_data_generator()
                i = 0
                sample = next(data_gen)

            # Print debug info:
            if debug:
                print(f"Sample {i}, {sample['debug_data']}")

            # Yield data in format required by tensorflow:
            X = [sample[f] for f in self.features]
            X = tuple([np.expand_dims(np.expand_dims(x, -1), 0) for x in X])
            yield X, np.expand_dims(sample["y"], 0)
            i += 1

    def get_tf_datasets(self):
        # Create the TF Dataset:
        train_data = tf.data.Dataset.from_generator(
            self.keras_train_gen,
            output_signature=(
                (  # X-variables:
                    tf.TensorSpec(shape=(None, 1, None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.params["n_days_temp"], None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.params["n_days_precip"], None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.params["n_swe"], None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.params["n_et"], None, None, 1), dtype=tf.float32),
                ),
                # y-variable:
                tf.TensorSpec(shape=(1, self.params["n_days_y"]), dtype=tf.float32)
            )
        )

        val_data = tf.data.Dataset.from_generator(
            self.keras_val_gen,
            output_signature=(
                (  # X-variables:
                    tf.TensorSpec(shape=(None, 1, None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.params["n_days_temp"], None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.params["n_days_precip"], None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.params["n_swe"], None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.params["n_et"], None, None, 1), dtype=tf.float32),
                ),
                # y-variable:
                tf.TensorSpec(shape=(1, self.params["n_days_y"]), dtype=tf.float32)
            )
        )

        test_data = tf.data.Dataset.from_generator(
            self.keras_test_gen,
            output_signature=(
                (  # X-variables:
                    tf.TensorSpec(shape=(None, 1, None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.params["n_days_temp"], None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.params["n_days_precip"], None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.params["n_swe"], None, None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.params["n_et"], None, None, 1), dtype=tf.float32),
                ),
                # y-variable:
                tf.TensorSpec(shape=(1, self.params["n_days_y"]), dtype=tf.float32)
            )
        )

        return train_data, val_data, test_data

    def train(self):
        """Train the model."""
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.tf_log_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            write_steps_per_second=False,
            update_freq=self.params["tf_board_update_freq"],
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )

        train_data, val_data, test_data = self.get_tf_datasets()
        train_steps_per_epoch = len(self.cnn_data.train_pairs)
        val_steps_per_epoch = len(self.cnn_data.val_pairs)
        optimizer = self.params["opt"](self.params["learning_rate"])  # NOQA
        self.model.compile(optimizer=optimizer, loss=self.params["loss"])
        self.model.fit(train_data, epochs=self.params["epochs"],
                       batch_size=1, steps_per_epoch=train_steps_per_epoch,
                       validation_data=val_data, validation_steps=val_steps_per_epoch,
                       validation_batch_size=1, validation_freq=1,
                       callbacks=[tensorboard_callback])

    def run_experiment(self):
        self.experiment_id = str(uuid.uuid4())
        experiment = self.params.copy()
        self.tf_log_dir = os.path.join(EXPERIMENT_DIR, self.experiment_id)
        if not os.path.exists(self.tf_log_dir):
            os.mkdir(self.tf_log_dir)

        experiment["id"] = self.experiment_id  # NOQA
        experiment["start_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")  # NOQA
        self.new_model()

        # Train the model:
        experiment["train_start_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")  # NOQA
        self.train()
        experiment["train_end_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")  # NOQA

        # Save the validation predictions:
        experiment["val_pred_start_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")  # NOQA
        self.val_pred = self.save_val_pred(self.model)
        experiment["val_pred_end_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")  # NOQA

        # Save the model JSON:
        self.save_model_json(self.model)

        # Save the full trained model:
        fp = os.path.join(EXPERIMENT_DIR, f"{self.experiment_id}__trained_model")
        self.model.save(fp)

        # Save the experiment metadata:
        experiment["end_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")  # NOQA
        self.save_metadata(experiment)

        return experiment

    def save_val_pred(self, model):
        _, val_data, _ = self.get_tf_datasets()
        n_val_steps = len(self.cnn_data.val_pairs)
        val_pred = model.predict(val_data, steps=n_val_steps)
        columns = [f"y_day_{y+1}" for y in range(self.params["n_days_y"])]
        df = pd.DataFrame(val_pred, columns=columns)
        df["gage"] = [t[0] for t in self.cnn_data.val_pairs]
        df["date"] = [t[1].to_pydatetime().date() for t in self.cnn_data.val_pairs]
        fp = os.path.join(EXPERIMENT_DIR, f"{self.experiment_id}_val_pred.csv")
        order = ["gage", "date"] + columns
        df[order].to_csv(fp, encoding="utf-8", index=False)
        return df[order]

    def save_metadata(self, metadata: dict):
        str_metadata = {k: str(v) for k, v in metadata.items()}
        fp = os.path.join(EXPERIMENT_DIR, f"{self.experiment_id}__metadata.yaml")
        with open(fp, "w") as f:
            yaml.safe_dump(str_metadata, f)

    def save_model_json(self, model):
        model_json = model.to_json()
        fp = os.path.join(EXPERIMENT_DIR, f"{self.experiment_id}__model.json")
        with open(fp, "w") as f:
            json.dump(model_json, f)
        return model_json
