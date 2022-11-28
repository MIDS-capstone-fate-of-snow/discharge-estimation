"""Script to run multiple experiments."""

import datetime
import json
import os
import uuid

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models  # NOQA
import yaml

from cnn_dataset import CNNSeqDataset
from cnn_models import SpaceTimeTransformerArchitecture
from utils import convert_datetime

pd.set_option("display.max_colwidth", 500)

DIR = os.getcwd()
DATA_DIR = os.path.join(os.path.dirname(DIR), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "training_data")
EXPERIMENT_DIR = os.path.join(os.path.dirname(DIR), "experiments")
if not os.path.exists(EXPERIMENT_DIR):
    os.mkdir(EXPERIMENT_DIR)


class Experiment:

    def __init__(self, **params):

        self.default_params = dict(
            gages=['11402000'],  # '11266500', '11402000', '11189500', '11318500', '11202710', '11208000', '11185500'],
            y_col="m3",  # "m3_per_area_km", "m3_per_area_miles"
            n_days_precip=7,
            n_days_temp=7,
            n_days_et=8,
            swe_days_relative=[7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84],
            n_days_y=1,
            y_seq=False,
            use_masks=True,
            shuffle_train=True,

            # Model architecture params:
            enc_dense_dim=24, enc_num_heads=4,
            hidden_dim=8, dropout=0.5,
            dropout_concat=False,
            pooling="flatten",

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
        )
        print(f"Num training examples = {len(self.cnn_data.train_pairs)}")

        assert len(self.params["gages"]) == 1
        experiment["architecture"] = "SpaceTimeTransformerArchitecture"
        model = self.get_spacetimetransformer_model()

        # Create the datasets:
        output_signature = (
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
        self.train_data = tf.data.Dataset.from_generator(self.keras_train_gen, output_signature=output_signature)
        self.val_data = tf.data.Dataset.from_generator(self.keras_train_gen, output_signature=output_signature)
        self.test_data = tf.data.Dataset.from_generator(self.keras_train_gen, output_signature=output_signature)

        # Train the model:
        experiment["train_start_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

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
            self.train_data, epochs=self.params["epochs"], batch_size=1,
            steps_per_epoch=train_steps_per_epoch,
            validation_data=self.val_data, validation_steps=n_val_steps,
            validation_batch_size=1, validation_freq=1,
            callbacks=[tensorboard_callback, model_save_callback]
        )
        experiment["train_end_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        # Save the experiment results:

        # Save the validation predictions:
        experiment["val_pred_start_time"] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        n_val_steps = len(self.cnn_data.val_pairs)
        val_pred = model.predict(self.val_data, steps=n_val_steps)
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
        test_pred = model.predict(self.test_data, steps=n_test_steps)
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
            X = [sample[ft] for ft in self.features]
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
            X = [sample[ft] for ft in self.features]
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
            X = [sample[ft] for ft in self.features]
            X = tuple([np.expand_dims(np.expand_dims(x, -1), 0) for x in X])
            yield X, np.expand_dims(sample["y"], 0)
            i += 1

    def get_spacetimetransformer_model(self):
        # Create the model architecture:
        architecture = SpaceTimeTransformerArchitecture()

        model = architecture.get_model(
            gage=self.params["gages"][0],
            n_days_precip=self.params["n_days_precip"],
            n_days_temp=self.params["n_days_temp"],
            n_swe=self.params["n_swe"],
            n_et=self.params["n_et"],
            enc_dense_dim=self.params["enc_dense_dim"],
            enc_num_heads=self.params["enc_num_heads"],
            n_y=self.params["n_days_y"],
            hidden_dim=self.params["hidden_dim"],
            dropout=self.params["dropout"],
            pooling=self.params["pooling"],
        )
        return model


if __name__ == "__main__":

    # Dicts of experiment parameters to try:
    EXPERIMENT_PARAMS = [
        dict(
            n_days_precip=28, n_days_temp=28, n_days_et=32,
            swe_days_relative=[7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84],
            enc_dense_dim=24, enc_num_heads=4, hidden_dim=8, dropout=0.5, dropout_concat=False,
            epochs=5
        ),
        dict(
            n_days_precip=14, n_days_temp=14, n_days_et=16,
            swe_days_relative=[7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84],
            enc_dense_dim=24, enc_num_heads=4, hidden_dim=8, dropout=0.5, dropout_concat=False,
            epochs=5
        ),
        dict(
            n_days_precip=7, n_days_temp=7, n_days_et=8,
            swe_days_relative=[7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84],
            enc_dense_dim=48, enc_num_heads=4, hidden_dim=16, dropout=0.5, dropout_concat=True,
            epochs=5
        ),
        dict(
            n_days_precip=7, n_days_temp=7, n_days_et=8,
            swe_days_relative=[7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84],
            enc_dense_dim=48, enc_num_heads=4, hidden_dim=16, dropout=0.5, dropout_concat=True,
            epochs=5
        ),
        dict(
            n_days_precip=28, n_days_temp=28, n_days_et=32,
            swe_days_relative=[7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84],
            enc_dense_dim=48, enc_num_heads=8, hidden_dim=16, dropout=0.5, dropout_concat=True,
            epochs=5
        ),
        dict(
            n_days_precip=7, n_days_temp=7, n_days_et=8,
            swe_days_relative=[7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84],
            enc_dense_dim=24, enc_num_heads=4, hidden_dim=32, dropout=0.5, dropout_concat=False,
            epochs=5
        ),
        dict(
            n_days_precip=1, n_days_temp=21, n_days_et=8,
            swe_days_relative=[7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84],
            enc_dense_dim=48, enc_num_heads=8, hidden_dim=16, dropout=0.5, dropout_concat=True,
            epochs=5
        ),
        dict(
            n_days_precip=1, n_days_temp=21, n_days_et=8,
            swe_days_relative=[7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84],
            enc_dense_dim=24, enc_num_heads=4, hidden_dim=8, dropout=0.5, dropout_concat=True,
            epochs=5
        ),

    ]
    for gage in ['11402000', '11266500', '11402000', '11189500', '11318500', '11202710', '11208000', '11185500']:
        for experiment_params in EXPERIMENT_PARAMS:
            experiment_params["gage"] = gage  # NOQA
            experiment = Experiment(**experiment_params)
