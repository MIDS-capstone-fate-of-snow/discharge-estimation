"""CNN model architectures."""

import os

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models  # NOQA
import yaml

DIR, FILENAME = os.path.split(__file__)
DATA_DIR = os.path.join(os.path.dirname(DIR), "data")


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

    @staticmethod
    def get_causal_attention_mask(inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = causal_mask  # add missing else condition. Use causal mask when mask is not given
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)


def time_dist_cnn(observation_period: int, band_name: str,
                  n_filters: int = 16, kernel_size: tuple = (2, 2),
                  strides: tuple = (1, 1), activation: str = "relu",
                  inputs=None, pooling: str = "avg",
                  w: int = None, h: int = None):
    """Create time-distributed CNN with Global Pooling layer."""
    if inputs is None:
        inputs = keras.Input(shape=(observation_period, w, h, 1),
                             batch_size=None, name=f"{band_name}_inputs")
    conv_2d_layer = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                                  strides=strides, activation=activation)
    x = layers.TimeDistributed(conv_2d_layer, name=f"{band_name}_time_dist_conv2d")(inputs)
    if pooling == "avg":
        pooling_layer = layers.GlobalAveragePooling2D(data_format="channels_last", keepdims=False)
        name = f"global_{pooling}_pooling"
    elif pooling == "max":
        pooling_layer = layers.GlobalMaxPooling2D(data_format="channels_last", keepdims=False)
        name = f"global_{pooling}_pooling"
    elif pooling == "flatten":
        pooling_layer = layers.Flatten()
        name = "flatten"
    else:
        raise ValueError(f"Invalid `pooling` arg: {pooling}")
    outputs = layers.TimeDistributed(pooling_layer, name=f"{band_name}_{name}")(x)
    return inputs, outputs


class GAPTransArchitecture:
    """CNN architecture using Global Average Pooling and Transformer."""

    def __init__(self):
        pass

    @staticmethod
    def get_model(dem_kernal=(5, 5), dem_strides=(1, 1),
                  et_kernal=(4, 4), et_strides=(1, 1),
                  temp_kernal=(2, 2), temp_strides=(1, 1),
                  precip_kernal=(2, 2), precip_strides=(1, 1),
                  swe_kernal=(5, 5), swe_strides=(1, 1),
                  n_days_precip=7, n_days_temp=7,
                  n_swe=12, n_et=1,
                  enc_embed_dim=16, enc_dense_dim=32, enc_num_heads=2,
                  dec_embed_dim=16, dec_dense_dim=32, dec_num_heads=2,
                  n_y=14, hidden_dim=16, dropout=0.5, pooling: str = "avg"):
        """Create a new CNN model architecture with the specified parameters."""
        # Single image CNN inputs - dem:
        dem_inputs, dem_outputs = time_dist_cnn(
            1, "dem", hidden_dim, kernel_size=dem_kernal, strides=dem_strides, pooling=pooling)

        # Multiple image CNN inputs - temp / precip / swe / et:
        et_inputs, et_outputs = time_dist_cnn(
            n_et, "et", hidden_dim, kernel_size=et_kernal, strides=et_strides, pooling=pooling)
        temp_inputs, temp_outputs = time_dist_cnn(
            n_days_temp, "temp", hidden_dim, kernel_size=temp_kernal, strides=temp_strides, pooling=pooling)
        precip_inputs, precip_outputs = time_dist_cnn(
            n_days_precip, "precip", hidden_dim, kernel_size=precip_kernal, strides=precip_strides, pooling=pooling)
        swe_inputs, swe_outputs = time_dist_cnn(
            n_swe, "swe", hidden_dim, kernel_size=swe_kernal, strides=swe_strides, pooling=pooling)

        # Concatenate CNN outputs:
        concat = tf.keras.layers.Concatenate(axis=1)(
            [dem_outputs, temp_outputs, precip_outputs, swe_outputs, et_outputs]
        )

        # Transformer encoder:
        encoder_outputs = TransformerEncoder(embed_dim=enc_embed_dim, dense_dim=enc_dense_dim,  # NOQA
                                             num_heads=enc_num_heads)(concat)

        # Transformer decoder:
        x = TransformerDecoder(embed_dim=dec_embed_dim, dense_dim=dec_dense_dim,  # NOQA
                               num_heads=dec_num_heads)(concat, encoder_outputs)
        x = layers.Dropout(dropout)(x)
        x = layers.Flatten()(x)
        decoder_outputs = layers.Dense(n_y, activation="linear")(x)

        # Create the model:
        transformer = keras.Model([dem_inputs, temp_inputs, precip_inputs, swe_inputs, et_inputs], decoder_outputs)
        return transformer


class GAPTransMaxArchitecture:
    """CNN architecture using Global Average Pooling and Transformer, plus max-
    pooling layers to resize larger images."""

    def __init__(self):
        pass

    @staticmethod
    def et_cnn(observation_period: int, activation: str = "relu"):
        inputs = keras.Input(shape=(observation_period, None, None, 1), batch_size=None, name="ET_inputs")

        # First convolutional layer:
        conv_2d_layer = layers.Conv2D(
            filters=1, kernel_size=(3, 3), strides=(2, 2), activation=activation)
        x = layers.TimeDistributed(conv_2d_layer, name=f"ET_conv2d_0")(inputs)
        pooling_layer = layers.MaxPooling2D()
        outputs = layers.TimeDistributed(pooling_layer, name=f"ET_max_pooling_0")(x)

        # Second convolutional layer:
        conv_2d_layer1 = layers.Conv2D(
            filters=1, kernel_size=(2, 2), strides=(1, 1), activation=activation)
        x = layers.TimeDistributed(conv_2d_layer1, name="ET_conv2d_1")(outputs)
        pooling_layer1 = layers.MaxPooling2D()
        outputs1 = layers.TimeDistributed(pooling_layer1, name=f"ET_max_pooling_1")(x)

        return inputs, outputs1

    @staticmethod
    def swe_cnn(observation_period: int, activation: str = "relu"):
        inputs = keras.Input(shape=(observation_period, None, None, 1), batch_size=None, name="SWE_inputs")

        # First convolutional layer:
        conv_2d_layer = layers.Conv2D(
            filters=1, kernel_size=(5, 5), strides=(3, 3), activation=activation)
        x = layers.TimeDistributed(conv_2d_layer, name=f"SWE_conv2d_0")(inputs)
        pooling_layer = layers.MaxPooling2D()
        outputs = layers.TimeDistributed(pooling_layer, name=f"SWE_max_pooling_0")(x)

        # Second convolutional layer:
        conv_2d_layer1 = layers.Conv2D(
            filters=1, kernel_size=(3, 3), strides=(2, 2), activation=activation)
        x = layers.TimeDistributed(conv_2d_layer1, name="SWE_conv2d_1")(outputs)
        pooling_layer1 = layers.MaxPooling2D()
        outputs1 = layers.TimeDistributed(pooling_layer1, name=f"SWE_max_pooling_1")(x)

        return inputs, outputs1

    @staticmethod
    def dem_cnn(observation_period: int, activation: str = "relu"):
        inputs = keras.Input(shape=(observation_period, None, None, 1), batch_size=None, name="DEM_inputs")

        # First convolutional layer:
        conv_2d_layer = layers.Conv2D(
            filters=1, kernel_size=(5, 5), strides=(3, 3), activation=activation)
        x = layers.TimeDistributed(conv_2d_layer, name="DEM_conv2d_0")(inputs)
        pooling_layer = layers.MaxPooling2D()
        outputs = layers.TimeDistributed(pooling_layer, name=f"DEM_max_pooling_0")(x)

        # Second convolutional layer:
        conv_2d_layer1 = layers.Conv2D(
            filters=1, kernel_size=(4, 4), strides=(2, 2), activation=activation)
        x = layers.TimeDistributed(conv_2d_layer1, name="DEM_conv2d_1")(outputs)
        pooling_layer1 = layers.MaxPooling2D()
        outputs1 = layers.TimeDistributed(pooling_layer1, name=f"DEM_max_pooling_1")(x)

        # Third convolutional layer:
        conv_2d_layer2 = layers.Conv2D(
            filters=1, kernel_size=(3, 3), strides=(2, 2), activation=activation)
        x = layers.TimeDistributed(conv_2d_layer2, name="DEM_conv2d_2")(outputs1)
        pooling_layer2 = layers.MaxPooling2D()
        outputs2 = layers.TimeDistributed(pooling_layer2, name=f"DEM_max_pooling_2")(x)

        return inputs, outputs2

    def get_model(self, kernal=(2, 2), strides=(1, 1),
                  n_days_precip=7, n_days_temp=7,
                  n_swe=12, n_et=1,
                  enc_embed_dim=16, enc_dense_dim=32, enc_num_heads=2,
                  dec_embed_dim=16, dec_dense_dim=32, dec_num_heads=2,
                  n_y=14, hidden_dim=16, dropout=0.5, cnn_activation="relu",
                  pooling: str = "avg"):
        """Create a new CNN model architecture with the specified parameters."""

        # Create the max-pooling layers first:
        maxpool_dem_inputs, maxpool_dem_outputs = self.dem_cnn(
            observation_period=1, activation=cnn_activation)
        maxpool_et_inputs, maxpool_et_outputs = self.et_cnn(
            observation_period=n_et, activation=cnn_activation)
        maxpool_swe_inputs, maxpool_swe_outputs = self.swe_cnn(
            observation_period=n_swe, activation=cnn_activation)

        # Create the GAP CNNs:
        gap_dem_inputs, gap_dem_outputs = time_dist_cnn(
            1, "dem", hidden_dim, kernel_size=kernal, strides=strides, inputs=maxpool_dem_outputs, pooling=pooling)
        gap_et_inputs, gap_et_outputs = time_dist_cnn(
            n_et, "et", hidden_dim, kernel_size=kernal, strides=strides, inputs=maxpool_et_outputs, pooling=pooling)
        gap_temp_inputs, gap_temp_outputs = time_dist_cnn(
            n_days_temp, "temp", hidden_dim, kernel_size=kernal, strides=strides, pooling=pooling)
        gap_precip_inputs, gap_precip_outputs = time_dist_cnn(
            n_days_precip, "precip", hidden_dim, kernel_size=kernal, strides=strides, pooling=pooling)
        gap_swe_inputs, gap_swe_outputs = time_dist_cnn(
            n_swe, "swe", hidden_dim, kernel_size=kernal, strides=strides, inputs=maxpool_swe_outputs, pooling=pooling)

        # Concatenate CNN outputs:
        concat = tf.keras.layers.Concatenate(axis=1)(
            [gap_dem_outputs, gap_temp_outputs, gap_precip_outputs, gap_swe_outputs, gap_et_outputs]
        )

        # Transformer encoder:
        encoder_outputs = TransformerEncoder(embed_dim=enc_embed_dim, dense_dim=enc_dense_dim,  # NOQA
                                             num_heads=enc_num_heads)(concat)

        # Transformer decoder:
        x = TransformerDecoder(embed_dim=dec_embed_dim, dense_dim=dec_dense_dim,  # NOQA
                               num_heads=dec_num_heads)(concat, encoder_outputs)
        x = layers.Dropout(dropout)(x)
        x = layers.Flatten()(x)
        decoder_outputs = layers.Dense(n_y, activation="linear")(x)

        # Create the model:
        transformer = keras.Model(
            [maxpool_dem_inputs, gap_temp_inputs, gap_precip_inputs, maxpool_swe_inputs, maxpool_et_inputs],
            decoder_outputs
        )
        return transformer


class ConvResizeGAPTransArchitecture:
    """CNN architecture using Global Average Pooling and Transformer, plus
    simple convolutional layer to resize larger images to same dimensions."""

    def __init__(self):
        pass

    @staticmethod
    def get_model(et_ksize=(9, 9), swe_ksize=(35, 35), dem_ksize=(138, 138),
                  kernal=(2, 2), strides=(1, 1),
                  n_days_precip=7, n_days_temp=7, n_swe=12, n_et=1,
                  enc_embed_dim=16, enc_dense_dim=32, enc_num_heads=2,
                  dec_embed_dim=16, dec_dense_dim=32, dec_num_heads=2,
                  final_dense_dim=14, one_day_pred=False,
                  hidden_dim=16, dropout=0.5, cnn_activation="relu",
                  pooling: str = "avg"):
        """Create a new CNN model architecture with the specified parameters."""

        # ET resizer:
        et_raw_inputs = keras.Input(shape=(n_et, None, None, 1), batch_size=None, name="et_inputs")
        et_conv_2d_layer = layers.Conv2D(
            filters=1, kernel_size=et_ksize, strides=et_ksize, activation=cnn_activation)
        et_resized_inputs = layers.TimeDistributed(et_conv_2d_layer, name="et_resizer")(et_raw_inputs)

        # SWE resizer:
        swe_raw_inputs = keras.Input(shape=(n_swe, None, None, 1), batch_size=None, name="swe_inputs")
        swe_conv_2d_layer = layers.Conv2D(
            filters=1, kernel_size=swe_ksize, strides=swe_ksize, activation=cnn_activation)
        swe_resized_inputs = layers.TimeDistributed(swe_conv_2d_layer, name="swe_resizer")(swe_raw_inputs)

        # DEM resizer:
        dem_raw_inputs = keras.Input(shape=(1, None, None, 1), batch_size=None, name="dem_inputs")
        dem_conv_2d_layer = layers.Conv2D(
            filters=1, kernel_size=dem_ksize, strides=dem_ksize, activation=cnn_activation)
        dem_resized_inputs = layers.TimeDistributed(dem_conv_2d_layer, name="dem_resizer")(dem_raw_inputs)

        # Create the GAP CNNs:
        gap_dem_inputs, gap_dem_outputs = time_dist_cnn(
            1, "dem", hidden_dim, kernel_size=kernal, strides=strides, inputs=dem_resized_inputs, pooling=pooling)
        gap_et_inputs, gap_et_outputs = time_dist_cnn(
            1, "et", hidden_dim, kernel_size=kernal, strides=strides, inputs=et_resized_inputs, pooling=pooling)
        gap_temp_inputs, gap_temp_outputs = time_dist_cnn(
            n_days_temp, "temp", hidden_dim, kernel_size=kernal, strides=strides, pooling=pooling)
        gap_precip_inputs, gap_precip_outputs = time_dist_cnn(
            n_days_precip, "precip", hidden_dim, kernel_size=kernal, strides=strides, pooling=pooling)
        gap_swe_inputs, gap_swe_outputs = time_dist_cnn(
            n_swe, "swe", hidden_dim, kernel_size=kernal, strides=strides, inputs=swe_resized_inputs, pooling=pooling)

        # Concatenate CNN outputs:
        concat = tf.keras.layers.Concatenate(axis=1)(
            [gap_dem_outputs, gap_temp_outputs, gap_precip_outputs, gap_swe_outputs, gap_et_outputs]
        )

        # Transformer encoder:
        encoder_outputs = TransformerEncoder(embed_dim=enc_embed_dim, dense_dim=enc_dense_dim,  # NOQA
                                             num_heads=enc_num_heads)(concat)

        # Transformer decoder:
        x = TransformerDecoder(embed_dim=dec_embed_dim, dense_dim=dec_dense_dim,  # NOQA
                               num_heads=dec_num_heads)(concat, encoder_outputs)
        x = layers.Dropout(dropout)(x)
        x = layers.Flatten()(x)

        # Final prediction layer(s):
        assert (final_dense_dim is not None) or one_day_pred
        if final_dense_dim is not None:
            x = layers.Dense(final_dense_dim, activation="linear")(x)
        if one_day_pred:
            x = layers.Dense(1, activation="linear")(x)

        # Create the model:
        transformer = keras.Model(
            [dem_raw_inputs, gap_temp_inputs, gap_precip_inputs, swe_raw_inputs, et_raw_inputs],
            x
        )
        return transformer


def get_kernal_stride(input_sizes: dict):
    """Use temp/precip size with kernal size (2, 2), stride (1, 1) to
    determine sizes of other bands.

    Args:
        input_sizes: dict of band name to tuple of pixel w/h dimensions.
    """

    w, h = input_sizes["precip"]
    assert input_sizes["temp"] == [w, h]
    w_kernal_ratio = 2 / w
    h_kernal_ratio = 2 / h
    w_stride_ratio = 1 / w
    h_stride_ratio = 1 / h

    kernal_stride = {k: {"kernal": (2, 2), "stride": (1, 1)} for k in ("precip", "temp")}
    for k in ("dem", "et", "swe"):
        w, h = input_sizes[k]
        kernal_w = int(w * w_kernal_ratio)
        kernal_h = int(h * h_kernal_ratio)
        stride_w = int(w * w_stride_ratio)
        stride_h = int(h * h_stride_ratio)
        kernal_stride[k] = {"kernal": (kernal_w, kernal_h), "stride": (stride_w, stride_h)}

    return kernal_stride


class ImgSizeCNNArchitecture:

    def __init__(self):
        # Get the sizes of the images for each streamgage:
        with open(os.path.join(DATA_DIR, "gage_img_sizes.yaml"), "r") as f:
            gage_sizes = yaml.safe_load(f)
        with open(os.path.join(DATA_DIR, "avg_img_sizes.yaml"), "r") as f:
            avg_sizes = yaml.safe_load(f)
        for k, v in avg_sizes.items():
            gage_sizes[k]["avg"] = v
        self.img_sizes = gage_sizes

    def get_model(self, gage: str, n_days_precip=7, n_days_temp=7, n_swe=12,
                  n_et=1, enc_dense_dim=32, enc_num_heads=2, n_y=14,
                  hidden_dim=8, dropout=0.5, pooling: str = "avg"):
        """Create a new ImgSizeCNN model architecture with the specified
        parameters."""
        input_sizes = {k: v[gage] for k, v in self.img_sizes.items()}

        kernal_stride = get_kernal_stride(input_sizes)

        # Single image CNN inputs - dem:
        dem_inputs, dem_outputs = time_dist_cnn(
            1, "dem", hidden_dim,
            kernel_size=kernal_stride["dem"]["kernal"], strides=kernal_stride["dem"]["stride"],
            pooling=pooling, w=input_sizes["dem"][0], h=input_sizes["dem"][1],
        )

        # Multiple image CNN inputs - temp / precip / swe / et:
        et_inputs, et_outputs = time_dist_cnn(
            n_et, "et", hidden_dim,
            kernel_size=kernal_stride["et"]["kernal"], strides=kernal_stride["et"]["stride"],
            pooling=pooling, w=input_sizes["et"][0], h=input_sizes["et"][1],
        )
        temp_inputs, temp_outputs = time_dist_cnn(
            n_days_temp, "temp", hidden_dim,
            kernel_size=kernal_stride["temp"]["kernal"], strides=kernal_stride["temp"]["stride"],
            pooling=pooling, w=input_sizes["temp"][0], h=input_sizes["temp"][1],
        )
        precip_inputs, precip_outputs = time_dist_cnn(
            n_days_precip, "precip", hidden_dim,
            kernel_size=kernal_stride["precip"]["kernal"], strides=kernal_stride["precip"]["stride"],
            pooling=pooling, w=input_sizes["precip"][0], h=input_sizes["precip"][1],
        )
        swe_inputs, swe_outputs = time_dist_cnn(
            n_swe, "swe", hidden_dim,
            kernel_size=kernal_stride["swe"]["kernal"], strides=kernal_stride["swe"]["stride"],
            pooling=pooling, w=input_sizes["swe"][0], h=input_sizes["swe"][1],
        )

        # Concatenate CNN outputs:
        concat = tf.keras.layers.Concatenate(axis=1)(
            [dem_outputs, temp_outputs, precip_outputs, swe_outputs, et_outputs]
        )

        # Transformer encoder:
        encoder_outputs = TransformerEncoder(  # NOQA
            embed_dim=concat.shape[-1], dense_dim=enc_dense_dim, num_heads=enc_num_heads)(concat)
        x = layers.GlobalMaxPooling1D()(encoder_outputs)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(n_y, activation="linear")(x)

        # Create the model:
        transformer = keras.Model(
            [dem_inputs, temp_inputs, precip_inputs, swe_inputs, et_inputs],
            outputs
        )

        return transformer


class SpaceTimeTransformerArchitecture:

    def __init__(self):
        # Get the sizes of the images for each streamgage:
        with open(os.path.join(DATA_DIR, "gage_img_sizes.yaml"), "r") as f:
            gage_sizes = yaml.safe_load(f)
        with open(os.path.join(DATA_DIR, "avg_img_sizes.yaml"), "r") as f:
            avg_sizes = yaml.safe_load(f)
        for k, v in avg_sizes.items():
            gage_sizes[k]["avg"] = v
        self.img_sizes = gage_sizes

    def get_model(self, gage: str, n_days_precip=7, n_days_temp=7, n_swe=12,
                  n_et=1, enc_dense_dim=32, enc_num_heads=2, n_y=14,
                  hidden_dim=8, dropout=0.5, dropout_concat: bool = False,
                  pooling: str = "avg"):
        """Create a new SpaceTimeTransformerArchitecture model architecture with
        the specified parameters."""
        input_sizes = {k: v[gage] for k, v in self.img_sizes.items()}

        kernal_stride = get_kernal_stride(input_sizes)

        # Single image CNN inputs - dem:
        dem_inputs, dem_outputs = time_dist_cnn(
            1, "dem", hidden_dim,
            kernel_size=kernal_stride["dem"]["kernal"], strides=kernal_stride["dem"]["stride"],
            pooling=pooling, w=input_sizes["dem"][0], h=input_sizes["dem"][1],
        )

        # Multiple image CNN inputs - temp / precip / swe / et:
        et_inputs, et_outputs = time_dist_cnn(
            n_et, "et", hidden_dim,
            kernel_size=kernal_stride["et"]["kernal"], strides=kernal_stride["et"]["stride"],
            pooling=pooling, w=input_sizes["et"][0], h=input_sizes["et"][1],
        )
        temp_inputs, temp_outputs = time_dist_cnn(
            n_days_temp, "temp", hidden_dim,
            kernel_size=kernal_stride["temp"]["kernal"], strides=kernal_stride["temp"]["stride"],
            pooling=pooling, w=input_sizes["temp"][0], h=input_sizes["temp"][1],
        )
        precip_inputs, precip_outputs = time_dist_cnn(
            n_days_precip, "precip", hidden_dim,
            kernel_size=kernal_stride["precip"]["kernal"], strides=kernal_stride["precip"]["stride"],
            pooling=pooling, w=input_sizes["precip"][0], h=input_sizes["precip"][1],
        )
        swe_inputs, swe_outputs = time_dist_cnn(
            n_swe, "swe", hidden_dim,
            kernel_size=kernal_stride["swe"]["kernal"], strides=kernal_stride["swe"]["stride"],
            pooling=pooling, w=input_sizes["swe"][0], h=input_sizes["swe"][1],
        )

        # Concatenate CNN outputs:
        concat = tf.keras.layers.Concatenate(axis=1, name="concat")(
            [dem_outputs, temp_outputs, precip_outputs, swe_outputs, et_outputs],
        )
        # THIS IS A REALLY HEAVY REGULARIZER THAT MAY BE USEFUL FOR TRAINING A FINAL MODEL ON MANY EPOCHS:
        if dropout_concat:
            concat = layers.Dropout(dropout, name="concat_dropout")(concat)

        # Transformer encoder time dimension:
        encoder_outputs = TransformerEncoder(  # NOQA
            embed_dim=concat.shape[-1], dense_dim=enc_dense_dim, num_heads=enc_num_heads,
            name="time_transformer"
        )(concat)
        encoder_outputs = layers.Dropout(dropout, name="time_transformer_dropout")(encoder_outputs)

        # Swap the time-space axis:
        dim_swap = keras.layers.Permute((2, 1), input_shape=encoder_outputs.shape, name="spacetime_swap")(encoder_outputs)

        # Transformer encoder space dimension:
        space_encoder_outputs = TransformerEncoder(  # NOQA
            embed_dim=dim_swap.shape[-1], dense_dim=enc_dense_dim, num_heads=enc_num_heads,
            name="space_transformer"
        )(dim_swap)
        space_encoder_outputs = layers.Dropout(dropout, name="space_transformer_dropout")(space_encoder_outputs)

        flatten = layers.Flatten(name="flatten_transformer_outputs")(space_encoder_outputs)
        x = layers.Dropout(dropout, name="dropout_transformer_outputs")(flatten)

        dense1 = layers.Dense(512, activation="relu", name="dense1")(x)
        dropout_dense1 = layers.Dropout(dropout, name="dropout_dense1")(dense1)

        dense2 = layers.Dense(128, activation="relu", name="dense2")(dropout_dense1)
        dropout_dense2 = layers.Dropout(dropout, name="dropout_dense2")(dense2)

        outputs = layers.Dense(n_y, activation="linear", name="prediction")(dropout_dense2)

        transformer = keras.Model(
            [dem_inputs, temp_inputs, precip_inputs, swe_inputs, et_inputs],
            outputs
        )

        return transformer
