"""CNN model architectures."""

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models  # NOQA


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
                  inputs=None):
    """Create time-distributed CNN with GAP."""
    if inputs is None:
        inputs = keras.Input(shape=(observation_period, None, None, 1),
                             batch_size=None, name=f"{band_name}_inputs")
    conv_2d_layer = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                                  strides=strides, activation=activation)
    x = layers.TimeDistributed(conv_2d_layer, name=f"{band_name}_conv2d")(inputs)
    pooling_layer = layers.GlobalAveragePooling2D(data_format="channels_last", keepdims=False)
    outputs = layers.TimeDistributed(pooling_layer, name=f"{band_name}_global_pooling")(x)
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
                  n_swe=12,
                  enc_embed_dim=16, enc_dense_dim=32, enc_num_heads=2,
                  dec_embed_dim=16, dec_dense_dim=32, dec_num_heads=2,
                  n_y=14, hidden_dim=16, dropout=0.5):
        """Create a new CNN model architecture with the specified parameters."""
        # Single image CNN inputs - dem / et:
        dem_inputs, dem_outputs = time_dist_cnn(
            1, "dem", hidden_dim, kernel_size=dem_kernal, strides=dem_strides)
        et_inputs, et_outputs = time_dist_cnn(
            1, "et", hidden_dim, kernel_size=et_kernal, strides=et_strides)

        # Multiple image CNN inputs - temp / precip / swe:
        temp_inputs, temp_outputs = time_dist_cnn(
            n_days_temp, "temp", hidden_dim, kernel_size=temp_kernal, strides=temp_strides)
        precip_inputs, precip_outputs = time_dist_cnn(
            n_days_precip, "precip", hidden_dim, kernel_size=precip_kernal, strides=precip_strides)
        swe_inputs, swe_outputs = time_dist_cnn(
            n_swe, "swe", hidden_dim, kernel_size=swe_kernal, strides=swe_strides)

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
                  n_swe=12,
                  enc_embed_dim=16, enc_dense_dim=32, enc_num_heads=2,
                  dec_embed_dim=16, dec_dense_dim=32, dec_num_heads=2,
                  n_y=14, hidden_dim=16, dropout=0.5, cnn_activation="relu"):
        """Create a new CNN model architecture with the specified parameters."""

        # Create the max-pooling layers first:
        maxpool_dem_inputs, maxpool_dem_outputs = self.dem_cnn(
            observation_period=1, activation=cnn_activation)
        maxpool_et_inputs, maxpool_et_outputs = self.et_cnn(
            observation_period=1, activation=cnn_activation)
        maxpool_swe_inputs, maxpool_swe_outputs = self.swe_cnn(
            observation_period=n_swe, activation=cnn_activation)

        # Create the GAP CNNs:
        gap_dem_inputs, gap_dem_outputs = time_dist_cnn(
            1, "dem", hidden_dim, kernel_size=kernal, strides=strides, inputs=maxpool_dem_outputs)
        gap_et_inputs, gap_et_outputs = time_dist_cnn(
            1, "et", hidden_dim, kernel_size=kernal, strides=strides, inputs=maxpool_et_outputs)
        gap_temp_inputs, gap_temp_outputs = time_dist_cnn(
            n_days_temp, "temp", hidden_dim, kernel_size=kernal, strides=strides)
        gap_precip_inputs, gap_precip_outputs = time_dist_cnn(
            n_days_precip, "precip", hidden_dim, kernel_size=kernal, strides=strides)
        gap_swe_inputs, gap_swe_outputs = time_dist_cnn(
            n_swe, "swe", hidden_dim, kernel_size=kernal, strides=strides, inputs=maxpool_swe_outputs)

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