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


class GAPTransArchitecture:
    """CNN architecture using Global Average Pooling and Trasnformer."""

    def __init__(self):
        pass

    @staticmethod
    def time_dist_cnn(observation_period: int, band_name: str,
                      n_filters: int = 16, kernel_size: tuple = (2, 2),
                      strides: tuple = (1, 1), activation: str = "relu"):
        """Create time-distributed CNN with GAP."""
        inputs = keras.Input(shape=(observation_period, None, None, 1),
                             batch_size=None, name=f"{band_name}_inputs")
        conv_2d_layer = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                                      strides=strides, activation=activation)
        x = layers.TimeDistributed(conv_2d_layer, name=f"{band_name}_conv2d")(inputs)
        pooling_layer = layers.GlobalAveragePooling2D(data_format="channels_last", keepdims=False)
        outputs = layers.TimeDistributed(pooling_layer, name=f"{band_name}_global_pooling")(x)
        return inputs, outputs

    def get_model(self, dem_kernal=(5, 5), dem_strides=(1, 1),
                  et_kernal=(4, 4), et_strides=(1, 1),
                  temp_kernal=(2, 2), temp_strides=(1, 1),
                  precip_kernal=(2, 2), precip_strides=(1, 1),
                  swe_kernal=(5, 5), swe_strides=(1, 1),
                  n_days_precip=7, n_days_temp=7,
                  swe_days_relative=range(7, 85, 7),
                  enc_embed_dim=16, enc_dense_dim=32, enc_num_heads=2,
                  dec_embed_dim=16, dec_dense_dim=32, dec_num_heads=2,
                  n_y=14, hidden_dim=16, dropout=0.5):
        """Create a new CNN model architecture with the specified parameters."""
        # Single image CNN inputs - dem / et:
        dem_inputs, dem_outputs = self.time_dist_cnn(
            1, "dem", hidden_dim, kernel_size=dem_kernal, strides=dem_strides)
        et_inputs, et_outputs = self.time_dist_cnn(
            1, "et", hidden_dim, kernel_size=et_kernal, strides=et_strides)

        # Multiple image CNN inputs - temp / precip / swe:
        temp_inputs, temp_outputs = self.time_dist_cnn(
            n_days_temp, "temp", hidden_dim, kernel_size=temp_kernal, strides=temp_strides)
        precip_inputs, precip_outputs = self.time_dist_cnn(
            n_days_precip, "precip", hidden_dim, kernel_size=precip_kernal, strides=precip_strides)
        swe_inputs, swe_outputs = self.time_dist_cnn(
            len(list(swe_days_relative)), "swe", hidden_dim, kernel_size=swe_kernal, strides=swe_strides)

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
