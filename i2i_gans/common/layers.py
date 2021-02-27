import tensorflow_addons as tfa

from tensorflow import keras


def conv2D(inputs, filters, use_spectral_norm=False, use_batch_norm=True):
    layer = keras.layers.Conv2D(
        filters=filters, kernel_size=4, strides=2, padding="same", use_bias=False
    )

    if use_spectral_norm:
        x = tfa.layers.SpectralNormalization(layer)(inputs)
    else:
        x = layer(inputs)

    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)

    x = keras.layers.LeakyReLU(0.2)(x)
    return x


def deconv2D(inputs, filters):
    x = keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=4, strides=2, padding="same", use_bias=False
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    return x
