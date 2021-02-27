import functools

import tensorflow as tf


def read_image(image):
    image = tf.io.decode_png(tf.io.read_file(image), channels=3)
    return tf.data.Dataset.from_tensors(image)


def preprocess_images(images):
    images = tf.image.resize(images, [128, 128]) / 127.5 - 1
    return images


def random_flip_left_right(func):
    @functools.wraps(func)
    def wrapper(images):
        images = func(images)
        images = tf.image.random_flip_left_right(images)
        return images
    return wrapper
