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


def build_input_pipeline(domain_files, dataset_size, batch_size, augment):
    d = tf.data.Dataset.list_files(domain_files).take(dataset_size)
    d = d.interleave(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    d = d.batch(batch_size, drop_remainder=True)

    if augment and "random_flip_left_right" in augment:
        preprocess = random_flip_left_right(preprocess_images)
    else:
        preprocess = preprocess_images

    d = d.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache()
    d = d.prefetch(tf.data.AUTOTUNE)
    return d
