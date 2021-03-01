import tensorflow as tf


def read_image(image):
    image = tf.io.decode_png(tf.io.read_file(image), channels=3)
    return tf.data.Dataset.from_tensors(image)


def normalize_image(image):
    image = tf.image.resize(image, [128, 128]) / 127.5 - 1
    return image


def build_input_pipeline(domain_files, dataset_size, batch_size, augment):
    d = tf.data.Dataset.list_files(domain_files).take(dataset_size)
    d = d.interleave(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    d = d.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    d = d.batch(batch_size, drop_remainder=True)

    if augment and "random_flip_left_right" in augment:
        augment_images = tf.image.random_flip_left_right
    else:
        augment_images = lambda x: x

    d = d.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE).cache()
    d = d.prefetch(tf.data.AUTOTUNE)
    return d
