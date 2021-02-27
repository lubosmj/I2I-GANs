import os
import tensorflow as tf

from contextlib import ExitStack

from tensorflow import keras

from i2i_gans import parsers, datasets, TraVeLGAN


class TraVeLGANParser(parsers.Parser):
    def init_train_subparser(self):
        super().init_train_subparser()

        self.train.add_argument("--siamese_dim", type=int, default=1000)
        self.train.add_argument("--lambda_travel", type=float, default=10.0)
        self.train.add_argument("--lambda_margin", type=float, default=10.0)
        self.train.add_argument("--lambda_gan", type=float, default=1.0)


def build_input_pipeline(domain_files, dataset_size, batch_size, augment):
    d = tf.data.Dataset.list_files(domain_files).take(dataset_size)
    d = d.interleave(datasets.read_image, num_parallel_calls=tf.data.AUTOTUNE)
    d = d.batch(batch_size, drop_remainder=True)

    if augment and "random_flip_left_right" in augment:
        preprocess = datasets.random_flip_left_right(datasets.preprocess_images)
    else:
        preprocess = datasets.preprocess_images

    d = d.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache()
    d = d.prefetch(tf.data.AUTOTUNE)
    return d


parser = TraVeLGANParser()
args = parser.parse_args()

checkpoint_filepath = os.path.join(args.checkpoints_dir, "travelgan_checkpoints.{epoch:03d}")
every_N_epochs = (args.dataset_size // args.batch_size) * args.checkpoints_freq
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_freq=every_N_epochs
)

train_A = build_input_pipeline(
    os.path.join(args.domain_A, "*.*"), args.dataset_size, args.batch_size, args.augment
)
train_B = build_input_pipeline(
    os.path.join(args.domain_B, "*.*"), args.dataset_size, args.batch_size, args.augment
)
dataset = tf.data.Dataset.zip((train_A, train_B))

strategy = tf.distribute.MirroredStrategy()

with ExitStack() as stack:
    if args.parallel:
        stack.enter_context(strategy.scope())

    travelgan = TraVeLGAN(**vars(args))
    travelgan.compile()

travelgan.fit(
    dataset, epochs=args.epochs, batch_size=args.batch_size, callbacks=[model_checkpoint_callback]
)
