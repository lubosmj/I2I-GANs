import os
import tensorflow as tf

from contextlib import ExitStack

from tensorflow import keras

from i2i_gans import parsers, datasets, callbacks, TraVeLGAN


class TraVeLGANParser(parsers.Parser):
    def init_train_subparser(self):
        super().init_train_subparser()

        self.train.add_argument("--siamese_dim", type=int, default=1000)
        self.train.add_argument("--lambda_travel", type=float, default=10.0)
        self.train.add_argument("--lambda_margin", type=float, default=10.0)
        self.train.add_argument("--lambda_gan", type=float, default=1.0)

        self.train.add_argument("--second_domain_B_files")


class TraVeLGANImageSampler(callbacks.ImageSampler):
    def __init__(self, every_N_epochs, samples_dir, domain_A_dataset, travelgan):
        super().__init__(every_N_epochs, samples_dir)

        self.real_A = domain_A_dataset.unbatch().take(self.NUMBER_OF_SAMPLES).batch(1)
        self.travelgan = travelgan

    def images_generator(self):
        for inputs in self.real_A:
            outputs = self.travelgan.generator(inputs)
            yield inputs[0], outputs[0]


def preprocessing_model(input_shape=(218, 178, 3), image_size=(128, 128)):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.ZeroPadding2D(padding=((0,25), (0,0)))(inputs)
    x = keras.layers.experimental.preprocessing.CenterCrop(*image_size)(x)
    x = keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)(x)
    return keras.Model(inputs=inputs, outputs=x)


def build_cropped_celeba_input_pipeline(domain_files, dataset_size, batch_size, augment):
    d = tf.data.Dataset.list_files(domain_files).take(dataset_size)
    d = d.interleave(datasets.read_image, num_parallel_calls=tf.data.AUTOTUNE)
    d = d.batch(batch_size, drop_remainder=True)
    d = d.map(preprocessing_model(), num_parallel_calls=tf.data.AUTOTUNE)

    if augment and "random_flip_left_right" in augment:
        augment_images = tf.image.random_flip_left_right
    else:
        augment_images = lambda x: x

    d = d.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE)
    d = d.cache()

    d = d.prefetch(tf.data.AUTOTUNE)
    return d


parser = TraVeLGANParser()
args = parser.parse_args()

checkpoint_filepath = os.path.join(args.checkpoints_dir, "travelgan_checkpoints.{epoch:03d}")
every_N_epochs = (args.dataset_size // args.batch_size) * args.checkpoints_freq
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_freq=every_N_epochs
)

train_A = build_cropped_celeba_input_pipeline(
    args.domain_A_files, args.dataset_size, args.batch_size, args.augment
)
if args.second_domain_B_files:
    train_B = datasets.build_input_pipeline(
        args.domain_B_files, args.dataset_size, args.batch_size, args.augment, cache=False
    )
    train_B2 = datasets.build_input_pipeline(
        args.second_domain_B_files, args.dataset_size, args.batch_size, args.augment, cache=False
    )
    train_B = tf.data.experimental.sample_from_datasets([train_B, train_B2])
else:
    train_B = datasets.build_input_pipeline(
        args.domain_B_files, args.dataset_size, args.batch_size, args.augment
    )

dataset = tf.data.Dataset.zip((train_A, train_B))

strategy = tf.distribute.MirroredStrategy()

with ExitStack() as stack:
    if args.parallel:
        stack.enter_context(strategy.scope())

    travelgan = TraVeLGAN(**vars(args))
    travelgan.compile()

travelgan_sampler = TraVeLGANImageSampler(args.samples_freq, args.samples_dir, train_A, travelgan)

travelgan.fit(
    dataset,
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks=[model_checkpoint_callback, travelgan_sampler],
)
