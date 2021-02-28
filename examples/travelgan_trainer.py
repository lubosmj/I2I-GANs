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


class TraVeLGANImageSampler(callbacks.ImageSampler):
    def __init__(self, every_N_epochs, samples_dir, domain_A_dataset, travelgan):
        super().__init__(every_N_epochs, samples_dir)

        self.real_A = domain_A_dataset.unbatch().take(self.NUMBER_OF_SAMPLES).batch(1)
        self.travelgan = travelgan

    def images_generator(self):
        for inputs in self.real_A:
            outputs = self.travelgan.generator(inputs)
            yield inputs[0], outputs[0]


parser = TraVeLGANParser()
args = parser.parse_args()

checkpoint_filepath = os.path.join(args.checkpoints_dir, "travelgan_checkpoints.{epoch:03d}")
every_N_epochs = (args.dataset_size // args.batch_size) * args.checkpoints_freq
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_freq=every_N_epochs
)

train_A = datasets.build_input_pipeline(
    os.path.join(args.domain_A, "*.*"), args.dataset_size, args.batch_size, args.augment
)
train_B = datasets.build_input_pipeline(
    os.path.join(args.domain_B, "*.*"), args.dataset_size, args.batch_size, args.augment
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
