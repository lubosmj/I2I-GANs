import os
import tensorflow as tf

from contextlib import ExitStack

from tensorflow import keras

from i2i_gans import parsers, datasets, callbacks, GcGAN


class GcGANParser(parsers.Parser):
    def init_train_subparser(self):
        super().init_train_subparser()

        self.train.add_argument("--lambda_a2b", type=float, default=10.0)
        self.train.add_argument("--lambda_gc", type=float, default=2.0)
        self.train.add_argument("--lambda_ident", type=float, default=0.3)
        self.train.add_argument("--lambda_gan", type=float, default=1.0)


class GcGANImageSampler(callbacks.ImageSampler):
    def __init__(self, every_N_epochs, samples_dir, domain_A_dataset, gcgan):
        super().__init__(every_N_epochs, samples_dir)

        self.real_A = domain_A_dataset.unbatch().take(self.NUMBER_OF_SAMPLES).batch(1)
        self.gcgan = gcgan

    def images_generator(self):
        for inputs in self.real_A:
            outputs = self.gcgan.generator(inputs)
            yield inputs[0], outputs[0]


parser = GcGANParser()
args = parser.parse_args()

checkpoint_filepath = os.path.join(args.checkpoints_dir, "gcgan_checkpoints.{epoch:03d}")
every_N_epochs = (args.dataset_size // args.batch_size) * args.checkpoints_freq
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_freq=every_N_epochs
)

train_A = datasets.build_input_pipeline(
    args.domain_A_files, args.dataset_size, args.batch_size, args.augment, cache=False
)
train_B = datasets.build_input_pipeline(
    args.domain_B_files, args.dataset_size, args.batch_size, args.augment, cache=False
)
dataset = tf.data.Dataset.zip((train_A, train_B))

strategy = tf.distribute.MirroredStrategy()

with ExitStack() as stack:
    if args.parallel:
        stack.enter_context(strategy.scope())

    discogan = GcGAN(**vars(args))
    discogan.compile()

discogan_sampler = GcGANImageSampler(args.samples_freq, args.samples_dir, train_A, discogan)

discogan.fit(
    dataset,
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks=[model_checkpoint_callback, discogan_sampler],
)
