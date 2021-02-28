import os
import tensorflow as tf

from contextlib import ExitStack

from tensorflow import keras

from i2i_gans import parsers, datasets, callbacks, DiscoGAN


class DiscoGANParser(parsers.Parser):
    def init_train_subparser(self):
        super().init_train_subparser()

        self.train.add_argument("--lambda_reconstr", type=float, default=1.0)
        self.train.add_argument("--lambda_fml", type=float, default=0.9)
        self.train.add_argument("--lambda_gan", type=float, default=0.1)


class DiscoGANImageSampler(callbacks.ImageSampler):
    def __init__(self, every_N_epochs, samples_dir, domain_A_dataset, discogan):
        super().__init__(every_N_epochs, samples_dir)

        self.real_A = domain_A_dataset.unbatch().take(self.NUMBER_OF_SAMPLES).batch(1)
        self.discogan = discogan

    def images_generator(self):
        for inputs in self.real_A:
            outputs = self.discogan.gen_B(inputs)
            yield inputs[0], outputs[0]


parser = DiscoGANParser()
args = parser.parse_args()

checkpoint_filepath = os.path.join(args.checkpoints_dir, "discogan_checkpoints.{epoch:03d}")
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

    discogan = DiscoGAN(**vars(args))
    discogan.compile()

discogan_sampler = DiscoGANImageSampler(args.samples_freq, args.samples_dir, train_A, discogan)

discogan.fit(
    dataset,
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks=[model_checkpoint_callback, discogan_sampler],
)
