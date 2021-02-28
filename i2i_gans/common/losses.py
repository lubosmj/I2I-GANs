import tensorflow as tf

from tensorflow.keras import losses

gan_loss_fn = losses.BinaryCrossentropy(reduction=losses.Reduction.NONE, from_logits=True)


def get_discriminator_loss(real_outputs, fake_outputs):
    real_loss = gan_loss_fn(tf.ones_like(real_outputs), real_outputs)
    fake_loss = gan_loss_fn(tf.zeros_like(fake_outputs), fake_outputs)
    return real_loss + fake_loss
