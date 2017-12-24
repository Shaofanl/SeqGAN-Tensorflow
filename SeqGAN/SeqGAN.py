import tensorflow as tf
from SeqGAN.Discriminator import Discriminator
from SeqGAN.Generator import Generator

import os
import shutil


class SeqGAN(object):
    def __init__(self):
        self.G = Generator()
        self.D = Discriminator()

    def train(self, sampler,
              pretrain_g_nb_epoch,
              tensorboard_dir='tensorboard/'):
        if os.path.exists:
            shutil.rmtree(tensorboard_dir)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(tensorboard_dir, sess.graph())

            sess.run(tf.global_variables_initializer())

            print 'pretraining...'
            for epoch in range(pretrain_g_nb_epoch):

