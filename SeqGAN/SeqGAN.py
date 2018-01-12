import tensorflow as tf
import numpy as np
from .Discriminator import Discriminator
from .Generator import Generator

import os
import shutil


class SeqGAN(object):
    def __init__(
        self,
        batch_size,
        seq_len,
        vocab_size,
        start_token,
        # generator
        g_emb_dim,
        g_hidden_dim,
        # discriminator
        d_emb_dim,
        d_filter_sizes,
        d_num_filters,
        # others
        log_generation
    ):
        self.generator = Generator(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            emb_dim=g_emb_dim,
            hidden_dim=g_hidden_dim,
            start_token=start_token,
        )
        self.discriminator = Discriminator(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            emb_size=d_emb_dim,
            filter_sizes=d_filter_sizes,
            num_filters=d_num_filters
        )
        self.log_generation = log_generation

    def train(self, sampler,
              evaluator=None, evaluate=False,
              total_epochs=200,
              pretrain_g_epochs=1000,
              pretrain_d_epochs=50,
              tensorboard_dir='tensorboard/'):
        # if os.path.exists(tensorboard_dir):
        #     shutil.rmtree(tensorboard_dir)
        # os.mkdir(tensorboard_dir)

        gen, dis = self.generator, self.discriminator
        batch_size = gen.batch_size

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

            sess.run(tf.global_variables_initializer())

            print 'pretraining Generator ...'
            for epoch in range(pretrain_g_epochs):
                print 'pretrain g epoch', epoch
                summary = gen.pretrain(sess, sampler(batch_size))
                writer.add_summary(summary, epoch)

                if evaluate and evaluator is not None:
                    evaluator(gen.generate(sess), epoch)

            print 'pretraining Discriminator ...'
            for epoch in range(pretrain_d_epochs):
                fake_samples = gen.generate(sess)
                real_samples = sampler(batch_size)
                samples = np.concatenate([fake_samples, real_samples])
                labels = np.concatenate([np.zeros((batch_size,)),
                                         np.ones((batch_size,))])
                for _ in range(3):
                    indices = np.random.choice(
                        len(samples), size=(batch_size,), replace=False)
                    dis.train(sess, samples[indices], labels[indices])

            print 'Start Adversarial Training ...'
            for epoch in range(total_epochs):
                print 'epoch', epoch
                for _ in range(1):
                    fake_samples = gen.generate(sess)
                    rewards = gen.get_reward(sess, fake_samples, 16, dis)
                    summary = gen.train(sess, fake_samples, rewards)
                    # np.set_printoptions(linewidth=np.inf,
                    #                     precision=3)
                    # print rewards.mean(0)
                writer.add_summary(summary, epoch)

                for _ in range(5):
                    fake_samples = gen.generate(sess)
                    real_samples = sampler(batch_size)
                    samples = np.concatenate([fake_samples, real_samples])
                    labels = np.concatenate([np.zeros((batch_size,)),
                                             np.ones((batch_size,))])
                    for _ in range(3):
                        indices = np.random.choice(
                            len(samples), size=(batch_size,), replace=False)
                        summary = dis.train(sess, samples[indices],
                                            labels[indices])
                writer.add_summary(summary, epoch)

                if self.log_generation:
                    summary = sess.run(
                        gen.image_summary,
                        feed_dict={gen.given_tokens: real_samples})
                    writer.add_summary(summary, epoch)

                if evaluate and evaluator is not None:
                    evaluator(gen.generate(sess), pretrain_g_epochs+epoch)

                np.save('generation', gen.generate(sess))
