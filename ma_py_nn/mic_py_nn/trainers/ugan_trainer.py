# -*- coding: utf-8 -*-
from mic_py_nn.trainers.base_train import BaseTrain
import numpy as np
import os
import tensorflow as tf


class UGANTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(UGANTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)
        self.config = config
        self.max_iter = config.trainer.max_num_step
        self.print_step = config.trainer.print_step
        self.discriminator_step_num = config.trainer.discriminator_step_num
        self.generator_step_num = config.trainer.generator_step_num
        self.saver_step = config.trainer.save_model_frequency
        self.train_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), 'log', 'train'), self.sess.graph)
        self.validation_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), 'log_validation', 'train_validation'),
                                                       self.sess.graph)
        self.global_step = 0

    def train(self):

        saver = tf.train.Saver(max_to_keep=None)

        unet_it = 10000
        for step in range(unet_it):

            unet_loss, _, unet_accuracy = self.sess.run([self.model.unet_loss, self.model.unet_optimize,
                                                         self.model.gan_accuracy],
                                                        feed_dict=self.next_dict(self.train_data, True))
            if step % self.print_step == 0:
                print(
                    'step {} --- generator_loss: {} --- accuracy: {}'.format(step, unet_loss, unet_accuracy))

        # save pretrained generator model
        saver.save(self.sess, os.path.join(self.config.checkpoint_dir, 'gan_model_v6_unet'), 0)

        for step in range(self.max_iter):
            gen_loss, dis_loss, accuracy = self.train_step()

            if step % self.print_step == 0:
                print(
                    'step {} --- generator_loss: {} --- discriminator_loss: {} --- accuracy: {}'.format(step, gen_loss,
                                                                                                        dis_loss,
                                                                                                        accuracy))

            if step % self.config.trainer.validation_frequency == 0:
                gen_loss, accuracy = self.valid_step()
                print(
                    'step {} --- generator_loss: {} --- accuracy {}'.format(step, gen_loss, accuracy))

            """
            unet_loss, unet_accuracy = self.sess.run([self.model.unet_loss, self.model.unet_accuracy],
                                                        feed_dict=self.next_dict(self.train_data))
            if step % self.print_step == 0:
                print(
                    'step {} --- generator_loss: {} --- accuracy: {}'.format(step, unet_loss, unet_accuracy))
            """

            if step % self.saver_step == 0 and step != 0:
                saver.save(self.sess, os.path.join(self.config.checkpoint_dir, 'gan_model_v6'), step)

            step += 1

    def train_step(self):
        
        gen_loss = 0
        accuracy = 0
        
        # generator step
        for _ in range(self.generator_step_num):
            gen_loss, _, accuracy, summary, _ = self.sess.run([self.model.gen_COM_loss,
                                                               self.model.gen_COM_optimize,
                                                               self.model.gan_accuracy, self.model.summary,
                                                               self.model.gen_loss],
                                                              feed_dict=self.next_dict(self.train_data))
            self.train_writer.add_summary(summary, self.global_step)
            self.global_step += 1

        av_dis_loss = 0
        # discriminator step
        for _ in range(self.discriminator_step_num):

            dis_loss, _, dis_out, summary = self.sess.run([self.model.dis_loss, self.model.dis_optimize,
                                                           self.model.dump_discriminator_output, self.model.summary],
                                                          feed_dict=self.next_dict(self.train_data))
            # print(dis_out)
            self.train_writer.add_summary(summary, self.global_step)
            self.global_step += 1
            av_dis_loss += dis_loss

        av_dis_loss /= self.discriminator_step_num

        return gen_loss, av_dis_loss, accuracy

    def valid_step(self):
        # generator step

        gen_loss, accuracy = self.sess.run([self.model.gen_COM_loss,
                                               self.model.gan_accuracy],
                                              feed_dict=self.next_dict(self.valid_data, phase=False))

        return gen_loss, accuracy

    def next_dict(self, data_stor, pretrain=False, phase=True):
        sp, noise, mix, M, _ = data_stor.next_batch()

        sp = np.transpose(np.absolute(sp), (0, 2, 1))
        sp = sp[:, :, :, np.newaxis]
        noise = np.transpose(np.absolute(noise), (0, 2, 1))
        mix = np.transpose(np.absolute(mix), (0, 2, 1))

        # take only speech mask
        # M = np.transpose(M, (0, 2, 1, 3))[:, :, :, 0]
        # M = M[:, :, :, np.newaxis]
        M = np.transpose(M, (0, 2, 1, 3))
        # mix_norm = (mix - mix.min()) / (mix.max() - mix.min())

        feed_dict = {
            self.model.x: mix,
            self.model.x_noise_orig: mix,
            self.model.y: M,
            self.model.keep_prob: 1,
            self.model.phase: phase,
            self.model.pretrain: pretrain
        }

        return feed_dict
