# -*- coding: utf-8 -*-
from mic_py_nn.trainers.base_train import BaseTrain
import numpy as np
import os
import tensorflow as tf


class RefineNetTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(RefineNetTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)
        self.max_iter = config.trainer.max_num_step
        self.print_step = config.trainer.print_step
        self.saver_step = config.trainer.save_model_frequency

        self.global_step = 0

        self.train_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), 'log_board', 'train'), self.sess.graph)
        # self.validation_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), 'log_validation', 'train_validation'),
        #                                               self.sess.graph)

    def train(self):

        for step in range(self.max_iter):
            loss, acc = self.train_step()

            if step % self.print_step == 0:
                print(
                    'step {} --- accuracy_1: {} --- accuracy_2: {} --- accuracy_3: {} --- loss: {}'.format(step, acc[0],
                                                                                                           acc[1],
                                                                                                           acc[2],
                                                                                                           loss))
            if step % self.config.trainer.validation_frequency == 0:
                val_loss = self.valid_step()
                print('step {} --- validation loss: {}'.format(step, val_loss))

            if step % self.saver_step == 0 and step != 0:
                self.model.saver.save(self.sess, os.path.join(self.config.checkpoint_dir, 'model_v1.1'), step)

            step += 1
            self.global_step += 1

    def train_step(self):
        sp, noise, mix, M, _ = self.train_data.next_batch()

        # sp = np.transpose(np.absolute(sp), (0, 2, 1))
        # noise = np.transpose(np.absolute(noise), (0, 2, 1))
        # mix = np.transpose(np.absolute(mix), (0, 2, 1))
        M = np.transpose(M, (0, 2, 1, 3))

        # mix_norm = (mix - mix.min()) / (mix.max() - mix.min())

        mix_0 = np.array([mix[i][0] for i in range(len(mix))])
        mix_1 = np.array([mix[i][1] for i in range(len(mix))])
        mix_2 = np.array([mix[i][2] for i in range(len(mix))])

        mix_0 = np.transpose(np.absolute(mix_0), (0, 2, 1))
        mix_1 = np.transpose(np.absolute(mix_1), (0, 2, 1))
        mix_2 = np.transpose(np.absolute(mix_2), (0, 2, 1))

        mix_norm_0 = (mix_0 - mix_0.min()) / (mix_0.max() - mix_0.min())
        mix_norm_1 = (mix_1 - mix_1.min()) / (mix_1.max() - mix_1.min())
        mix_norm_2 = (mix_2 - mix_2.min()) / (mix_2.max() - mix_2.min())

        feed_dict = {
            self.model.x_0: mix_norm_0,
            self.model.x_1: mix_norm_1,
            self.model.x_2: mix_norm_2,
            self.model.x_noise_orig: mix_0,
            self.model.y: M,
            self.model.keep_prob: 1,
            self.model.phase: True
        }

        loss, _, acc, summary = self.sess.run([self.model.loss, self.model.optimizer, self.model.accuracy,
                                               self.model.summary],
                                              feed_dict=feed_dict)
        self.train_writer.add_summary(summary, self.global_step)
        return loss, acc

    def valid_step(self):
        sp, noise, mix, M, _ = self.train_data.next_batch()

        # sp = np.transpose(np.absolute(sp), (0, 2, 1))
        # noise = np.transpose(np.absolute(noise), (0, 2, 1))
        # mix = np.transpose(np.absolute(mix), (0, 2, 1))
        M = np.transpose(M, (0, 2, 1, 3))

        # mix_norm = (mix - mix.min()) / (mix.max() - mix.min())

        mix_0 = np.array([mix[i][0] for i in range(len(mix))])
        mix_1 = np.array([mix[i][1] for i in range(len(mix))])
        mix_2 = np.array([mix[i][2] for i in range(len(mix))])

        mix_0 = np.transpose(np.absolute(mix_0), (0, 2, 1))
        mix_1 = np.transpose(np.absolute(mix_1), (0, 2, 1))
        mix_2 = np.transpose(np.absolute(mix_2), (0, 2, 1))

        mix_norm_0 = (mix_0 - mix_0.min()) / (mix_0.max() - mix_0.min())
        mix_norm_1 = (mix_1 - mix_1.min()) / (mix_1.max() - mix_1.min())
        mix_norm_2 = (mix_2 - mix_2.min()) / (mix_2.max() - mix_2.min())

        feed_dict = {
            self.model.x_0: mix_norm_0,
            self.model.x_1: mix_norm_1,
            self.model.x_2: mix_norm_2,
            self.model.x_noise_orig: mix_0,
            self.model.y: M,
            self.model.keep_prob: 1,
            self.model.phase: False
        }

        loss, _ = self.sess.run([self.model.loss, self.model.optimizer], feed_dict=feed_dict)
        return loss
