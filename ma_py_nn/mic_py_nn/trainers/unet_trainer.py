# -*- coding: utf-8 -*-
from mic_py_nn.trainers.base_train import BaseTrain
import numpy as np
import os


class UNetTrainer(BaseTrain):

    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(UNetTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)
        self.max_iter = config.trainer.max_num_step
        self.print_step = config.trainer.print_step
        self.saver_step = config.trainer.save_model_frequency

    def train(self):

        for step in range(self.max_iter):
            loss, acc = self.train_step()

            if step % self.print_step == 0:
                print('step {} --- accuracy_1: {} --- accuracy_2: {} --- accuracy_3: {} --- loss: {}'.format(step, acc[0], acc[1],
                                                                                                acc[2], loss))
            if step % self.config.trainer.validation_frequency == 0:
                val_loss = self.valid_step()
                print('step {} --- validation loss: {}'.format(step, val_loss))

            if step % self.saver_step == 0 and step != 0:
                self.model.saver.save(self.sess, os.path.join(self.config.checkpoint_dir, 'unet_model_v11'), step)

            '''
            if step % 15000 == 0 and step != 0 and step < 50000:
                self.model.learning_rate *= 0.5
            '''

            step += 1

    def train_step(self):
        sp, noise, mix, M, _ = self.train_data.next_batch()

        sp = np.transpose(np.absolute(sp), (0, 2, 1))
        noise = np.transpose(np.absolute(noise), (0, 2, 1))
        mix = np.transpose(np.absolute(mix), (0, 2, 1))
        M = np.transpose(M, (0, 2, 1, 3))

        mix_norm = (mix - mix.min()) / (mix.max() - mix.min())

        feed_dict = {
            self.model.x: mix_norm,
            self.model.x_noise_orig: mix,
            self.model.y: M,
            self.model.keep_prob: 1,
            self.model.phase: True
        }

        loss, _, acc = self.sess.run([self.model.loss, self.model.optimizer, self.model.accuracy], feed_dict=feed_dict)
        return loss, acc

    def valid_step(self):
        sp, noise, mix, M, _ = self.valid_data.next_batch()
        
        sp = np.transpose(np.absolute(sp), (0, 2, 1))
        noise = np.transpose(np.absolute(noise), (0, 2, 1))
        mix = np.transpose(np.absolute(mix), (0, 2, 1))
        M = np.transpose(M, (0, 2, 1, 3))

        mix_norm = (mix - mix.min()) / (mix.max() - mix.min())

        feed_dict = {
            self.model.x: mix_norm,
            self.model.x_noise_orig: mix,
            self.model.y: M,
            self.model.keep_prob: 1,
            self.model.phase: False
        }

        loss, _ = self.sess.run([self.model.loss, self.model.optimizer], feed_dict=feed_dict)
        return loss
