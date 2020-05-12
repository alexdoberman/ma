import tensorflow as tf
import soundfile as sf
import numpy as np
import os

from mic_py_nn.trainers.base_train import BaseTrain


np.set_printoptions(threshold=np.nan)


class TDCNNTrainer(BaseTrain):

    def __init__(self, sess, model, train_data, valid_data, config, logger, unet_model, unet_session):

        super(TDCNNTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)

        self.config = config

        self.max_num_step = config.trainer.max_num_step

        self.batch_size = config.batcher.batch_size
        self.print_step = config.trainer.print_step
        self.valid_fr = config.trainer.validation_frequency
        self.saver_step = config.trainer.save_model_frequency

        self.unet_model = unet_model
        self.unet_session = unet_session

        self.valid_log = None
        self.global_step = 0

        # self.train_writer = tf.summary.FileWriter(os.path.join(config.summary_dir, 'log', 'train'), self.sess.graph)
        # self.validation_writer = tf.summary.FileWriter(os.path.join(config.summary_dir, 'log_validation',
        #                                                            'train_validation'), self.sess.graph)

    def train(self, save_valid_res=False):
        current_mean_accuracy = 0
        current_mean_loss = 0
        for step in range(self.max_num_step):

            loss, acc, pr_tup = self.train_step()

            current_mean_accuracy += acc
            current_mean_loss += loss

            if step % self.print_step == 0 and step != 0:
                current_mean_accuracy /= self.print_step
                current_mean_loss /= self.print_step

                print('step: {}-{} --- mean_accuracy: {} --- mean_loss: {}'.format(step-self.print_step, step,
                                                                                   current_mean_accuracy,
                                                                                   current_mean_loss))
                print('step {} --- accuracy: {} --- loss: {}'.format(step, acc, loss))
                current_mean_loss = 0
                current_mean_accuracy = 0
                # print(pr_tup)

            if step % self.valid_fr == 0 and step != 0:
                val_loss, val_acc = self.valid_step()
                print('validation accuracy: {} --- validation loss: {}'.format(val_acc, val_loss))

                '''
                real_data_mse = self.mic_valid()
                print('validation on real data!')
                for key, value in real_data_mse.items():
                    print('snr: {}, mse: {}, f1: {}'.format(key, value[0], value[1]))
                '''

            if step % self.saver_step == 0:
                self.model.saver.save(self.sess, os.path.join(self.config.checkpoint_dir, 'crnn'), self.global_step)

    def train_step(self):
        sp, mus, mix, mask = self.train_data.next_batch()

        mix = np.transpose(np.absolute(mix), (0, 2, 1))

        mix_norm = (mix - mix.min()) / (mix.max() - mix.min())
        feed_dict = {
            self.unet_model.x: mix_norm,
            self.unet_model.keep_prob: 1,
            self.unet_model.phase: True
        }

        unet_prediction = self.unet_session.run(self.unet_model.prediction,
                                                feed_dict=feed_dict)

        speech_mask = unet_prediction[:, :, :, 0]

        batch_size, _, context_len = mix.shape

        feed_dict = {
            self.model.x: speech_mask,
            self.model.y: mask,
            self.model.keep_prob: 1
        }

        pred, loss, _, acc = self.sess.run([self.model.prediction,
                                            self.model.loss, self.model.optimize, self.model.accuracy],
                                           feed_dict=feed_dict)

        self.global_step += 1
        return loss, acc, (np.reshape(pred[0], newshape=context_len), mask[0])

    def valid_step(self):

        sp, mus, mix, mask = self.train_data.next_batch()

        mix = np.transpose(np.absolute(mix), (0, 2, 1))

        mix_norm = (mix - mix.min()) / (mix.max() - mix.min())
        feed_dict = {
            self.unet_model.x: mix_norm,
            self.unet_model.keep_prob: 1,
            self.unet_model.phase: True
        }

        unet_prediction = self.unet_session.run(self.unet_model.prediction,
                                                feed_dict=feed_dict)

        speech_mask = unet_prediction[:, :, :, 0]

        feed_dict = {
            self.model.x: speech_mask,
            self.model.y: mask,
            self.model.keep_prob: 1
        }

        pred, loss, acc = self.sess.run([self.model.prediction, self.model.loss, self.model.accuracy],
                                        feed_dict=feed_dict)

        return loss, acc
