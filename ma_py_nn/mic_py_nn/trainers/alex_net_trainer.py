import tensorflow as tf
import numpy as np
import os

from mic_py_nn.trainers.base_train import BaseTrain


np.set_printoptions(threshold=np.nan)


class AlexNetTrainer(BaseTrain):

    def __init__(self, sess, model, train_data, valid_data, config, logger):

        super(AlexNetTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)

        self.max_num_step = config.trainer.max_num_step

        self.batch_size = config.batcher.batch_size
        self.print_step = config.trainer.print_step
        self.valid_fr = config.trainer.validation_frequency
        self.saver_step = config.trainer.save_model_frequency

        self.valid_log = None
        self.global_step = 0

        # self.train_writer = tf.summary.FileWriter(os.path.join(config.summary_dir, 'log', 'train'), self.sess.graph)
        # self.validation_writer = tf.summary.FileWriter(os.path.join(config.summary_dir, 'log_validation',
        #                                                            'train_validation'), self.sess.graph)

    def train(self, save_valid_res=False):

        for step in range(self.max_num_step):

            loss, acc, pr_tup = self.train_step()

            if step % self.print_step == 0:
                print('step {} --- accuracy: {} --- loss: {}'.format(step, acc, loss))
                # print(pr_tup)

            if step % self.valid_fr == 0:
                val_loss, val_acc = self.valid_step()
                print('validation accuracy: {} --- validation loss: {}'.format(val_acc, val_loss))

            if step % self.saver_step == 0:
                self.model.saver.save(self.sess, os.path.join(self.config.checkpoint_dir, 'crnn'), self.global_step)

    def train_step(self):
        sp, mus, mix, mask = self.train_data.next_batch()

        mix = np.transpose(np.absolute(mix), (0, 2, 1))
        batch_size, _, context_len = mix.shape

        feed_dict = {
            self.model.x: mix,
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

        feed_dict = {
            self.model.x: mix,
            self.model.y: mask,
            self.model.keep_prob: 1
        }

        pred, loss, acc = self.sess.run([self.model.prediction, self.model.loss, self.model.accuracy],
                                        feed_dict=feed_dict)

        return loss, acc
